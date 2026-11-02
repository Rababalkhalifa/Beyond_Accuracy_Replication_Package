#!/usr/bin/env python3
"""RQ2 focused contrasts with cluster-aware, cross-seed synchronized bootstrap.

Reuses the audited prediction loader from Temporal_coverage_rankings.ipynb.
For each contrast, a unique evaluation document (digest + gold label) is a
bootstrap cluster across all five seeds, and across periods when repeated.
Each cluster receives one multinomial count shared by every occurrence.
Model comparisons remain paired on matched examples. Bootstrap only describes
resampling of observed document clusters conditional on fitted models, seeds,
periods, and cohorts; it is not evidence of out-of-time generalizability.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import numpy as np
import rq2_paired_uncertainty as prior

CONTRASTS=[]
for setting in ('Yelp n500','Yelp n1000'):
    for period in ('within','short','long'):
        CONTRASTS.append((setting,period,'phi_minus_qwen_at_80',
                          (period,'phi3.5:latest',80),(period,'qwen2.5:3b-instruct',80)))
for period in ('within','short','long'):
    CONTRASTS.append(('Climate n500',period,'qwen_80_minus_100',
                      (period,'qwen2.5:3b-instruct',80),(period,'qwen2.5:3b-instruct',100)))
for period in ('short','long'):
    CONTRASTS.append(('Climate n500',period,'distilbert_period_minus_reference_at_80',
                      (period,'distilbert',80),('within','distilbert',80)))


def build_index(core,source):
    """Keep original matched-ID verification, plus underlying document keys."""
    import zipfile
    paths={}
    for stem in core['STEMS'].values():
        matches=list(Path(source).rglob(stem+'*.zip'))
        if len(matches)!=1:
            raise ValueError(f'Expected exactly one {stem}*.zip under {source}; found {len(matches)}; use one archive directory')
        paths[stem]=matches[0]
    archives={key:zipfile.ZipFile(paths[stem]) for key,stem in core['STEMS'].items()}
    needed={(setting,period,seed,model)
            for setting,_,_,a,b in CONTRASTS
            for period,model,pct in (a,b)
            for seed in core['SEEDS']}
    raw={}
    try:
        for setting,period,seed,model in sorted(needed):
            rows=core['prediction_rows'](archives,setting,model,seed,period)
            gold,pred,digest=(('true_label','predicted_label','text_sha256') if setting.startswith('Climate')
                             else ('gold_label','pred_label','text_sha1'))
            ids=[];vals=[]
            for r in rows:
                token=(r['example_id'],r[digest],r[gold])
                if not r[digest] or r[pred] == '' or r.get('status','ok')!='ok':
                    raise ValueError(f'Incomplete evaluation row: {(setting,period,seed,model)}')
                ids.append(token)
                vals.append((float(r['confidence']),float(r[gold]!=r[pred])))
            if len(set(ids))!=len(ids):
                raise ValueError(f'Duplicate identities within {(setting,period,seed,model)}')
            scores=np.asarray(vals,dtype=float)
            if not np.isfinite(scores).all() or np.any((scores[:,0]<0)|(scores[:,0]>1)):
                raise ValueError('Bad confidence or correctness values')
            raw[setting,period,seed,model]=dict(zip(ids,vals))
        pools={}
        for setting,period,seed,_ in sorted(needed):
            k=(setting,period,seed)
            if k in pools: continue
            models=sorted(x[3] for x in needed if x[:3]==k)
            identities=sorted(raw[(*k,models[0])])
            for m in models:
                if set(raw[(*k,m)])!=set(identities):
                    raise ValueError(f'Model pool mismatch: {k}/{m}')
            # Digest + label identifies an underlying document across seeds/periods.
            # If the same digest is assigned conflicting gold labels, stop instead of silently combining.
            dkeys=[(x[1],x[2]) for x in identities]
            # Repeated digest+label records are retained as separate rows but
            # resampled together as one document cluster. Never deduplicate rows.
            # This handles duplicated reviews within the same evaluation pool.
            # The audit reports their multiplicities explicitly.
            digest_labels=defaultdict(set)
            for digest,label in dkeys:
                digest_labels[digest].add(label)
            conflicts=sum(len(labels)>1 for labels in digest_labels.values())
            if conflicts:
                raise ValueError(f'{conflicts} document digests have conflicting gold labels in pool {k}; investigate before resampling')
            pools[k]={'doc_keys':dkeys,'models':{m:(np.array([raw[(*k,m)][id][0] for id in identities]),
                                                      np.array([raw[(*k,m)][id][1] for id in identities]))
                                               for m in models}}
        return pools
    finally:
        for z in archives.values():z.close()


def overlap_audit(pools):
    by_setting={}
    for setting in ('Climate n500','Yelp n500','Yelp n1000'):
        entries={ (period,seed):set(p['doc_keys']) for (s,period,seed),p in pools.items() if s==setting }
        if not entries: continue
        count=Counter(key for ids in entries.values() for key in ids)
        per_pool={ (period,seed):Counter(p['doc_keys']) for (s,period,seed),p in pools.items() if s==setting }
        within_pool_repeats={f'{period}_seed{seed}':{'duplicate_cluster_count':sum(v>1 for v in c.values()),
                          'extra_repeated_rows':sum(v-1 for v in c.values()),
                          'maximum_multiplicity':max(c.values())}
                          for (period,seed),c in per_pool.items() if any(v>1 for v in c.values())}
        seed_within=[]
        for period in ('within','short','long'):
            sets=[ids for (p,s),ids in entries.items() if p==period]
            if len(sets)>1:
                intersections=[len(sets[i]&sets[j]) for i in range(len(sets)) for j in range(i+1,len(sets))]
                seed_within.append({'period':period,'min_pairwise_overlap':min(intersections),
                                    'max_pairwise_overlap':max(intersections)})
        period_sets={p:set().union(*(v for (pp,s),v in entries.items() if pp==p))
                     for p in ('within','short','long')}
        cross_period={f'{a}_vs_{b}':len(period_sets[a]&period_sets[b])
                      for a,b in (('within','short'),('within','long'),('short','long'))}
        by_setting[setting]={'unique_document_clusters':len(count),
             'document_occurrences_across_seed_period_pools':sum(count.values()),
             'documents_repeated_in_multiple_pools':sum(c>1 for c in count.values()),
             'max_pool_occurrences_of_single_document':max(count.values()),
             'within_pool_duplicate_clusters':within_pool_repeats,
             'clustering_policy':'Keep every evaluation row; repeated digest+gold rows receive the same bootstrap weight within and across pools.',
             'seed_overlap_within_period':seed_within,'cross_period_unique_document_overlap':cross_period}
    return by_setting


def risk(pools,setting,seed,part,weights):
    period,model,pct=part
    c,e=pools[setting,period,seed]['models'][model]
    return prior.exact_risk(c,e,pct,weights[(period,seed)])


def run(pools,replicates,seed):
    rng=np.random.default_rng(seed)
    results=[]
    for setting,period,contrast,a,b in CONTRASTS:
        required={(p,s) for p,_,_ in (a,b) for s in range(42,47)}
        clusters=sorted(set().union(*(pools[(setting,p,s)]['doc_keys'] for p,s in required)))
        lookup={k:i for i,k in enumerate(clusters)}
        offsets={(p,s):np.array([lookup[k] for k in pools[setting,p,s]['doc_keys']],dtype=np.intp)
                 for p,s in required}
        ones={(p,s):np.ones(len(offsets[p,s]),dtype=np.int64) for p,s in required}
        point=np.array([risk(pools,setting,s,a,ones)-risk(pools,setting,s,b,ones) for s in range(42,47)])
        draws=np.empty(replicates)
        for it in range(replicates):
            global_w=rng.multinomial(len(clusters),np.full(len(clusters),1/len(clusters)))
            weights={k:global_w[ix] for k,ix in offsets.items()}
            if any(weights[k].sum()<5 for k in weights):
                raise RuntimeError('Insufficient retained observations during cluster bootstrap')
            draws[it]=np.mean([risk(pools,setting,s,a,weights)-risk(pools,setting,s,b,weights)
                               for s in range(42,47)])
        l,h=np.quantile(draws,[0.025,0.975])
        results.append({'setting':setting,'period':period,'contrast':contrast,
                        'a_model':a[1],'a_period':a[0],'a_coverage_pct':a[2],
                        'b_model':b[1],'b_period':b[0],'b_coverage_pct':b[2],
                        'mean_difference_pp':100*point.mean(),'ci_95_low_pp':100*l,
                        'ci_95_high_pp':100*h,'seed_sd_pp':100*statistics.stdev(point),
                        'seed_diffs_negative':int((point<0).sum()),'seed_diffs_positive':int((point>0).sum()),
                        'unique_document_clusters_in_contrast':len(clusters),
                        'replicates':replicates,'ci_method':'synchronized document-cluster bootstrap; fixed seeds and models'})
        print(f'{setting}, {period}, {contrast}: {100*point.mean():+.3f} pp [{100*l:+.3f},{100*h:+.3f}] clusters={len(clusters)}',flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--notebook',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--replicates',type=int,default=3000)
    p.add_argument('--seed',type=int,default=20261102)
    args=p.parse_args()
    if args.replicates<200:p.error('Choose at least 200 bootstrap replicates')
    core=prior.get_core(args.notebook)
    print('Verifying original 180 conditions and 14,580 curves...',flush=True)
    records,curves=core['audited_results'](args.source_dir)
    pools=build_index(core,args.source_dir)
    overlap=overlap_audit(pools)
    results=run(pools,args.replicates,args.seed)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    csvfile=args.output_dir/'RQ2_focused_cluster_bootstrap.csv'
    with csvfile.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    (args.output_dir/'RQ2_document_overlap_audit.json').write_text(json.dumps(overlap,indent=2),encoding='utf-8')
    notes={'verified_conditions':len(records),'verified_curve_rows':len(curves),'replicates':args.replicates,
           'bootstrap_random_seed':args.seed,
           'design':'Document clusters identified by digest and gold label within setting. Duplicate rows within the same pool are retained and jointly resampled, never silently dropped. Multinomial cluster weights are synchronized across all seed and period occurrences within each contrast. Model comparisons are paired. Risk is recomputed from confidence and correctness for each resample, preserving fractional confidence-tie handling.',
           'scope':'Conditional on archived models, five fixed seeds, evaluated cohorts and historical periods. Not an interval for new model training seeds, new periods, or shifted populations.',
           'limitations':'Repeated texts are clustered by digest+gold, even when they represent separately collected instances; this may be conservative or may not match the true sampling design. Document digests may not capture semantic duplicates, common authors/businesses, or cohort sampling strata. 11 exploratory comparisons are not multiplicity-adjusted. Temporal contrasts compare observed period cohorts, not necessarily causal temporal degradation. If sample selection uses a more complex design, adapt the resampling unit.'}
    (args.output_dir/'RQ2_improved_method_notes.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
    print('Saved',csvfile)

if __name__=='__main__':main()
