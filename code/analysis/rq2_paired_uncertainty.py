#!/usr/bin/env python3
"""RQ2 focused uncertainty analysis from original archived predictions.

Reads Temporal_coverage_rankings.ipynb computational core and its five input ZIPs.
No training. Checks archived metrics first. Resamples matched examples within
setting/period/seed, identically across compared models. The resulting 95% CIs
are CONDITIONAL on the five recorded seeds and sampled evaluation cohorts; they
are not intervals for arbitrary future periods, and do not account fully for
cross-seed reuse of examples. Treat exploratory comparisons as exploratory.
"""
import argparse, csv, io, json, math, statistics, zipfile
from pathlib import Path
from collections import Counter
import numpy as np

CONTRASTS=[]
for setting in ('Yelp n500','Yelp n1000'):
    for period in ('within','short','long'):
        CONTRASTS.append((f'{setting}: Phi minus Qwen, 80%',setting,(period,'phi3.5:latest',80),(period,'qwen2.5:3b-instruct',80)))
for period in ('within','short','long'):
    CONTRASTS.append((f'Climate: Qwen 80% minus 100%, {period}', 'Climate n500', (period,'qwen2.5:3b-instruct',80),(period,'qwen2.5:3b-instruct',100)))
for period in ('short','long'):
    CONTRASTS.append((f'Climate: DistilBERT {period} minus reference, 80%', 'Climate n500',(period,'distilbert',80),('within','distilbert',80)))

def get_core(notebook):
    obj=json.loads(Path(notebook).read_text(encoding='utf-8'))
    found=[]
    for cell in obj['cells']:
        if cell['cell_type']=='code':
            code=''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
            if 'def audited_results(' in code and 'def prediction_rows(' in code: found.append(code)
    if len(found)!=1: raise ValueError('Cannot locate exactly one audit code cell in notebook')
    scope={'__name__':'_rq2_notebook_core_'}
    exec(compile(found[0],str(notebook),'exec'),scope)
    return scope

def exact_risk(conf,wrong,pct, weights=None):
    """Expected retained risk with fractional inclusion of confidence ties.
    Sample weights are integer multinomial bootstrap counts for examples.
    """
    if weights is None: weights=np.ones(len(conf),dtype=np.int64)
    n=int(weights.sum()); retained=(n*pct)//100
    if retained<=0: return float('nan')
    order=np.argsort(-conf,kind='stable')
    c=conf[order];w=weights[order];e=wrong[order]
    levels,ix=np.unique(c,return_inverse=True)
    counts=np.bincount(ix,weights=w,minlength=len(levels))[::-1]
    errs=np.bincount(ix,weights=w*e,minlength=len(levels))[::-1]
    cum=np.cumsum(counts)
    boundary=int(np.searchsorted(cum,retained,side='left'))
    prior=cum[boundary-1] if boundary else 0
    prev_error=errs[:boundary].sum()
    return float((prev_error+(retained-prior)*errs[boundary]/counts[boundary])/retained)

def build_data(core,source, contrasts):
    paths={}
    for key,stem in core['STEMS'].items():
        matches=list(Path(source).rglob(stem+'*.zip'))
        if len(matches)!=1: raise ValueError(f'Expected exactly one {stem}*.zip under {source}, found {len(matches)}: {matches}')
        paths[key]=matches[0]
    archive={k:zipfile.ZipFile(p) for k,p in paths.items()}
    needed=set()
    for _,setting,a,b in contrasts:
        for period,model,_ in (a,b):
            for seed in core['SEEDS']: needed.add((setting,period,int(seed),model))
    try:
        out={}
        for setting,period,seed,model in sorted(needed):
            rows=core['prediction_rows'](archive,setting,model,seed,period)
            gold,pred,digest=(('true_label','predicted_label','text_sha256') if setting.startswith('Climate')
                              else ('gold_label','pred_label','text_sha1'))
            identifiers=[(r['example_id'],r[digest],r[gold]) for r in rows]
            conf=np.asarray([float(r['confidence']) for r in rows])
            wrong=np.asarray([float(r[gold]!=r[pred]) for r in rows])
            if len(identifiers)!=len(set(identifiers)) or not np.isfinite(conf).all() or (conf<0).any() or (conf>1).any():
                raise ValueError(f'Invalid rows: {setting}/{period}/{seed}/{model}')
            out[setting,period,seed,model]={identifiers[i]:(conf[i],wrong[i]) for i in range(len(rows))}
        # Align each model to identical (id,digest,gold) ordering within pool.
        pools={}
        for setting,period,seed,_ in needed:
            pool=(setting,period,seed)
            if pool in pools:continue
            names=sorted(k[3] for k in needed if k[:3]==pool)
            ref=out[(*pool,names[0])]
            ids=sorted(ref)
            arrays={}
            for model in names:
                indexed=out[(*pool,model)]
                if set(indexed)!=set(ids): raise ValueError(f'Unmatched evaluation pool: {pool} {model}')
                arrays[model]=(np.array([indexed[i][0] for i in ids]), np.array([indexed[i][1] for i in ids]))
            pools[pool]=arrays
        return pools
    finally:
        for z in archive.values(): z.close()

def evaluate(pools,setting,seed,part,weights):
    period,model,pct=part
    conf,wrong=pools[setting,period,seed][model]
    return exact_risk(conf,wrong,pct,weights[setting,period,seed])

def analysis(pools,boot,random_seed):
    rng=np.random.default_rng(random_seed)
    findings=[]
    for label,setting,a,b in CONTRASTS:
        observed=[]
        for seed in range(42,47):
            weights={}
            for period in {a[0],b[0]}:
                n=len(next(iter(pools[setting,period,seed].values()))[0])
                weights[setting,period,seed]=np.ones(n,dtype=np.int64)
            observed.append(evaluate(pools,setting,seed,a,weights)-evaluate(pools,setting,seed,b,weights))
        draws=np.empty(boot)
        for i in range(boot):
            per_seed=[]
            for seed in range(42,47):
                weights={}
                for period in {a[0],b[0]}:
                    n=len(next(iter(pools[setting,period,seed].values()))[0])
                    weights[setting,period,seed]=rng.multinomial(n,np.full(n,1/n))
                per_seed.append(evaluate(pools,setting,seed,a,weights)-evaluate(pools,setting,seed,b,weights))
            draws[i]=np.mean(per_seed)
        lo,hi=np.quantile(draws,[.025,.975]);m=float(np.mean(observed))
        findings.append(dict(comparison=label,mean_diff_pp=100*m,ci_95_low_pp=100*lo,ci_95_high_pp=100*hi,
                             seed_sd_pp=100*statistics.stdev(observed),min_seed_diff_pp=100*min(observed),
                             max_seed_diff_pp=100*max(observed),seeds_negative=sum(v<0 for v in observed),
                             seeds_positive=sum(v>0 for v in observed),n_seeds=5,bootstrap_replicates=boot,
                             interval_type='conditional paired example bootstrap; fixed 5 seeds'))
        print(f'{label}: {100*m:+.3f} pp [95% conditional CI {100*lo:+.3f}, {100*hi:+.3f}]; seed SD {100*statistics.stdev(observed):.3f}')
    return findings

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--notebook',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--replicates',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261101)
    args=p.parse_args()
    if args.replicates<100: p.error('Use at least 100 replicates; recommend 2000 or 5000')
    core=get_core(args.notebook)
    print('Running original archive audit (no model training)...',flush=True)
    records,curves=core['audited_results'](args.source_dir)
    pools=build_data(core,args.source_dir,CONTRASTS)
    result=analysis(pools,args.replicates,args.seed)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    csv_path=args.output_dir/'RQ2_focused_paired_bootstrap.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=result[0].keys());w.writeheader();w.writerows(result)
    metadata={'archive_verified_conditions':len(records),'archived_curve_rows':len(curves),
              'note':'All intervals are conditional on five fixed seeds; example-level paired bootstrap within each seed. When periods differ, each period is resampled independently. Cross-seed overlap, cohort sampling design, historical-period generalization, and multiplicity are not fully accounted for. Do not present as definitive significance tests.',
              'random_seed':args.seed,'replicates':args.replicates}
    (args.output_dir/'RQ2_bootstrap_method_notes.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Saved',csv_path)
if __name__=='__main__':main()
