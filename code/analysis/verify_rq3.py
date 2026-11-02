#!/usr/bin/env python3
"""Audit reference-period P1/P3/P5 results from saved predictions only.

Core analysis uses the Python standard library. Matplotlib is optional.
No inference, fitting, requests, silent label repairs, or row dropping.
"""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import re
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

SEEDS = (42, 43, 44, 45, 46)
SETTINGS = ('Climate n500', 'Yelp n500', 'Yelp n1000')
MODELS = ('phi3.5:latest', 'qwen2.5:3b-instruct')
SLUGS = ('phi3.5_latest', 'qwen2.5_3b-instruct')
PROMPTS = ('p1_minimal', 'p3_json_strict', 'p5_abstain')
SHORT = dict(zip(MODELS, ('Phi', 'Qwen')))
SOURCE_STEMS = {'climate':'completed_prediction_audit',
                'yelp':'original_yelp_frozen_predictions',
                'yelp_metrics':'yelp_results_audit'}
TOL = 1e-8


def fail(condition, message):
    if not condition:
        raise ValueError(message)


def csv_rows(z, suffix):
    found = [n for n in z.namelist() if n.replace('\\','/').endswith(suffix)]
    fail(len(found)==1, f'Expected one ZIP member ending {suffix!r}; found {len(found)}')
    with z.open(found[0]) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline=''))), found[0]


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def boolean(value):
    value=str(value).strip().lower()
    if value=='true':return True
    if value=='false':return False
    raise ValueError(f'Unexpected boolean representation: {value!r}')


def finite_number(value):
    if isinstance(value,bool):return False
    return isinstance(value,(int,float)) and math.isfinite(value)


def reject_duplicates(pairs):
    obj={}
    for key,value in pairs:
        if key in obj:raise ValueError('Duplicate JSON key')
        obj[key]=value
    return obj


def inspect_raw(raw, labels):
    """Independent strict JSON/schema audit; does not rewrite saved results."""
    try:
        obj=json.loads(raw.strip(),object_pairs_hook=reject_duplicates,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError,TypeError):return 'json_error',None
    if not isinstance(obj,dict) or set(obj)!={'label','confidence','abstain'}:
        return 'schema_error',obj
    if (obj['label'] not in labels or not finite_number(obj['confidence']) or
            not 0<=obj['confidence']<=1 or not isinstance(obj['abstain'],bool)):
        return 'schema_error',obj
    return 'ok',obj


def accuracy(gold,pred):
    return sum(a==b for a,b in zip(gold,pred))/len(gold) if gold else None


def macro_f1(gold,pred,labels):
    if not gold:return None
    scores=[]
    for label in labels:
        tp=sum(a==label and b==label for a,b in zip(gold,pred))
        fp=sum(a!=label and b==label for a,b in zip(gold,pred))
        fn=sum(a==label and b!=label for a,b in zip(gold,pred))
        den=2*tp+fp+fn
        scores.append(2*tp/den if den else 0.)
    return statistics.mean(scores)


def ece(gold,pred,confidence):
    if not gold:return None
    fail(all(math.isfinite(c) and 0<=c<=1 for c in confidence),'ECE confidence outside [0,1]')
    # These are the np.linspace(0,1,11) IEEE-754 endpoints used by the
    # archived evaluators, including 0.30000000000000004 and 0.6000000000000001.
    bounds=[i*0.1 for i in range(10)]+[1.]
    value=0.
    for b,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
        ids=[i for i,c in enumerate(confidence) if c>=lo and (c<=hi if b==9 else c<hi)]
        if ids:
            value+=abs(sum((gold[i]==pred[i])-confidence[i] for i in ids))/len(gold)
    return value


def quantile(values,q):
    if not values:return None
    v=sorted(values);x=(len(v)-1)*q;lo=math.floor(x);hi=math.ceil(x)
    return v[lo]+(v[hi]-v[lo])*(x-lo)


def condition_scores(rows,labels):
    """Explicit denominators for fallback, usable-output and all-attempt scores."""
    n=len(rows)
    gold=[r['gold'] for r in rows];pred=[r['pred'] for r in rows]
    conf=[r['confidence'] for r in rows]
    usable=[r for r in rows if r['usable']]
    kept=[r for r in usable if not r['abstain']]
    abstained=[r for r in usable if r['abstain']]
    ug=[r['gold'] for r in usable];up=[r['pred'] for r in usable]
    invalid_pred=[r['pred'] if r['usable'] else '__FAILURE__' for r in rows]
    kg=[r['gold'] for r in kept];kp=[r['pred'] for r in kept]
    ag=[r['gold'] for r in abstained];ap=[r['pred'] for r in abstained]
    times=[r['latency'] for r in rows]
    goodtimes=[r['latency'] for r in usable]
    stored_keep=[r for r in rows if not r['abstain']]
    out=dict(n_attempted=n,n_usable=len(usable),n_failure=n-len(usable),
        n_retained=len(kept),n_valid_abstained=len(abstained),
        archived_accuracy=accuracy(gold,pred),archived_macro_f1=macro_f1(gold,pred,labels),
        archived_ece=ece(gold,pred,conf),
        archived_abstain_rate=sum(r['abstain'] for r in rows)/n,
        archived_coverage=len(stored_keep)/n,
        archived_retained_risk=(1-accuracy([r['gold'] for r in stored_keep],[r['pred'] for r in stored_keep])) if stored_keep else None,
        failure_aware_accuracy=accuracy(gold,invalid_pred),
        failure_aware_macro_f1=macro_f1(gold,invalid_pred,labels),
        usable_accuracy=accuracy(ug,up),usable_macro_f1=macro_f1(ug,up,labels),
        usable_ece=ece(ug,up,[r['confidence'] for r in usable]),
        failure_rate=(n-len(usable))/n,coverage_all_attempts=len(kept)/n,
        abstain_rate_all_attempts=len(abstained)/n,
        abstain_rate_among_usable=len(abstained)/len(usable) if usable else None,
        retained_risk=(1-accuracy(kg,kp)) if kept else None,
        abstained_best_guess_risk=(1-accuracy(ag,ap)) if abstained else None,
        latency_recorded_mean=statistics.mean(times),latency_recorded_median=statistics.median(times),
        latency_recorded_p95=quantile(times,.95),
        latency_usable_mean=statistics.mean(goodtimes) if goodtimes else None,
        n_zero_recorded_latency=sum(t==0 for t in times),
        n_request_error=sum(r['request_failed'] for r in rows),
        n_saved_json_error=sum(r['status']=='json_error' for r in rows),
        n_saved_schema_error=sum(r['status']=='schema_error' for r in rows),
        n_raw_json_error_after_response=sum(not r['request_failed'] and r['raw_status']=='json_error' for r in rows),
        n_raw_schema_error_after_response=sum(not r['request_failed'] and r['raw_status']=='schema_error' for r in rows),
        n_raw_schema_ok=sum(not r['request_failed'] and r['raw_status']=='ok' for r in rows),
        n_raw_saved_disagreement=sum(r['raw_saved_disagreement'] for r in rows),
        n_non_p5_abstain_flags=sum(r['abstain'] and r['usable'] for r in rows),
        n_p5_abstain_confidence_above_half=sum(r['abstain'] and r['usable'] and r['confidence']>.5 for r in rows),
        n_failure_fallback_label_correct=sum(not r['usable'] and r['gold']==r['pred'] for r in rows))
    fail(math.isclose(out['failure_rate']+out['coverage_all_attempts']+out['abstain_rate_all_attempts'],1.,abs_tol=1e-12),
         'Failure/abstention/retained denominator partition failed')
    return out


def load_condition(z,setting,model,seed,prompt):
    climate=setting.startswith('Climate');slug=SLUGS[MODELS.index(model)]
    n=1000 if setting.endswith('1000') else 500
    suffix=(f'predictions/frozen/{slug}/{prompt}/climate/seed_{seed}/within_n500_per_label.csv' if climate else
            f'predictions/frozen/{slug}/seed{seed}/n{n}_within_{prompt}.csv')
    records,member=csv_rows(z,suffix)
    labels=('BELIEVER','DENIER') if climate else ('NEG','POS')
    goldcol,predcol,hashcol,rawcol=(('true_label','predicted_label','text_sha256','raw_response') if climate else
                                 ('gold_label','pred_label','text_sha1','raw_output'))
    rows=[]
    for r in records:
        fail(r[goldcol] in labels,f'Gold label outside task domain: {member}')
        confidence=float(r['confidence']);latency=float(r['latency_sec'])
        fail(math.isfinite(confidence) and 0<=confidence<=1,f'Invalid stored confidence: {member}')
        fail(math.isfinite(latency) and latency>=0,f'Invalid stored latency: {member}')
        h=r[hashcol];fail(re.fullmatch('[0-9a-f]{'+('64' if climate else '40')+'}',h) is not None,'Invalid saved hash')
        status=r['status'].strip();request_failed=status=='request_error' or bool(r.get('request_error','').strip())
        abstain=boolean(r['abstain']);raw_status,obj=inspect_raw(r.get(rawcol,''),labels)
        disagreement=False
        if raw_status=='ok':
            disagreement=(obj['label']!=r[predcol] or obj['abstain']!=abstain or
                          not math.isclose(float(obj['confidence']),confidence,abs_tol=1e-9))
        usable=status=='ok' and r[predcol] in labels and not request_failed
        if climate:
            for key,expected in [('seed',str(seed)),('period','within'),('prompt_id',prompt),('model',model),('n_per_label',str(n))]:
                fail(r[key]==expected,f'Row provenance differs from archive path: {member}, {key}')
            fail(boolean(r['is_correct'])==(r[goldcol]==r[predcol]),'Saved correctness flag disagrees with labels')
        rows.append(dict(id=r['example_id'],hash=h,gold=r[goldcol],pred=r[predcol],confidence=confidence,
            abstain=abstain,status=status,request_failed=request_failed,usable=usable,raw_status=raw_status,
            raw_saved_disagreement=disagreement,latency=latency,decode_seed=r.get('decode_seed','')))
    fail(len(rows)==2*n,f'Wrong number of attempted rows: {member}')
    fail(Counter(r['gold'] for r in rows)=={lab:n for lab in labels},f'Unbalanced attempted gold pool: {member}')
    identities=[(r['id'],r['hash'],r['gold']) for r in rows]
    fail(len(set(identities))==len(rows),f'Duplicate row identities: {member}')
    fail(len({r['id'] for r in rows})==len(rows),f'Duplicate example IDs: {member}')
    rows.sort(key=lambda r:(r['hash'],r['id']))
    return rows,labels,member


def archived_metric_index(climate_z,yelp_metrics_z):
    result={}
    records,_=csv_rows(climate_z,'metrics_per_condition.csv')
    for r in records:
        if r.get('family')!='frozen' or r['dataset']!='climate' or r['period']!='within':continue
        key=('Climate n500',r['model'],int(r['seed']),r['prompt_id'])
        fail(key not in result,f'Duplicate archived metric {key}')
        result[key]={k:r[k] for k in ['accuracy','macro_f1','ece','abstain_rate','coverage','invalid_rate','latency_mean_sec']}
        result[key]['risk_covered']=str(1-float(r['covered_accuracy'])) if r['covered_accuracy'] else ''
    records,_=csv_rows(yelp_metrics_z,'metrics_frozen_partial.csv')
    for r in records:
        if r['split']!='within':continue
        key=(f"Yelp n{int(r['eval_n_per_label'])}",r['model'],int(r['seed']),r['prompt'])
        fail(key not in result,f'Duplicate archived metric {key}')
        result[key]={k:r[k] for k in ['accuracy','macro_f1','ece','abstain_rate','coverage','invalid_rate','latency_mean_sec','risk_covered']}
    return result


def check_archived(metrics,expected,key):
    mapping={'accuracy':'archived_accuracy','macro_f1':'archived_macro_f1','ece':'archived_ece',
             'abstain_rate':'archived_abstain_rate','coverage':'archived_coverage',
             'invalid_rate':'failure_rate','latency_mean_sec':'latency_recorded_mean','risk_covered':'archived_retained_risk'}
    checks=[]
    for old,new in mapping.items():
        if expected.get(old,'')=='':continue
        recorded=float(expected[old]);computed=metrics[new]
        fail(computed is not None and math.isclose(recorded,computed,rel_tol=0.,abs_tol=TOL),
             f'Archived {old} mismatch for {key}: recorded={recorded}, recomputed={computed}')
        checks.append(dict(setting=key[0],model=key[1],seed=key[2],prompt=key[3],metric=old,
                           recorded=recorded,recomputed=computed,absolute_difference=abs(recorded-computed)))
    return checks


def summaries(seed_metrics):
    groups=defaultdict(list)
    for r in seed_metrics:groups[r['setting'],r['model'],r['prompt']].append(r)
    excluded={'setting','model','prompt','seed','archive_role','member','decode_seeds'}
    output=[]
    for key,rows in groups.items():
        fail({r['seed'] for r in rows}==set(SEEDS),'Missing seed in summary')
        for metric in rows[0]:
            if metric in excluded:continue
            values=[r[metric] for r in rows if r[metric] is not None]
            output.append(dict(setting=key[0],model=key[1],prompt=key[2],metric=metric,
                mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values)>1 else None,
                minimum=min(values) if values else None,maximum=max(values) if values else None,n_defined_seeds=len(values)))
    return output


def prompt_comparisons(conditions,seed_metrics):
    scores={(r['setting'],r['model'],r['seed'],r['prompt']):r for r in seed_metrics}
    ranges=[];effects=[];common_rows=[]
    for setting in SETTINGS:
        for seed in SEEDS:
            allsets=[conditions[setting,m,seed,p] for m in MODELS for p in PROMPTS]
            ids=[(r['id'],r['hash'],r['gold']) for r in allsets[0]]
            fail(all([(r['id'],r['hash'],r['gold']) for r in rows]==ids for rows in allsets),'Unmatched prompt/model examples')
            common=[i for i in range(len(ids)) if all(rows[i]['usable'] for rows in allsets)]
            labels=('BELIEVER','DENIER') if setting.startswith('Climate') else ('NEG','POS')
            for model in MODELS:
                base=[scores[setting,model,seed,p] for p in PROMPTS]
                common_f1={}
                for prompt in PROMPTS:
                    rows=conditions[setting,model,seed,prompt]
                    common_f1[prompt]=macro_f1([rows[i]['gold'] for i in common],[rows[i]['pred'] for i in common],labels)
                    common_rows.append(dict(setting=setting,model=model,seed=seed,prompt=prompt,n_common=len(common),n_attempted=len(ids),
                        macro_f1=common_f1[prompt],accuracy=accuracy([rows[i]['gold'] for i in common],[rows[i]['pred'] for i in common])))
                for score_name in ['archived_macro_f1','failure_aware_macro_f1','usable_macro_f1','common_matched_macro_f1']:
                    vals=[common_f1[p] for p in PROMPTS] if score_name=='common_matched_macro_f1' else [r[score_name] for r in base]
                    defined=all(v is not None for v in vals)
                    ranges.append(dict(setting=setting,model=model,seed=seed,score=score_name,n_common=len(common),
                        range=max(vals)-min(vals) if defined else None,
                        best_prompt=PROMPTS[vals.index(max(vals))] if defined else '',
                        worst_prompt=PROMPTS[vals.index(min(vals))] if defined else ''))
                for a,b in itertools.combinations(PROMPTS,2):
                    ra=conditions[setting,model,seed,a];rb=conditions[setting,model,seed,b]
                    both=[i for i in range(len(ids)) if ra[i]['usable'] and rb[i]['usable']]
                    for metric in ['archived_macro_f1','failure_aware_macro_f1','usable_macro_f1','common_matched_macro_f1']:
                        va=common_f1[a] if metric=='common_matched_macro_f1' else scores[setting,model,seed,a][metric]
                        vb=common_f1[b] if metric=='common_matched_macro_f1' else scores[setting,model,seed,b][metric]
                        effects.append(dict(setting=setting,model=model,seed=seed,prompt_a=a,prompt_b=b,metric=metric,
                            a_minus_b=va-vb if va is not None and vb is not None else None,
                            both_usable_n=len(both),label_disagreement_rate=(sum(ra[i]['pred']!=rb[i]['pred'] for i in both)/len(both)) if both else None))
    rg=defaultdict(list);eg=defaultdict(list)
    for r in ranges:rg[r['setting'],r['model'],r['score']].append(r)
    for r in effects:eg[r['setting'],r['model'],r['prompt_a'],r['prompt_b'],r['metric']].append(r)
    rs=[];es=[]
    for key,rows in rg.items():
        vals=[r['range'] for r in rows if r['range'] is not None]
        rs.append(dict(setting=key[0],model=key[1],score=key[2],mean_seed_range=statistics.mean(vals) if vals else None,
            sample_sd=statistics.stdev(vals) if len(vals)>1 else None,n_defined_seeds=len(vals),
            smallest_common_n=min(r['n_common'] for r in rows)))
    for key,rows in eg.items():
        vals=[r['a_minus_b'] for r in rows if r['a_minus_b'] is not None]
        es.append(dict(setting=key[0],model=key[1],prompt_a=key[2],prompt_b=key[3],metric=key[4],
            mean_a_minus_b=statistics.mean(vals) if vals else None,
            sample_sd_paired_difference=statistics.stdev(vals) if len(vals)>1 else None,
            seeds_a_higher=sum(v>1e-12 for v in vals),seeds_b_higher=sum(v< -1e-12 for v in vals),
            seeds_tied=sum(abs(v)<=1e-12 for v in vals)))
    return ranges,rs,effects,es,common_rows


def p5_rows(seed_metrics):
    index={(r['setting'],r['model'],r['seed'],r['prompt']):r for r in seed_metrics}
    output=[]
    for setting in SETTINGS:
        for model in MODELS:
            for seed in SEEDS:
                p5=index[setting,model,seed,PROMPTS[2]];p3=index[setting,model,seed,PROMPTS[1]]
                risk_all=1-p5['usable_accuracy'] if p5['usable_accuracy'] is not None else None
                output.append(dict(setting=setting,model=model,seed=seed,
                    n_attempted=p5['n_attempted'],n_failure=p5['n_failure'],n_retained=p5['n_retained'],
                    n_abstained=p5['n_valid_abstained'],coverage=p5['coverage_all_attempts'],
                    abstain_rate=p5['abstain_rate_all_attempts'],failure_rate=p5['failure_rate'],
                    p5_retained_risk=p5['retained_risk'],p5_all_usable_best_guess_risk=risk_all,
                    p5_abstained_best_guess_risk=p5['abstained_best_guess_risk'],
                    retained_minus_all_usable_risk=(p5['retained_risk']-risk_all) if p5['retained_risk'] is not None and risk_all is not None else None,
                    p3_usable_risk=1-p3['usable_accuracy'] if p3['usable_accuracy'] is not None else None,
                    p5_minus_p3_all_attempt_accuracy=p5['failure_aware_accuracy']-p3['failure_aware_accuracy']))
    return output


def manuscript_checks(path,summary,range_summary):
    if path is None:return []
    text=path.read_text(encoding='utf-8-sig')
    start=text.find(r'\label{tab:frozen-reliability}')
    fail(start>=0,'Manuscript table label tab:frozen-reliability not found')
    start=text.find(r'\begin{tabular}',start);end=text.find(r'\end{tabular}',start)
    fail(start>=0 and end>start,'Manuscript table could not be isolated')
    d={(r['setting'],r['model'],r['prompt'],r['metric']):r['mean'] for r in summary}
    rg={(r['setting'],r['model'],r['score']):r['mean_seed_range'] for r in range_summary}
    output=[];setting=None
    for line in text[start:end].splitlines():
        if '&' not in line:continue
        cells=line.split('&')
        if len(cells)!=8:continue
        for s in SETTINGS:
            if s in cells[0]:setting=s
        model=next((m for m in MODELS if ('phi' if m==MODELS[0] else 'qwen') in cells[1].lower()),None)
        if not setting or not model:continue
        values=[]
        for cell in cells[2:]:
            matches=re.findall(r'[+-]?\d+\.\d+',cell)
            fail(len(matches)==1,'Ambiguous numeric manuscript table cell')
            values.append(float(matches[0]))
        metrics=['prompt_sensitivity','canonical_ece','p5_abstain_rate','p5_coverage','p5_retained_risk','canonical_latency']
        expected=[rg[setting,model,'archived_macro_f1'],d[setting,model,PROMPTS[1],'usable_ece'],
                  d[setting,model,PROMPTS[2],'abstain_rate_all_attempts'],d[setting,model,PROMPTS[2],'coverage_all_attempts'],
                  d[setting,model,PROMPTS[2],'retained_risk'],d[setting,model,PROMPTS[1],'latency_recorded_mean']]
        for metric,old,new in zip(metrics,values,expected):
            output.append(dict(setting=setting,model=model,metric=metric,manuscript_value=old,recomputed_value=new,
                matches_to_four_decimals=new is not None and abs(old-new)<=.00005000001,
                interpretation=('Archived score range includes fallback labels on failed rows; also inspect failure-aware/common-matched ranges.' if metric=='prompt_sensitivity' else 'Compared with the condition claimed in the manuscript prose.')))
    fail(len(output)==36,'Expected six manuscript rows and six metrics per row')
    return output


def canonical_checks(path,seed_metrics):
    if path is None:return []
    with path.open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
    anchor={}
    for r in records:
        if r.get('period')!='within' or r['model'] not in MODELS or r['setting'] not in SETTINGS:continue
        key=(r['setting'],r['model'],int(r['seed']))
        fail(key not in anchor,'Duplicate P3 reference anchor')
        anchor[key]=r
    fail(len(anchor)==30,'Expected 30 P3 reference anchors')
    checks=[]
    for r in seed_metrics:
        if r['prompt']!=PROMPTS[1]:continue
        key=(r['setting'],r['model'],r['seed'])
        for metric,new in [('accuracy','archived_accuracy'),('macro_f1','archived_macro_f1'),('ece','archived_ece')]:
            recorded=float(anchor[key][metric]);computed=r[new]
            fail(math.isclose(recorded,computed,rel_tol=0.,abs_tol=TOL),f'P3 canonical anchor mismatch: {key}, {metric}')
            checks.append(dict(setting=key[0],model=key[1],seed=key[2],metric=metric,
                canonical_value=recorded,recomputed=computed,absolute_difference=abs(recorded-computed)))
    return checks


def tex_outputs(out,summary,range_summary):
    d={(r['setting'],r['model'],r['prompt'],r['metric']):r for r in summary}
    rg={(r['setting'],r['model'],r['score']):r for r in range_summary}
    def display(r,mean='mean',sd='sample_sd'):
        return '--' if r[mean] is None else f"${r[mean]:.4f} \\pm {r[sd]:.4f}$" if r[sd] is not None else f"${r[mean]:.4f}$"
    def table(name,caption,label,header,rows):
        lines=[r'\begin{table*}[t]',r'\centering\scriptsize',r'\caption{'+caption+'}',r'\label{'+label+'}',
               r'\begin{tabular}{'+'lll'+'r'*(len(header)-3)+'}',r'\toprule',' & '.join(header)+r' \\',r'\midrule']
        lines+=[' & '.join(row)+r' \\' for row in rows]
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
        (out/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    rows=[]
    for setting in SETTINGS:
        for model in MODELS:
            for pi,p in enumerate(PROMPTS):
                rows.append([setting,SHORT[model],f'P{[1,3,5][pi]}']+[display(d[setting,model,p,k]) for k in
                    ['failure_aware_macro_f1','usable_macro_f1','failure_rate','usable_ece']])
    table('Table_RQ3_prompt_results.tex',
        r'Reference-period prompt results, means $\pm$ sample SD across five sampling seeds. All-attempt Macro-F1 treats unsuccessful outputs as an unassigned prediction, contributing a false negative to the true task class. P5 classification scores use the required best-guess label, including when abstention is flagged. Usable-output Macro-F1 and ECE condition on successful parsed outputs and may use different subsets. Failure rates use all attempted rows; archived fallback scores are provided separately in the CSV outputs. These are configuration comparisons, not controlled paraphrase tests.',
        'tab:rq3-prompt-results',['Setting','Model','Prompt','All-attempt F1','Usable-output F1','Failure rate','Usable-output ECE'],rows)
    rows=[]
    for setting in SETTINGS:
        for model in MODELS:
            rows.append([setting,SHORT[model],'P5']+[display(d[setting,model,PROMPTS[2],k]) for k in
                ['abstain_rate_all_attempts','coverage_all_attempts','retained_risk']])
    table('Table_RQ3_P5_abstention.tex',
        r'Explicit P5 abstention at reference, means $\pm$ sample SD. Abstention and coverage use all attempted examples as their denominator; failure is a separate category. Retained risk uses successful non-abstained outputs only. When no examples are retained, risk is undefined. P5 changes the prompt and may change labels, so its operating point does not estimate a P3 confidence-threshold trade-off.',
        'tab:rq3-p5-abstention',['Setting','Model','Prompt','Abstention rate','Coverage','Retained risk'],rows)
    rows=[]
    for setting in SETTINGS:
        for model in MODELS:
            rows.append([setting,SHORT[model],'P1/P3/P5']+[display(rg[setting,model,k],mean='mean_seed_range') for k in
                ['archived_macro_f1','failure_aware_macro_f1','common_matched_macro_f1']])
    table('Table_RQ3_configuration_ranges.tex',
        r'Prompt-configuration Macro-F1 ranges, means $\pm$ sample SD of within-seed ranges across P1/P3/P5. Archived ranges reproduce stored fallback-label scoring. All-attempt ranges incorporate unsuccessful outputs. Common-matched ranges use the same examples successful under all six model--prompt combinations within each seed; these conditional subsets may exclude difficult or failure-prone cases. No range is a pure paraphrase effect.',
        'tab:rq3-configuration-ranges',['Setting','Model','Configurations','Archived range','All-attempt range','Common-matched range'],rows)


def plot_outputs(out,seed_metrics,common_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    colors={MODELS[0]:'#009E73',MODELS[1]:'#D55E00'}
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(11.4,6.8),squeeze=False)
    for col,setting in enumerate(SETTINGS):
        for row in range(2):
            ax=axs[row,col]
            for mi,model in enumerate(MODELS):
                values=[];sds=[]
                for prompt in PROMPTS:
                    rs=[r for r in (seed_metrics if row==0 else common_rows) if r['setting']==setting and r['model']==model and r['prompt']==prompt]
                    v=[r['failure_aware_macro_f1'] if row==0 else r['macro_f1'] for r in rs]
                    v=[x for x in v if x is not None]
                    values.append(statistics.mean(v) if v else math.nan)
                    sds.append(statistics.stdev(v) if len(v)>1 else 0.)
                ax.errorbar(range(3),values,yerr=sds,color=colors[model],marker='o' if mi==0 else 's',capsize=3,label=SHORT[model])
            ax.set_xticks(range(3),['P1','P3','P5']);ax.set_title(setting)
            ax.set_ylabel('All-attempt Macro-F1' if row==0 else 'Common-matched Macro-F1')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.grid(axis='y',alpha=.2)
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=2,loc='upper center',frameon=False)
    fig.tight_layout(rect=(0,.105,1,.94))
    fig.text(.04,.025,'Means ±1 sample SD, not confidence intervals. Top: unsuccessful outputs treated as unassigned predictions.\nBottom: identical successful-output subsets across both models and all three prompts within each seed; conditional comparisons.',fontsize=8.5)
    for ext in ['svg','pdf','png']:fig.savefig(out/('RQ3_prompt_performance.'+ext),dpi=300,facecolor='white')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8));matrix=[];names=[]
    for setting in SETTINGS:
        for model in MODELS:
            names.append(setting+' · '+SHORT[model]);matrix.append([])
            for prompt in PROMPTS:
                rs=[r for r in seed_metrics if r['setting']==setting and r['model']==model and r['prompt']==prompt]
                matrix[-1].append((sum(r['n_failure'] for r in rs),sum(r['n_attempted'] for r in rs)))
    import numpy as np
    rates=np.array([[n/den for n,den in row] for row in matrix])
    ax.imshow(rates,cmap='Oranges',aspect='auto',vmin=0,vmax=max(.01,float(rates.max())))
    for i,row in enumerate(matrix):
        for j,(n,den) in enumerate(row):ax.text(j,i,f'{n:,} / {den:,}',ha='center',va='center',color='white' if rates[i,j]>rates.max()/2 and n else 'black')
    ax.set_xticks(range(3),['P1','P3','P5']);ax.set_yticks(range(6),names)
    ax.set_title('Unsuccessful outputs / attempted rows across five seeds')
    fig.tight_layout(rect=(0,.075,1,1));fig.text(.04,.02,'Counts include repeated texts across seeds. Request failures and response-format failures are distinguished in the CSV audit.',fontsize=8.5)
    for ext in ['svg','pdf','png']:fig.savefig(out/('RQ3_output_failures.'+ext),dpi=300,facecolor='white')
    plt.close(fig)


def report(out,seed_metrics,summary,range_summary,comparison,audit):
    d={(r['setting'],r['model'],r['prompt'],r['metric']):r['mean'] for r in summary}
    rindex={(r['setting'],r['model'],r['score']):r['mean_seed_range'] for r in range_summary}
    lines=['# RQ3: verified saved-prediction results','',
        f"Verified {audit['condition_count']} reference-period conditions and {audit['attempted_rows']:,} saved attempted rows. "
        f"Reproduced {audit['archived_metric_checks']} archived metric entries. No models were invoked or fitted.",'',
        '## P1/P3/P5 performance and failure audit','',
        '| Setting | Model | Prompt | Archived F1 | All-attempt F1 | Usable-output F1 | Failed rows |',
        '|---|---|---|---:|---:|---:|---:|']
    for setting in SETTINGS:
        for model in MODELS:
            for prompt in PROMPTS:
                vals=[d[setting,model,prompt,k] for k in ['archived_macro_f1','failure_aware_macro_f1','usable_macro_f1']]
                rs=[r for r in seed_metrics if r['setting']==setting and r['model']==model and r['prompt']==prompt]
                lines.append(f"| {setting} | {SHORT[model]} | {prompt[:2].upper()} | "+' | '.join(f'{v:.4f}' if v is not None else 'undefined' for v in vals)+f" | {sum(r['n_failure'] for r in rs)} / {sum(r['n_attempted'] for r in rs)} |")
    lines+=['','## P5 explicit abstention','',
        '| Setting | Model | Abstention rate | Operational coverage | Retained risk |',
        '|---|---|---:|---:|---:|']
    for setting in SETTINGS:
        for model in MODELS:
            vals=[d[setting,model,PROMPTS[2],k] for k in ['abstain_rate_all_attempts','coverage_all_attempts','retained_risk']]
            lines.append(f'| {setting} | {SHORT[model]} | '+' | '.join(f'{v:.6f}' if v is not None else 'undefined' for v in vals)+' |')
    lines+=['','## Configuration sensitivity','',
        '| Setting | Model | Mean archived range | Mean all-attempt range | Mean common-matched range |',
        '|---|---|---:|---:|---:|']
    for setting in SETTINGS:
        for model in MODELS:
            vals=[rindex[setting,model,k] for k in ['archived_macro_f1','failure_aware_macro_f1','common_matched_macro_f1']]
            lines.append(f'| {setting} | {SHORT[model]} | '+' | '.join(f'{v:.4f}' if v is not None else 'undefined' for v in vals)+' |')
    failed=[r for r in seed_metrics if r['n_failure']]
    lines+=['','## Interpretation and limits','',
        '- P1/P3/P5 vary instructions and response/abstention requirements; their ranges are configuration sensitivity, not controlled paraphrase robustness.',
        '- The historical Yelp evaluator assigns a fallback label and confidence 0 to failed requests. Archived metrics reproduce this policy; such fallback rows are not successful model predictions.',
        '- All-attempt accuracy counts unsuccessful outputs as incorrect. All-attempt Macro-F1 uses an unassigned prediction with fixed task labels: a failure contributes a false negative to its true class, not a new averaged class. It is not algebraically a constant penalty to the archived F1.',
        '- P5 classification scores use its required best-guess label even on valid outputs flagged abstain. This evaluates label quality separately from whether the model would autonomously retain the case; retained coverage/risk is reported separately.',
        '- Usable-output metrics condition on archived success status and valid stored predictions. The separate strict raw JSON/schema audit can flag additional disagreements; it never silently rewrites the stored results.',
        '- Common-matched comparisons preserve identical successful examples across both models and all prompts within each seed. These are conditional complete-case analyses and may exclude failure-prone cases.',
        '- P5 may alter labels as well as abstention. Compare retained risk with P5 best-guess risk before claiming that abstention helps; do not treat P5 as thresholding fixed P3 predictions.',
        '- Near-zero abstention or zero failures on these rows does not establish calibrated abstention or universal format robustness. Undefined risk and undefined abstained-subset error remain blank rather than being replaced by zero.',
        '- Latency is recorded per row and per prompt. Failed requests may have latency 0 in the archive, so those entries do not measure actual failed-request wall time. Model digests, hardware, retry histories and inference nondeterminism cannot be reconstructed from this audit.',
        '- Seed SD is descriptive variation, not a standard error of independent experiments. Texts overlap across seeds and the Yelp sample-size settings are nested. No significance tests or population confidence intervals are claimed.']
    if failed:
        lines+=['','### Failure-bearing conditions']
        for r in failed:lines.append(f"- {r['setting']}, {r['model']}, seed {r['seed']}, {r['prompt']}: {r['n_failure']} failed rows, {r['n_request_error']} request errors; {r['n_failure_fallback_label_correct']} fallback labels happened to match gold.")
    flags=[r for r in seed_metrics if r['n_non_p5_abstain_flags']]
    if flags:
        lines+=['','### Non-P5 abstention flags',
            'These valid-label rows are scored by their labels for canonical classification. A flag outside P5 does not redefine the P3 confidence-selection experiment.']
        for r in flags:lines.append(f"- {r['setting']}, {r['model']}, seed {r['seed']}, {r['prompt']}: {r['n_non_p5_abstain_flags']} valid-label abstention flag(s).")
    if comparison:
        n=sum(not r['matches_to_four_decimals'] for r in comparison)
        lines+=['','## Manuscript Table 11 audit',f'{n} of {len(comparison)} cells differ from the condition specified in its prose. See manuscript_table_checks.csv. This identifies disagreement with the supplied prediction archives, not the origin of the old numbers.']
    (out/'Verified_findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(args):
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    paths={}
    for role,stem in SOURCE_STEMS.items():
        explicit=getattr(args,role+'_archive',None)
        matches=[explicit] if explicit else list(args.source_dir.rglob(stem+'*.zip'))
        fail(len(matches)==1,f'Expected exactly one {stem}*.zip; pass an explicit --{role.replace("_","-")}-archive if needed')
        fail(zipfile.is_zipfile(matches[0]),f'Incomplete or invalid ZIP: {matches[0]}')
        paths[role]=matches[0]
    archives={key:zipfile.ZipFile(path) for key,path in paths.items()}
    conditions={};seed_metrics=[];metric_checks=[];row_issues=[];alignment=[];retry=[]
    try:
        for role,z in archives.items():fail(z.testzip() is None,f'Archive CRC failed: {role}')
        saved=archived_metric_index(archives['climate'],archives['yelp_metrics'])
        for setting in SETTINGS:
            for seed in SEEDS:
                reference=None
                for model in MODELS:
                    for prompt in PROMPTS:
                        key=(setting,model,seed,prompt)
                        role='climate' if setting.startswith('Climate') else 'yelp'
                        rows,labels,member=load_condition(archives[role],setting,model,seed,prompt)
                        identities=[(r['id'],r['hash'],r['gold']) for r in rows]
                        if reference is None:reference=identities
                        fail(identities==reference,f'Unmatched examples across prompt/model cells: {key}')
                        conditions[key]=rows
                        scores=condition_scores(rows,labels)
                        if prompt=='p5_abstain':scores['n_non_p5_abstain_flags']=0
                        else:scores['n_p5_abstain_confidence_above_half']=0
                        fail(key in saved,f'Archived summary missing for {key}')
                        metric_checks.extend(check_archived(scores,saved[key],key))
                        seed_metrics.append(dict(setting=setting,model=model,seed=seed,prompt=prompt,
                            archive_role=role,member=member,decode_seeds=';'.join(sorted({r['decode_seed'] for r in rows if r['decode_seed']})),**scores))
                        for r in rows:
                            if r['request_failed']:
                                retry.append(dict(setting=setting,model=model,sampling_seed=seed,prompt=prompt,
                                    example_id=r['id'],text_hash=r['hash'],gold_label=r['gold'],decode_seed=r['decode_seed'],
                                    archive_role=role,member=member,
                                    note='Retry requires original text and identical recorded inference configuration; never replace this row with a different sampled example.'))
                            if not r['usable'] or r['raw_status']!='ok' or r['raw_saved_disagreement'] or (prompt!='p5_abstain' and r['abstain']) or (prompt=='p5_abstain' and r['abstain'] and r['confidence']>.5):
                                row_issues.append(dict(setting=setting,model=model,seed=seed,prompt=prompt,
                                    example_id=r['id'],text_hash=r['hash'],saved_status=r['status'],request_failed=r['request_failed'],
                                    raw_status=r['raw_status'],raw_saved_disagreement=r['raw_saved_disagreement'],
                                    saved_abstain=r['abstain'],saved_confidence=r['confidence']))
                alignment.append(dict(setting=setting,seed=seed,n_matched_examples=len(reference),n_model_prompt_cells=6,
                    all_six_identity_hash_gold_lists_equal=True))
    finally:
        for z in archives.values():z.close()
    fail(len(seed_metrics)==90,'Expected 90 reference-period conditions')
    anchor_checks=canonical_checks(args.canonical_metrics,seed_metrics)
    summary=summaries(seed_metrics)
    ranges,rs,effects,es,common=prompt_comparisons(conditions,seed_metrics)
    abstention=p5_rows(seed_metrics)
    overlap=[]
    for setting in SETTINGS:
        hashes=[r['hash'] for seed in SEEDS for r in conditions[setting,MODELS[0],seed,PROMPTS[1]]]
        overlap.append(dict(setting=setting,rows_across_seeds=len(hashes),unique_texts=len(set(hashes)),repeated_occurrences=len(hashes)-len(set(hashes))))
    comparison=manuscript_checks(args.manuscript_tex,summary,rs)
    outputs={'seed_metrics.csv':seed_metrics,'metric_summary.csv':summary,'archived_metric_checks.csv':metric_checks,
        'alignment_checks.csv':alignment,'row_issues.csv':row_issues,'sampling_overlap.csv':overlap,
        'configuration_ranges_per_seed.csv':ranges,'configuration_ranges_summary.csv':rs,
        'paired_prompt_effects_per_seed.csv':effects,'paired_prompt_effects_summary.csv':es,
        'common_matched_scores.csv':common,'p5_abstention_per_seed.csv':abstention,
        'manuscript_table_checks.csv':comparison,'canonical_anchor_checks.csv':anchor_checks,
        'request_retry_manifest.csv':retry}
    for name,rows in outputs.items():write_csv(out/name,rows)
    audit=dict(condition_count=len(seed_metrics),attempted_rows=sum(r['n_attempted'] for r in seed_metrics),
        archived_metric_checks=len(metric_checks),maximum_archived_metric_difference=max(r['absolute_difference'] for r in metric_checks),
        canonical_anchor_checks=len(anchor_checks),
        failure_rows=sum(r['n_failure'] for r in seed_metrics),raw_saved_disagreements=sum(r['n_raw_saved_disagreement'] for r in seed_metrics),
        saved_success_raw_invalid=sum(1 for rows in conditions.values() for r in rows if r['usable'] and r['raw_status']!='ok'),
        seeds=list(SEEDS),reference_period='within',models=list(MODELS),prompts=list(PROMPTS),
        canonical_anchor='P3 only; no averaging across prompts for canonical ECE or latency.',
        coverage_denominator='All attempted rows; failure + valid abstention + retained fractions sum to one.',
        range_aggregation='Compute max-minus-min across prompts within each seed, then mean and sample SD across seeds.',
        uncertainty='Seed sample SD only, not independent-replicate standard errors or population confidence intervals.',
        provenance_limit='Input hashes identify supplied archives; they do not prove model digest, dataset annotation provenance, hardware, or absence of pretraining exposure.',
        inputs=[dict(role=role,name=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for role,p in paths.items()],
        manuscript_sha256=hashlib.sha256(args.manuscript_tex.read_bytes()).hexdigest() if args.manuscript_tex else None,
        canonical_anchor_sha256=hashlib.sha256(args.canonical_metrics.read_bytes()).hexdigest() if args.canonical_metrics else None,
        python=sys.version.split()[0])
    (out/'Audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
    fail(audit['raw_saved_disagreements']==0 and audit['saved_success_raw_invalid']==0,
         'Independent raw-response integrity audit requires review; inspect row_issues.csv. Outputs are not certified.')
    tex_outputs(out,summary,rs);report(out,seed_metrics,summary,rs,comparison,audit)
    if not args.no_figures:plot_outputs(out,seed_metrics,common)
    print(f"Verified {len(seed_metrics)} conditions, {audit['attempted_rows']:,} attempted rows, {len(metric_checks)} archived metrics.")
    print(f"Failed rows: {audit['failure_rows']}; raw/saved value disagreements: {audit['raw_saved_disagreements']}.")
    print('Outputs:',out)
    return audit


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True,help='Folder containing the three saved-prediction/metric ZIPs.')
    p.add_argument('--output-dir',type=Path,default=Path('RQ3_Verified_Output'))
    for role in SOURCE_STEMS:p.add_argument('--'+role.replace('_','-')+'-archive',type=Path)
    p.add_argument('--manuscript-tex',type=Path,help='Optional manuscript containing tab:frozen-reliability.')
    p.add_argument('--canonical-metrics',type=Path,help='Optional independently saved RQ1/RQ2 seed-metric CSV for P3 reference checks.')
    p.add_argument('--no-figures',action='store_true',help='Standard-library-only verification; skip matplotlib plots.')
    return p


if __name__=='__main__':
    run(parser().parse_args())
