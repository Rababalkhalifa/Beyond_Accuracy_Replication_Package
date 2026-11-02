#!/usr/bin/env python3
"""Reproduce the three-panel RQ2 uncertainty-contrast figure from saved bootstrap CSV.

Does not rerun the bootstrap or invent confidence intervals.

PowerShell:
 python plot_rq2_uncertainty_contrasts.py `
   --input "C:\\Research\\RQ2_Uncertainty_Improved_V2\\RQ2_focused_cluster_bootstrap.csv" `
   --output-dir "C:\\Research\\RQ2_Uncertainty_Improved_V2"

Requires: matplotlib, numpy.
"""
import argparse
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PERIODS = {'within': 'Reference', 'short': 'Short', 'long': 'Long'}
SECTIONS = [
    ('A  Yelp: Phi minus Qwen at 80%', '#1875A5',
     [('Yelp n500', 'within', 'phi_minus_qwen_at_80', 'n500 · Reference'),
      ('Yelp n500', 'short', 'phi_minus_qwen_at_80', 'n500 · Short'),
      ('Yelp n500', 'long', 'phi_minus_qwen_at_80', 'n500 · Long'),
      ('Yelp n1000', 'within', 'phi_minus_qwen_at_80', 'n1000 · Reference'),
      ('Yelp n1000', 'short', 'phi_minus_qwen_at_80', 'n1000 · Short'),
      ('Yelp n1000', 'long', 'phi_minus_qwen_at_80', 'n1000 · Long')]),
    ('B  Climate: Qwen 80% minus 100%', '#C16A23',
     [('Climate n500', p, 'qwen_80_minus_100', PERIODS[p])
      for p in ('within','short','long')]),
    ('C  Climate: DistilBERT temporal change', '#17835D',
     [('Climate n500', 'short', 'distilbert_period_minus_reference_at_80', 'Short − Reference'),
      ('Climate n500', 'long', 'distilbert_period_minus_reference_at_80', 'Long − Reference')])
]

def parse_data(path):
    expected = {'setting','period','contrast','mean_difference_pp','ci_95_low_pp','ci_95_high_pp'}
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not expected.issubset(set(reader.fieldnames or [])):
            raise ValueError('Missing fields: '+', '.join(sorted(expected - set(reader.fieldnames or []))))
        out = {}
        for row in reader:
            key = (row['setting'],row['period'],row['contrast'])
            if key in out: raise ValueError(f'Duplicate comparison: {key}')
            x,lo,hi = (float(row[f]) for f in ('mean_difference_pp','ci_95_low_pp','ci_95_high_pp'))
            if not all(map(math.isfinite,(x,lo,hi))) or not lo<=x<=hi:
                raise ValueError(f'Invalid point or confidence interval: {key}')
            out[key] = (x,lo,hi)
    required = {(s,p,c) for _,_,items in SECTIONS for s,p,c,_ in items}
    if set(out) != required:
        raise ValueError(f'Contrasts mismatch; missing={sorted(required-set(out))}; extra={sorted(set(out)-required)}')
    return out

def create_figure(rows, output_dir, dpi=240):
    fig,axes = plt.subplots(1,3,figsize=(13.6,5.2),gridspec_kw={'width_ratios':[1.55,1,0.9]})
    for ax,(title,color,items) in zip(axes,SECTIONS):
        values = np.array([rows[(setting,period,contrast)] for setting,period,contrast,_ in items])
        x,lo,hi = values.T
        y = np.arange(len(items))[::-1]
        ax.errorbar(x,y,xerr=[x-lo,hi-x],fmt='o',markersize=5.8,color=color,
                    ecolor=color,elinewidth=1.5,capsize=3,zorder=3)
        ax.axvline(0,color='#5C646E',linewidth=1,linestyle='--')
        ax.set_yticks(y,[label for *_,label in items],fontsize=9)
        ax.set_title(title,fontsize=10.5,loc='left',weight='bold',pad=12)
        ax.grid(axis='x',alpha=.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x',labelsize=9)
        for spine in ('top','right','left'): ax.spines[spine].set_visible(False)
        ax.set_ylim(-.65,len(items)-.35)
        ax.set_xlabel('Retained-error difference (percentage points)',fontsize=9)
        ax.set_xlim(min(lo.min()-.2,-.2),max(hi.max()+.2,.2))
    # fig.suptitle('RQ2: focused retained-error contrasts with document-cluster bootstrap intervals',
    #              fontsize=13,weight='bold',x=.035,ha='left')
    fig.subplots_adjust(left=.13,right=.99,top=.81,bottom=.19,wspace=.43)
    # fig.text(.035,.055,'Points: five-seed mean differences. Whiskers: conditional, unadjusted 95% percentile bootstrap intervals (3,000 replicates).',
    #          fontsize=9,color='#444C55')
    # fig.text(.035,.023,'Intervals account for repeated exact-document records; they do not cover new training seeds, new time periods, or multiple testing.',
    #          fontsize=8.7,color='#444C55')
    output_dir.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','svg','png'):
        path=output_dir/f'RQ2_uncertainty_contrasts.{ext}'
        fig.savefig(path,bbox_inches='tight',dpi=dpi if ext=='png' else None)
        print('Wrote',path)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input',type=Path,required=True,help='RQ2_focused_cluster_bootstrap.csv')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--dpi',type=int,default=240)
    args=parser.parse_args()
    create_figure(parse_data(args.input),args.output_dir,args.dpi)

if __name__=='__main__':main()
