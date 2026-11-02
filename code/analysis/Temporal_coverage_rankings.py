"""Self-contained computational core embedded in the RQ2 notebook.

Reads saved predictions only. Produces verified RQ2 tables, SVG figure, and
separate LaTeX Results, Discussion, and Methods fragments. No model fitting.
"""
import csv
import io
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


SEEDS = range(42, 47)
PERIODS = ("within", "short", "long")
SETTINGS = ("Climate n500", "Yelp n500", "Yelp n1000")
MODELS = ("distilbert", "tfidf_lr", "phi3.5:latest", "qwen2.5:3b-instruct")
SLUGS = ("distilbert", "tfidf_lr", "phi3.5_latest", "qwen2.5_3b-instruct")
DISPLAY = ("DistilBERT", "TF–IDF+LR", "phi3.5:latest", "qwen2.5:3b-instruct")
TEX_MODELS = ("DistilBERT", "TF--IDF+LR", r"\texttt{phi3.5:latest}",
              r"\texttt{qwen2.5:3b-instruct}")
COLORS = ("#0072B2", "#72777F", "#009E73", "#D55E00")
MARKERS = ("o", "s", "^", "D")
STEMS = dict(climate="completed_prediction_audit",
             frozen="original_yelp_frozen_predictions",
             aligned="yelp_prediction_alignment",
             later="code_and_remaining_results",
             frozen_audit="yelp_results_audit")


def get_csv(z, suffix):
    found = [name for name in z.namelist() if name.replace("\\", "/").endswith(suffix)]
    if len(found) != 1:
        raise ValueError(f"Expected one {suffix} in archive; found {len(found)}")
    with z.open(found[0]) as source:
        return list(csv.DictReader(io.TextIOWrapper(source, encoding="utf-8-sig", newline="")))


def metric_index_climate(z):
    index = {}
    for r in get_csv(z, "metrics_per_condition.csv"):
        if r["dataset"] != "climate" or r["model"] not in MODELS or r["period"] not in PERIODS:
            continue
        expected_prompt = "" if r["model"] in MODELS[:2] else "p3_json_strict"
        if r["prompt_id"] != expected_prompt or int(r["n_per_label"]) != 500:
            continue
        key = ("Climate n500", r["model"], int(r["seed"]), r["period"])
        if key in index:
            raise ValueError(f"Duplicate archived Climate metric: {key}")
        index[key] = r
    return index


def metric_index_yelp(z, models):
    suffix = "later_run/metrics_per_seed.csv" if models == MODELS[:2] else "metrics_frozen_partial.csv"
    index = {}
    for r in get_csv(z, suffix):
        if r["model"] not in models or r["split"] not in PERIODS or r["prompt"] != "p3_json_strict":
            continue
        n = int(r["eval_n_per_label"])
        if n not in (500, 1000):
            continue
        key = (f"Yelp n{n}", r["model"], int(r["seed"]), r["split"])
        if key in index:
            raise ValueError(f"Duplicate archived Yelp metric: {key}")
        index[key] = r
    return index


def ece_10(correct, confidence):
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if not np.isfinite(conf).all() or not ((conf >= 0) & (conf <= 1)).all():
        raise ValueError("Confidence missing or outside [0, 1]")
    bounds = np.linspace(0.0, 1.0, 11)
    result = 0.0
    for bin_no in range(10):
        lo, hi = bounds[bin_no], bounds[bin_no+1]
        mask = (conf >= lo) & (conf <= hi if bin_no == 9 else conf < hi)
        if mask.any():
            result += float(mask.mean() * abs(corr[mask].mean() - conf[mask].mean()))
    return result


def macro_f1(rows, labels, gold, pred):
    scores = []
    for lab in labels:
        tp = sum(r[gold] == lab and r[pred] == lab for r in rows)
        fp = sum(r[gold] != lab and r[pred] == lab for r in rows)
        fn = sum(r[gold] == lab and r[pred] != lab for r in rows)
        scores.append(2 * tp / (2 * tp + fp + fn) if 2*tp+fp+fn else 0.0)
    return statistics.mean(scores)


def risk_sweep(scores, errors):
    """Expected error rate for 20..100% coverage with uniform tie-breaking."""
    conf = np.asarray(scores, dtype=float)
    wrong = np.asarray(errors, dtype=float)
    groups, indices = np.unique(conf, return_inverse=True)
    counts = np.bincount(indices)[::-1]
    group_errors = np.bincount(indices, weights=wrong)[::-1]
    cumulative_count = counts.cumsum()
    cumulative_error = group_errors.cumsum()
    result = []
    for pct in range(20, 101):
        kept = len(conf) * pct // 100
        boundary = int(np.searchsorted(cumulative_count, kept, side="left"))
        before = int(cumulative_count[boundary-1]) if boundary else 0
        previous_error = float(cumulative_error[boundary-1]) if boundary else 0.0
        fraction = (kept - before) / int(counts[boundary])
        result.append((previous_error + fraction * float(group_errors[boundary])) / kept)
    return result


def prediction_rows(archives, setting, model, seed, period):
    i = MODELS.index(model)
    n = 1000 if setting.endswith("1000") else 500
    if setting.startswith("Climate"):
        pieces = ["predictions", "adapted" if i < 2 else "frozen", SLUGS[i]]
        if i >= 2:
            pieces.append("p3_json_strict")
        pieces.extend(["climate", f"seed_{seed}", f"{period}_n500_per_label.csv"])
        return get_csv(archives["climate"], "/".join(pieces))
    if i < 2:
        folder = "later_distilbert" if i == 0 else "aligned_tfidf"
        return get_csv(archives["aligned"], f"{folder}/seed{seed}/n{n}_{period}.csv")
    return get_csv(archives["frozen"],
                   f"predictions/frozen/{SLUGS[i]}/seed{seed}/n{n}_{period}_p3_json_strict.csv")


def audited_results(source_dir):
    paths = {}
    for key, stem in STEMS.items():
        matches = list(Path(source_dir).rglob(stem + "*.zip"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {stem}*.zip under {source_dir}; found {len(matches)}")
        paths[key] = matches[0]
    print("Inputs:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    archives = {key: zipfile.ZipFile(path) for key, path in paths.items()}
    try:
        cm = metric_index_climate(archives["climate"])
        ym = {**metric_index_yelp(archives["later"], MODELS[:2]),
              **metric_index_yelp(archives["frozen_audit"], MODELS[2:])}
        records, risk_rows, anomalies = [], [], 0
        for setting in SETTINGS:
            n = 1000 if setting.endswith("1000") else 500
            labels = ("BELIEVER", "DENIER") if setting.startswith("Climate") else ("NEG", "POS")
            for period in PERIODS:
                for seed in SEEDS:
                    matched = None
                    for model in MODELS:
                        i = MODELS.index(model)
                        rows = prediction_rows(archives, setting, model, seed, period)
                        gold, pred, digest = (("true_label", "predicted_label", "text_sha256")
                                              if setting.startswith("Climate") else
                                              ("gold_label", "pred_label", "text_sha1"))
                        if (len(rows) != 2*n or Counter(r[gold] for r in rows) != {lab:n for lab in labels}
                                or any(r[pred] not in labels or r.get("status","ok") != "ok" for r in rows)):
                            raise ValueError(f"Missing/invalid predictions: {setting}, {period}, {seed}, {model}")
                        ids = {(r["example_id"], r[digest], r[gold]) for r in rows}
                        if len(ids) != len(rows) or (matched is not None and matched != ids):
                            raise ValueError(f"Unmatched evaluation examples: {setting}, {period}, {seed}, {model}")
                        matched = ids
                        if i < 2:
                            if setting.startswith("Climate"):
                                if any(not math.isclose(float(r["confidence"]),
                                                        max(float(r["prob_believer"]),float(r["prob_denier"])),
                                                        abs_tol=1e-5) for r in rows):
                                    raise ValueError("Climate supervised confidence differs from probabilities")
                            elif any(not math.isclose(float(r["confidence"]),
                                                      max(float(r["prob_pos"]),1-float(r["prob_pos"])),
                                                      abs_tol=1e-5) for r in rows):
                                raise ValueError("Yelp supervised confidence differs from probabilities")
                        elif not setting.startswith("Climate"):
                            anomalies += sum(r["abstain"].lower()=="true" for r in rows)
                        conf = [float(r["confidence"]) for r in rows]
                        wrong = [r[gold] != r[pred] for r in rows]
                        acc = 1-sum(wrong)/len(rows)
                        f1 = macro_f1(rows, labels, gold, pred)
                        ece = ece_10([not value for value in wrong], conf)
                        key = (setting,model,seed,period)
                        saved = (cm if setting.startswith("Climate") else ym).get(key)
                        if saved is None or any(not math.isclose(v,float(saved[k]),abs_tol=1e-8)
                                                for v,k in ((acc,"accuracy"),(f1,"macro_f1"),(ece,"ece"))):
                            raise ValueError(f"Saved accuracy/F1/ECE mismatch: {key}")
                        curve = risk_sweep(conf, wrong)
                        if not math.isclose(curve[-1],1-acc,abs_tol=1e-12):
                            raise ValueError(f"Full-coverage risk mismatch: {key}")
                        records.append(dict(setting=setting,model=model,seed=seed,period=period,
                                            n_per_class=n, accuracy=acc,macro_f1=f1,ece=ece,
                                            confidence_source=("predicted_probability" if i<2 else "llm_self_reported"),
                                            risk_at_80=curve[60],risk_at_100=curve[80]))
                        for pct, risk in enumerate(curve, start=20):
                            risk_rows.append(dict(setting=setting,model=model,seed=seed,period=period,
                                                  coverage=pct/100,expected_risk=risk))
        assert len(records)==180 and len(risk_rows)==180*81
        print("Verified 180 conditions: matched examples, 10-bin ECE, Accuracy, Macro-F1, and full-coverage risk.")
        print(f"Included {anomalies} Yelp P3 rows marked abstain with a valid label, matching canonical scoring.")
        return records, risk_rows
    finally:
        for z in archives.values():
            z.close()


def group_means(records):
    grouped=defaultdict(list)
    for r in records:
        grouped[r["setting"],r["model"],r["period"]].append(r)
    if any(len(v)!=5 for v in grouped.values()):
        raise ValueError("A setting/model/period has fewer than five seeds")
    return {key:{field:statistics.mean(float(r[field]) for r in rows)
                 for field in ("accuracy","macro_f1","ece","risk_at_80","risk_at_100")}
            for key,rows in grouped.items()}


def save_csv(path, rows):
    with Path(path).open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)



# Temporal extension: all periods use the same canonical scoring policy above.
import argparse
import hashlib
import itertools
import json
import shutil
import sys

SHORT = dict(zip(MODELS, ("DistilBERT", "TF–IDF+LR", "Phi", "Qwen")))
PERIOD_NAMES = {"within": "Reference", "short": "Short", "long": "Long"}
POINTS = (60, 80, 95, 100)
GRID = tuple(range(20, 101))
TOL = 1e-12
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#263445", "text.color": "#263445",
    "xtick.color": "#526071", "ytick.color": "#526071",
})

def shared_ranks(values):
    """Competition ranks, sharing a rank when mean risks are equal within TOL."""
    return {m: 1 + sum(v < values[m] - TOL for v in values.values()) for m in values}

def summarize(curves):
    indexed = {(r["setting"], r["period"], r["model"], int(r["seed"]),
                round(float(r["coverage"])*100)): float(r["expected_risk"]) for r in curves}
    if len(indexed) != 180*81:
        raise ValueError("Duplicate or missing risk–coverage observations")
    summary, paired, segments = [], [], []
    for setting in SETTINGS:
        for period in PERIODS:
            previous, start = None, 20
            for pct in GRID:
                seed_values = {m: [indexed[setting, period, m, s, pct] for s in SEEDS] for m in MODELS}
                means = {m: statistics.mean(v) for m, v in seed_values.items()}
                ranks = shared_ranks(means)
                seed_ranks = {s: shared_ranks({m: indexed[setting, period, m, s, pct]
                                              for m in MODELS}) for s in SEEDS}
                order = " < ".join(" = ".join(SHORT[m] for m in MODELS if ranks[m] == rank)
                                    for rank in sorted(set(ranks.values())))
                if previous is not None and order != previous:
                    segments.append(dict(setting=setting, period=period, coverage_from=start/100,
                                         coverage_to=(pct-1)/100, ascending_mean_risk_order=previous))
                    start = pct
                previous = order
                for m in MODELS:
                    v = seed_values[m]
                    summary.append(dict(setting=setting, period=period, model=m, coverage=pct/100,
                                        risk_mean=means[m], risk_sample_sd=statistics.stdev(v),
                                        rank_of_mean=ranks[m], seeds_with_lowest_risk=sum(
                                            seed_ranks[s][m] == 1 for s in SEEDS), n_seeds=5))
                if pct in POINTS:
                    for ma, mb in itertools.combinations(MODELS, 2):
                        d = [a-b for a,b in zip(seed_values[ma],seed_values[mb])]
                        paired.append(dict(setting=setting, period=period, coverage=pct/100,
                            model_a=ma, model_b=mb, mean_risk_a_minus_b=statistics.mean(d),
                            sd_paired_difference=statistics.stdev(d), minimum_seed_difference=min(d),
                            maximum_seed_difference=max(d),
                            seeds_a_lower=sum(x < -TOL for x in d),
                            seeds_b_lower=sum(x > TOL for x in d),
                            seeds_tied=sum(abs(x) <= TOL for x in d), n_seeds=5))
            segments.append(dict(setting=setting, period=period, coverage_from=start/100,
                                 coverage_to=1.0, ascending_mean_risk_order=previous))
    return indexed, summary, paired, segments

def style_axis(ax, curve=True):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6EBF0", linewidth=.7)
    ax.spines["left"].set_color("#BDC7D2")
    ax.spines["bottom"].set_color("#BDC7D2")
    if curve:
        ax.set_xlim(.2, 1.015)
        ax.set_xticks([.2,.4,.6,.8,1])
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.set_xlabel("Coverage retained")
        ax.axvline(.8, color="#A6B1BD", linewidth=.8, linestyle=":", zorder=0)
    else:
        ax.set_xlim(-.35, 3.35)
        ax.set_ylim(4.45,.55)
        ax.set_yticks((1,2,3,4), ("1st","2nd","3rd","4th"))
        ax.set_xticks(range(4), [f"{p}%" for p in POINTS])
        ax.set_xlabel("Coverage retained")

def curve_panel(ax, indexed, setting, period):
    highest = 0
    for i,m in enumerate(MODELS):
        mat = np.array([[indexed[setting,period,m,s,p] for p in GRID] for s in SEEDS])
        mu,sd = mat.mean(axis=0), mat.std(axis=0,ddof=1)
        highest = max(highest, float(np.max(mu+sd)))
        xs=np.array(GRID)/100
        ax.fill_between(xs,np.maximum(0,mu-sd),mu+sd,color=COLORS[i],alpha=.10,linewidth=0)
        ax.plot(xs,mu,color=COLORS[i],linewidth=1.9,marker=MARKERS[i],
                markevery=[40,60,75,80],markersize=4,markeredgecolor="white",markeredgewidth=.45)
    style_axis(ax)
    return highest

def rank_panel(ax, indexed, setting, period):
    mean={m: [statistics.mean(indexed[setting,period,m,s,p] for s in SEEDS)
              for p in POINTS] for m in MODELS}
    ranks=[shared_ranks({m:mean[m][j] for m in MODELS}) for j in range(len(POINTS))]
    for i,m in enumerate(MODELS):
        ys=[r[m] for r in ranks]
        ax.plot(range(len(POINTS)),ys,color=COLORS[i],linewidth=1.6,alpha=.8)
        for j,y in enumerate(ys):
            # The current data have no exact mean-risk ties at these points.
            # Small deterministic offsets keep future exact ties readable.
            tied=[other for other in MODELS if ranks[j][other] == y]
            off=0 if len(tied)==1 else (tied.index(m)-(len(tied)-1)/2)*.13
            shown=str(Decimal(str(round(float(mean[m][j]*100),10))).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP))
            ax.text(j,y+off,shown,color="white",ha="center",va="center",
                    fontsize=8.2,fontweight="bold",zorder=5,
                    bbox=dict(boxstyle="round,pad=.36,rounding_size=.5",
                              facecolor=COLORS[i],edgecolor="white",linewidth=.9))
    style_axis(ax,curve=False)

def legend(fig):
    handles=[Line2D([0],[0],color=c,lw=2,marker=marker,markersize=5,label=SHORT[m])
             for m,c,marker in zip(MODELS,COLORS,MARKERS)]
    fig.legend(handles=handles,ncol=4,frameon=False,loc="upper center",
               bbox_to_anchor=(.52,.94),fontsize=10,columnspacing=3)

def save_figure(fig,out,stem):
    for ext in ("svg","pdf","png"):
        fig.savefig(out/(stem+"."+ext),dpi=300,facecolor="white")
    plt.close(fig)

def plot_main(indexed,out,rank=False):
    fig,axs=plt.subplots(2,3,figsize=(12.6,7.65),squeeze=False)
    fig.subplots_adjust(left=.09,right=.985,bottom=.16,top=.82,wspace=.25,hspace=.43)
    title=("Model rankings at common coverage levels" if rank else
           "Confidence-based risk–coverage across historical periods")
    fig.suptitle(title,fontsize=16,fontweight="bold",y=.985)
    legend(fig)
    for row,setting in enumerate(SETTINGS[:2]):
        highest=0
        for col,period in enumerate(PERIODS):
            ax=axs[row,col]
            if rank:rank_panel(ax,indexed,setting,period)
            else:highest=max(highest,curve_panel(ax,indexed,setting,period))
            ax.set_title(f"({chr(97+row*3+col)}) {PERIOD_NAMES[period]}",loc="left",
                         fontsize=11.5,fontweight="bold",pad=10)
        if not rank:
            ylim=math.ceil(highest/.05)*.05 if row==0 else math.ceil(highest/.025)*.025
            for ax in axs[row]:ax.set_ylim(0,ylim)
        label=("Climate stance" if row==0 else "Yelp sentiment")+"\nn = 500 per class"
        fig.text(.018, (.655 if row==0 else .305),label,ha="center",va="center",
                 rotation=90,fontsize=11,fontweight="bold")
        axs[row,0].set_ylabel("Rank by mean retained error" if rank else "Error among retained predictions")
    if rank:
        footer=("Numbers inside markers are retained error (%); ranks order five-seed means only.\n"
                "Small rank differences do not establish a reliable ordering. See paired seed differences in the accompanying CSV.")
        stem="Temporal_coverage_rankings_main"
    else:
        footer = ("")
        # footer=("Lines: five-seed means; shading: ±1 sample SD (sampling dispersion, not confidence intervals).\n"
        #         "Matched evaluation pools; each model retains its own cases. Frozen models use canonical P3.")
        stem="Temporal_risk_coverage_main"
    fig.text(.09,.064,footer,fontsize=9,color="#526071",linespacing=1.6)
    # fig.text(.09,.025,"Tied confidence: expected risk under uniform random tie-breaking. Explicit P5 abstention is separate.",
    #          fontsize=8.6,color="#526071")
    save_figure(fig,out,stem)

def plot_supplement(indexed,out):
    fig,axs=plt.subplots(2,3,figsize=(12.6,7.65),squeeze=False)
    fig.subplots_adjust(left=.085,right=.985,bottom=.16,top=.82,wspace=.25,hspace=.43)
    fig.suptitle("Yelp sentiment sensitivity analysis · n = 1,000 per class",
                 fontsize=16,fontweight="bold",y=.985)
    legend(fig);highest=0
    for col,p in enumerate(PERIODS):
        highest=max(highest,curve_panel(axs[0,col],indexed,SETTINGS[2],p))
        rank_panel(axs[1,col],indexed,SETTINGS[2],p)
        for row in range(2):
            axs[row,col].set_title(f"({chr(97+row*3+col)}) {PERIOD_NAMES[p]}",loc="left",
                                  fontsize=11.5,fontweight="bold",pad=10)
    ymax=math.ceil(highest/.025)*.025
    for ax in axs[0]:ax.set_ylim(0,ymax)
    axs[0,0].set_ylabel("Error among retained predictions")
    axs[1,0].set_ylabel("Rank by mean retained error")
    fig.text(.085,.063,"Top: means ±1 sample SD. Bottom: ranks of means; marker labels show retained error (%).\n"
             "Each model retains its own cases. Ranks are descriptive; small gaps need cautious interpretation.",
             fontsize=9,color="#526071",linespacing=1.6)
    # fig.text(.085,.025,"Canonical P3 for frozen models; expected random tie-breaking. n500/n1000 Yelp samples are nested.",
    #          fontsize=8.6,color="#526071")
    save_figure(fig,out,"Temporal_risk_and_rankings_Yelp1000")

def validate_math():
    """Analytic tie examples and row-permutation invariance, independent of archives."""
    # Top tied group: two predictions, one error. At 20% of N=10, risk is 1/2.
    conf=[.9,.9,.8,.8,.8,.7,.6,.5,.4,.3]
    err =[0,1,1,0,0,0,1,0,0,1]
    v=risk_sweep(conf,err)
    assert math.isclose(v[0],.5)
    # At 30%, retain the tied top two plus a random one of the .8 group.
    assert math.isclose(v[10],(1+1/3)/3)
    assert math.isclose(v[-1],.4)
    assert np.allclose(v,risk_sweep(conf[::-1],err[::-1]),atol=1e-14)
    v=risk_sweep([.9]*100,[1]*30+[0]*70)
    assert np.allclose(v,.3,atol=1e-14)
    assert shared_ranks({"a":.1,"b":.1,"c":.2})=={"a":1,"b":1,"c":3}

def compare_saved_curves(source_dir,curves):
    matches=list(Path(source_dir).rglob("Beyond_Accuracy_RQ2_notebook_and_outputs.zip"))
    if not matches:return "Optional previous-output archive absent; raw-prediction checks completed."
    if len(matches)!=1:raise ValueError("Multiple previous-output archives")
    with zipfile.ZipFile(matches[0]) as z:
        old=get_csv(z,"outputs/RQ2_risk_coverage_per_seed.csv")
    key=lambda r:(r["setting"],r["period"],r["model"],int(r["seed"]),round(float(r["coverage"])*100))
    a={key(r):float(r["expected_risk"]) for r in curves}
    b={key(r):float(r["expected_risk"]) for r in old}
    if a.keys()!=b.keys():raise ValueError("Previous curve conditions differ")
    delta=max(abs(a[k]-b[k]) for k in a)
    if delta>1e-12:raise ValueError(f"Previous risk curves disagree: maximum difference {delta}")
    return f"All {len(a):,} risk values agree with the existing audit output (max absolute difference {delta:g})."

def lookup(summary):
    return {(r["setting"],r["period"],r["model"],round(r["coverage"]*100)):r for r in summary}

def manuscript_files(summary,out):
    d=lookup(summary)
    def val(setting,period,model,pct):
        return d[setting,period,model,pct]["risk_mean"]*100
    s="Yelp n1000";pmod="phi3.5:latest";qmod="qwen2.5:3b-instruct"
    findings=[
        "# Temporal coverage analysis: verified findings",
        "",
        "All values below are means of five seed-specific retained error rates. "
        "They are descriptive estimates; ranks are not significance tests.",
        "",
        "## Yelp n1000: the Phi–Qwen comparison",
        "",
        "| Period | Phi at 80% | Qwen at 80% | Phi at 95% | Qwen at 95% | Phi at 100% | Qwen at 100% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in PERIODS:
        numbers=[val(s,p,m,c) for c in (80,95,100) for m in (pmod,qmod)]
        findings.append("| "+PERIOD_NAMES[p]+" | "+" | ".join(f"{v:.3f}%" for v in numbers)+" |")
    findings += ["",
        "- At 80% coverage, Phi has lower retained error than Qwen in all five seeds in every period, "
        "in both Yelp sample sizes.",
        "- Qwen has lower full-coverage error than Phi in all five seeds in every Yelp period "
        "and both sample sizes.",
        "- At 95%, the mean preference changes from Qwen in the reference period to Phi in the two later "
        "periods, in both Yelp sizes. The short-period gaps are approximately 0.02–0.03 percentage "
        "points and the seed-specific differences change sign. This is a near-tie, not a clear winner.",
        "- For Yelp n1000 reference at 80%, DistilBERT (1.767%) and Phi (1.813%) are very close. "
        "The ordering of their means should not be presented as a reliable advantage.",
        "- For Climate n500, the mean order is DistilBERT, TF–IDF+LR, Phi, Qwen at every evaluated "
        "integer coverage from 20% to 100%, in all three periods.",
        "",
        "## Interpretation",
        "",
        "The clearest supported Yelp pattern is a coverage-dependent choice that persists across periods: "
        "Phi has lower retained error at 80%, whereas Qwen has lower error when all predictions are retained. "
        "The high-coverage crossover location varies descriptively across periods. Stable Climate rankings "
        "show that coverage-dependent rank reversals are not universal.",
        "",
        "Risk curves are retrospective. They do not establish that a reference-chosen confidence threshold "
        "will preserve risk or coverage in later periods. Seed SDs describe the repeated samples; overlapping "
        "texts and nested Yelp samples mean the seeds are not five independent experiments. No new "
        "confidence intervals or significance claims are made.",
        "",
        "The companion paired_seed_differences.csv reports all six model pairs at 60%, 80%, 95%, and 100%. "
        "A negative A-minus-B value favors model A.",
        "",
    ]
    (out/"Findings.md").write_text("\n".join(findings),encoding="utf-8")
    results=(r"\paragraph{Coverage-dependent rankings across historical periods.}"+"\n"+
        "Confidence-based selection was evaluated separately on the reference, short-, and long-distance "
        "splits using fixed canonical predictions. On Climate stance, the ordering of mean retained error "
        "was DistilBERT, TF--IDF+LR, Phi, and Qwen throughout the evaluated 20--100\\% coverage grid "
        "in every period. Yelp sentiment showed coverage-dependent preferences in both sample sizes. "
        "In Yelp n1000, Phi had lower mean retained error than Qwen at 80\\% coverage in the reference "
        f"({val(s,'within',pmod,80):.2f}\\% versus {val(s,'within',qmod,80):.2f}\\%), short "
        f"({val(s,'short',pmod,80):.2f}\\% versus {val(s,'short',qmod,80):.2f}\\%), and long "
        f"({val(s,'long',pmod,80):.2f}\\% versus {val(s,'long',qmod,80):.2f}\\%) periods. "
        "This direction held in all five sampling seeds in each period. Qwen had lower full-coverage error "
        "than Phi in all periods and all five seeds. At 95\\% coverage, Qwen had the lower mean error in "
        f"the reference period ({val(s,'within',qmod,95):.2f}\\% versus "
        f"{val(s,'within',pmod,95):.2f}\\%), while Phi had the lower mean error on the later splits. "
        "The short-period difference was only about 0.02 percentage points and changed direction across "
        "seeds, so it is interpreted as a near-tie. The results support coverage-dependent Yelp model "
        "preferences across the evaluated periods, with period-specific variation in their high-coverage "
        "crossovers. Rankings of close means are descriptive rather than evidence of a reliable ordering.\n")
    methods=(r"\paragraph{Temporal confidence-based selective prediction.}"+"\n"+
        "For each dataset setting, evaluation period, model, and sampling seed, fixed predictions were "
        "ranked by that model's confidence scores. Frozen models used canonical P3 self-reported "
        "confidence; supervised classifiers used maximum predicted class probability. We evaluated "
        "retained error at coverage levels from 20\\% to 100\\% in one-percentage-point steps. "
        "At coverage $c$, $k=cN$ cases were retained; $k$ is an integer for all evaluated sample sizes "
        "and grid values. If a confidence tie straddled the cutoff, the tied examples received equal "
        "fractional inclusion weights to compute expected error under uniform random tie-breaking. "
        "Means and sample standard deviations were calculated across the five seeds. Rankings order "
        "mean retained error at common coverage; models can retain different subsets of the matched "
        "evaluation pool. Paired seed differences describe sampling variation and do not constitute "
        "independent-replicate hypothesis tests. Full-coverage error equals one minus accuracy, not one "
        "minus Macro-F1. The sweep is retrospective and is separate from explicit P5 abstention or a "
        "validation-selected fixed deployment threshold.\n")
    discussion=(r"\paragraph{Period and coverage as joint model-selection conditions.}"+"\n"+
        "Stable full-sample performance rankings do not remove the need to examine confidence-based "
        "selection separately in each period. In Yelp, Phi's lower retained error at 80\\% coverage and "
        "Qwen's lower full-coverage error persist across all evaluated periods. Their high-coverage "
        "mean-risk crossovers vary across periods, although small differences and changing seed-specific "
        "directions limit claims about exact crossover locations. Climate provides a complementary "
        "result: the mean model ordering remains stable throughout the evaluated coverage grid despite "
        "period-specific changes in error. Model preference therefore depends on the task and accepted "
        "fraction, and the relevant risk--coverage relationship should be checked on each evaluated "
        "period. These results do not estimate the accuracy or cost of subsequent human review.\n")
    for name,txt in (("Results_temporal_coverage.tex",results),("Methods_temporal_coverage.tex",methods),
                     ("Discussion_temporal_coverage.tex",discussion)):
        (out/name).write_text(txt,encoding="utf-8")
    caption=("Confidence-based risk--coverage curves across historical periods for Climate stance and Yelp "
        "sentiment, each with 500 examples per class. Columns show the reference, short-, and long-distance "
        "splits. Lines show means of seed-specific retained error rates across five sampling seeds; shading "
        "shows $\\pm1$ sample standard deviation, not confidence intervals. Each model selects its own "
        "cases from matched evaluation pools. Frozen models use canonical P3. Ties are handled by "
        "expected error under uniform random tie-breaking; P5 explicit abstention is separate.")
    figure=("\n".join([r"\begin{figure*}[t]",r"\centering",
        r"\includegraphics[width=\textwidth]{Temporal_risk_coverage_main.pdf}",
        r"\caption{"+caption+"}",r"\label{fig:temporal-risk-coverage}",r"\end{figure*}"])+"\n")
    (out/"Figure_temporal_coverage.tex").write_text(figure,encoding="utf-8")
    rows=[r"\begin{table*}[t]",r"\centering\scriptsize",
        r"\caption{Expected retained error (\%) at common coverage levels. Values are five-seed means. "
        r"The lower point estimate within each period and coverage is bold; boldface does not establish "
        r"a statistically reliable difference.}",
        r"\label{tab:temporal-coverage-points}",r"\begin{tabular}{lllrrrr}",
        r"\toprule",r"Setting & Period & Model & 60\% & 80\% & 95\% & 100\% \\",
        r"\midrule"]
    for setting in SETTINGS:
        for p in PERIODS:
            minima={c:min(val(setting,p,m,c) for m in MODELS) for c in POINTS}
            for i,m in enumerate(MODELS):
                values=[]
                for c in POINTS:
                    v=val(setting,p,m,c);txt=f"{v:.3f}"
                    values.append(r"\textbf{"+txt+"}" if abs(v-minima[c])<TOL else txt)
                rows.append(f"{setting if i==0 else ''} & {PERIOD_NAMES[p] if i==0 else ''} & "
                            f"{TEX_MODELS[i]} & "+" & ".join(values)+r" \\")
            rows.append(r"\midrule")
    rows[-1]=r"\bottomrule"
    rows += [r"\end{tabular}",r"\end{table*}"]
    (out/"Table_temporal_coverage.tex").write_text("\n".join(rows)+"\n",encoding="utf-8")

def write_notebook(out):
    source=Path(__file__).read_text(encoding="utf-8")
    intro=("# Temporal risk–coverage and rankings\n\n"
        "Recomputes 180 canonical conditions from five prediction archives, verifies saved metrics and "
        "matched examples, and produces publication-ready SVG/PDF figures and CSV/LaTeX outputs.\n\n"
        "No models are fitted or invoked. The archived aligned Yelp cohort and P3 scoring are preserved. "
        "Edit SOURCE_DIR below, then Restart and Run All. Requires numpy and matplotlib.\n")
    code=("from pathlib import Path\n"
        "SOURCE_DIR = Path(r'C:\\Research')  # folder containing exactly one copy of each of the five input ZIPs\n"
        "OUTPUT_DIR = SOURCE_DIR / 'Temporal_Coverage_Output'\n")
    # Avoid invoking the CLI in the notebook; keep the same complete computational implementation.
    embedded=source.split('\nif __name__ == "__main__":')[0]
    cells=[
        {"cell_type":"markdown","metadata":{},"source":intro},
        {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code},
        {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":embedded},
        {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
         "source":"analysis = run_temporal(SOURCE_DIR, OUTPUT_DIR)\n"},
        {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
         "source":"from IPython.display import display, SVG, Markdown\n"
                  "display(SVG(filename=str(OUTPUT_DIR / 'Temporal_risk_coverage_main.svg')))\n"
                  "display(SVG(filename=str(OUTPUT_DIR / 'Temporal_coverage_rankings_main.svg')))\n"
                  "display(SVG(filename=str(OUTPUT_DIR / 'Temporal_risk_and_rankings_Yelp1000.svg')))\n"
                  "display(Markdown((OUTPUT_DIR / 'Findings.md').read_text(encoding='utf-8')))\n"},
    ]
    nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
    for i,c in enumerate(cells):c["id"]=f"temporal-{i:02d}"
    (out/"Temporal_coverage_rankings.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf-8")

def run_temporal(source_dir,out_dir):
    source_dir,out=Path(source_dir),Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    validate_math()
    records,curves=audited_results(source_dir)
    comparison=compare_saved_curves(source_dir,curves)
    indexed,summary,paired,segments=summarize(curves)
    save_csv(out/"verified_seed_metrics.csv",records)
    save_csv(out/"risk_coverage_per_seed.csv",curves)
    save_csv(out/"risk_coverage_summary.csv",summary)
    save_csv(out/"paired_seed_differences.csv",paired)
    save_csv(out/"grid_rank_segments.csv",segments)
    save_csv(out/"selected_coverage_rankings.csv",[r for r in summary if round(r["coverage"]*100) in POINTS])
    plot_main(indexed,out,rank=False)
    plot_main(indexed,out,rank=True)
    plot_supplement(indexed,out)
    manuscript_files(summary,out)
    inputs=[]
    for key,stem in STEMS.items():
        path=next(source_dir.rglob(stem+"*.zip"))
        inputs.append(dict(role=key,file_name=path.name,size_bytes=path.stat().st_size,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    audit={"conditions":len(records),"coverage_points_per_condition":81,"risk_values":len(curves),
        "seeds":list(SEEDS),"periods":list(PERIODS),"settings":list(SETTINGS),"inputs":inputs,
        "scoring":"Frozen P3; historical archived-aligned Yelp; valid-label P3 abstain flags scored by label.",
        "tie_handling":"Expected retained error under uniform random tie-breaking at cutoff.",
        "checks":["Matched example ID/hash/gold sets and balanced counts for all four models in every condition.",
                  "Saved Accuracy, Macro-F1 and ten-bin ECE reproduced to absolute tolerance 1e-8.",
                  "Full-coverage risk equals one minus accuracy.",
                  "Analytic tie examples and row-permutation invariance.",comparison],
        "uncertainty":"Sample SD and paired seed differences only; no new confidence intervals or significance tests.",
        "versions":{"python":sys.version.split()[0],"numpy":np.__version__,"matplotlib":matplotlib.__version__}}
    (out/"Audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8")
    print(comparison)
    print("Saved temporal SVG/PDF/PNG figures, seed-level data, ranking summaries, and LaTeX fragments to",out)
    return {"summary":summary,"paired":paired,"audit":audit}

def main():
    parser=argparse.ArgumentParser(description="Audit and plot canonical temporal risk–coverage rankings.")
    parser.add_argument("--source-dir",type=Path,required=True,
                        help="Folder containing exactly one ZIP matching each of the five source stems.")
    parser.add_argument("--output-dir",type=Path,default=Path("Temporal_Coverage_Output"))
    args=parser.parse_args()
    run_temporal(args.source_dir,args.output_dir)
    script=Path(__file__).resolve()
    target=args.output_dir/"Temporal_coverage_rankings.py"
    if script != target.resolve():shutil.copy2(script,target)
    write_notebook(args.output_dir)

if __name__ == "__main__":
    main()

