"""Phase 3 comparison module (docs/phase3_protocol.md §9-10). Validation data only.

* E1/E2: mean ± std over seeds of classification, probabilistic and (descriptive)
  calibration metrics; McNemar (indicative) and paired NLL comparisons.
* Selection of the default model with the FROZEN rule of §10 (needs the three E1
  models and the E4 benchmark; otherwise reported as pending).
* E3: learning curves (F1, accuracy, NLL vs real fraction), area under the F1 curve,
  paired comparisons at equal seed / subset (descriptive).
* Figures: learning curves, reliability diagrams, cost/performance plot.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --out <dir> --force
Outputs: reports/comparison/phase3/comparison.json, comparison.md, *.png
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ivai.config import REPORTS_DIR, load_data_config  # noqa: E402
from ivai.data.subsets import FRACTIONS, fraction_tag  # noqa: E402
from ivai.evaluation.calibration import ece  # noqa: E402
from ivai.evaluation.comparison import check_aligned, learning_curve, load_experiment, summarize  # noqa: E402
from ivai.evaluation.plots import GRID, MUTED, SERIES  # noqa: E402
from ivai.evaluation.stats import clopper_pearson, mcnemar_exact, paired_nll_comparison, select_default_model  # noqa: E402

E1 = ["custom_cnn", "resnet50_ft", "mobilenetv2_ft"]
E2 = ["mobilenetv2_scratch", "resnet50_probe", "mobilenetv2_probe"]
E2_PAIRS = [("mobilenetv2_ft", "mobilenetv2_scratch"), ("resnet50_ft", "resnet50_probe"),
            ("mobilenetv2_ft", "mobilenetv2_probe")]
LABEL = {"custom_cnn": "CNN personnalisé", "resnet50_ft": "ResNet50 (FT)", "mobilenetv2_ft": "MobileNetV2 (FT)",
         "mobilenetv2_scratch": "MobileNetV2 (depuis zéro)", "resnet50_probe": "ResNet50 (sonde)",
         "mobilenetv2_probe": "MobileNetV2 (sonde)"}


def fmt(m: dict, p: int = 4) -> str:
    return f"{m['mean']:.{p}f}" + ("" if m["std"] is None else f" ± {m['std']:.{p}f}")


def pairwise(exps: dict, pairs: list[tuple[str, str]]) -> dict:
    out = {}
    for a, b in pairs:
        if a not in exps or b not in exps:
            continue
        A, B = exps[a], exps[b]
        check_aligned(A, B)
        mc = {s: mcnemar_exact(A.preds[s]["true"] == A.preds[s]["pred"], B.preds[s]["true"] == B.preds[s]["pred"])
              for s in A.seeds}
        out[f"{a} vs {b}"] = {"mcnemar_per_seed": mc,
                              "paired_nll": paired_nll_comparison(A.per_image_nll(), B.per_image_nll())}
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "comparison" / "phase3")
    ap.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    ap.add_argument("--benchmark", type=Path, default=REPORTS_DIR / "benchmarks" / "phase3" / "benchmark.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if (args.out / "comparison.json").exists() and not args.force:
        raise SystemExit(f"{args.out / 'comparison.json'} exists; use --force to regenerate this derived report.")
    args.out.mkdir(parents=True, exist_ok=True)
    classes = load_data_config().classes

    names = E1 + E2 + [f"{b}_{fraction_tag(f)}" for b in E1 for f in FRACTIONS]
    exps = {n: e for n in names if (e := load_experiment(n, classes, args.reports_dir)) is not None}
    if not exps:
        raise SystemExit("no experiment found")
    summaries = {n: summarize(e) for n, e in exps.items()}
    report: dict = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split": "validation",
                    "warning": "Validation results are not an independent estimate of generalisation (protocol §12).",
                    "experiments_found": sorted(exps), "summaries": summaries}

    # ---- comparisons -------------------------------------------------------
    e1_pairs = [(a, b) for i, a in enumerate(E1) for b in E1[i + 1:]]
    report["pairwise_e1"] = pairwise(exps, e1_pairs)
    report["pairwise_e2"] = pairwise(exps, E2_PAIRS)

    # ---- selection (frozen rule) -------------------------------------------
    missing = [n for n in E1 if n not in exps]
    bench = json.loads(args.benchmark.read_text(encoding="utf-8")) if args.benchmark.exists() else None
    if missing or bench is None or bench.get("quick_smoke_test"):
        report["selection"] = {"status": "pending",
                               "reason": (f"missing E1 experiments {missing}" if missing else "") +
                                         ("" if bench and not bench.get("quick_smoke_test") else " missing E4 benchmark")}
    else:
        check_aligned(*(exps[n] for n in E1))
        cand = {n: {"mean_errors": summaries[n]["metrics"]["n_errors"]["mean"],
                    "mean_recall_per_class": summaries[n]["mean_recall_per_class"]} for n in E1}
        sel = select_default_model(cand, {n: exps[n].per_image_nll() for n in E1},
                                   {n: bench["models"][n]["latency"]["cpu_fp32_b1"]["median_of_medians_ms"] for n in E1},
                                   {n: bench["models"][n]["state_dict_fp32_mb"] for n in E1})
        report["selection"] = {"status": "done", **sel}

    # ---- E3 learning curves ------------------------------------------------
    curves, e3_pairs = {}, {}
    for b in E1:
        pts = {}
        for f in FRACTIONS:
            n = f"{b}_{fraction_tag(f)}"
            if n in summaries:
                pts[summaries[n]["train_subset"]["fraction_real"]] = summaries[n]
        if b in summaries:
            pts[1.0] = summaries[b]
        if pts:
            curves[b] = learning_curve(pts)
    for f in FRACTIONS:
        tag = fraction_tag(f)
        pairs = [(f"{a}_{tag}", f"{c}_{tag}") for i, a in enumerate(E1) for c in E1[i + 1:]]
        res = pairwise(exps, pairs)
        if res:
            e3_pairs[tag] = res
    report["e3_learning_curves"] = curves
    report["e3_pairwise_descriptive"] = e3_pairs
    report["resolution_n270"] = {k: clopper_pearson(k, 270) for k in range(6)}
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    # ---- markdown ----------------------------------------------------------
    md = ["# Phase 3 : comparaison des modèles (validation)", "",
          "> Les résultats de validation ne sont pas une estimation indépendante de la généralisation "
          "(protocole §12). ECE descriptive uniquement.", "",
          "| Expérience | Accuracy | F1 macro | Erreurs /270 | NLL | Brier | ECE (descr.) | Paramètres |",
          "|---|---|---|---|---|---|---|---|"]
    for n in [x for x in names if x in summaries]:
        m = summaries[n]["metrics"]
        md.append(f"| {n} | {fmt(m['accuracy'])} | {fmt(m['f1_macro'])} | {fmt(m['n_errors'], 1)} | {fmt(m['nll'])} | "
                  f"{fmt(m['brier'])} | {fmt(m['ece'])} | {summaries[n]['parameters']['total']:,} |")
    for title, block in [("E1 : comparaisons appariées", report["pairwise_e1"]), ("E2 : ablations", report["pairwise_e2"])]:
        if block:
            md += ["", f"## {title}", "", "| Paire | d_s (NLL) | IC 95 % | Verdict NLL | McNemar p par graine |", "|---|---|---|---|---|"]
            for k, v in block.items():
                pn = v["paired_nll"]
                md.append(f"| {k} | {', '.join(f'{d:+.4f}' for d in pn['d_s'])} | [{pn['ci95'][0]:+.4f}, {pn['ci95'][1]:+.4f}] | "
                          f"{pn['verdict']} | {', '.join(f'{s}: {r['p_value']:.3f}' for s, r in v['mcnemar_per_seed'].items())} |")
    md += ["", "## Sélection (règle figée, §10)", "", f"```\n{json.dumps(report['selection'], indent=2, default=float)[:4000]}\n```"]
    if curves:
        md += ["", "## E3 : courbes d'apprentissage (F1 macro moyen)", "", "| Modèle | Fractions réelles | F1 | AUC F1 |", "|---|---|---|---|"]
        for b, c in curves.items():
            md.append(f"| {b} | {', '.join(f'{x:.4f}' for x in c['fractions'])} | {', '.join(f'{y:.4f}' for y in c['mean']['f1_macro'])} | "
                      f"{'' if c['auc_f1'] is None else f'{c['auc_f1']:.4f}'} |")
    (args.out / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # ---- figures -----------------------------------------------------------
    def style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for k, (b, c) in enumerate(curves.items()):
            for ax, key in zip(axes, ("f1_macro", "nll")):
                std = [0 if s is None else s for s in c["std"][key]]
                ax.errorbar(c["fractions"], c["mean"][key], yerr=std, color=SERIES[k], marker="o", capsize=3,
                            linewidth=2, label=LABEL.get(b, b))
        axes[0].set_ylabel("F1 macro (validation)")
        axes[1].set_ylabel("NLL (validation)")
        axes[1].set_yscale("log")
        for ax in axes:
            ax.set_xlabel("Fraction réelle du train")
            style(ax)
        axes[0].legend(frameon=False)
        fig.suptitle("E3 : efficacité en données (moyenne ± écart-type, 3 graines)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(args.out / "e3_learning_curves.png", dpi=150, facecolor="white")
        plt.close(fig)

    shown = [n for n in E1 + E2 if n in exps]
    if shown:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="calibration parfaite")
        for k, n in enumerate(shown):
            conf = np.concatenate([exps[n].probs_and_labels(s)[0].max(1) for s in exps[n].seeds])
            corr = np.concatenate([(lambda p, y: (p.argmax(1) == y))(*exps[n].probs_and_labels(s)) for s in exps[n].seeds])
            bins = [b for b in ece(conf, corr)["bins"] if b["count"]]
            ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], marker="o",
                    color=SERIES[k % len(SERIES)], linewidth=1.5, label=LABEL.get(n, n))
            for b in bins:  # bin counts: most points rest on 1-3 images
                ax.annotate(f"n={b['count']}", (b["confidence"], b["accuracy"]), xytext=(4, -10 - 9 * k),
                            textcoords="offset points", fontsize=7, color=SERIES[k % len(SERIES)])
        ax.set_xlabel("Confiance moyenne de l'intervalle")
        ax.set_ylabel("Accuracy de l'intervalle")
        ax.set_title("Diagramme de fiabilité (3 graines regroupées, 15 intervalles)\n"
                     "descriptif : la plupart des intervalles sont vides", fontsize=10, fontweight="bold")
        style(ax)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out / "reliability_diagram.png", dpi=150, facecolor="white")
        plt.close(fig)

    if bench is not None and not bench.get("quick_smoke_test"):
        pts = [n for n in shown if n in bench["models"]]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k, n in enumerate(pts):
            x = bench["models"][n]["latency"]["cpu_fp32_b1"]["median_of_medians_ms"]
            y = summaries[n]["metrics"]["nll"]["mean"]
            ax.scatter(x, y, s=40 + 4 * bench["models"][n]["state_dict_fp32_mb"], color=SERIES[k % len(SERIES)],
                       edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(LABEL.get(n, n), (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Latence CPU médiane, batch 1, 4 threads (ms)")
        ax.set_ylabel("NLL de validation (moyenne 3 graines)")
        ax.set_title("Coût contre qualité probabiliste (taille du point ∝ taille du modèle)", fontweight="bold", fontsize=10)
        style(ax)
        fig.tight_layout()
        fig.savefig(args.out / "cost_vs_nll.png", dpi=150, facecolor="white")
        plt.close(fig)
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
