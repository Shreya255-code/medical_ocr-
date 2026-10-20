"""Run the full demo:  python run_demo.py            (synthetic blurry prescription)"""
import argparse, json, pickle, time, pathlib
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import synth
from pipeline import *  # noqa

OUT = pathlib.Path(__file__).parent / "outputs"; OUT.mkdir(exist_ok=True)
GT = " ".join(synth.LINES)
BLUE, ORANGE, GRAY = "#2F6FDB", "#D9731F", "#6B7280"
STATUS = {"AUTO": "#2E9E5B", "SUGGEST_CONFIRM": "#E0A800", "HUMAN_REVIEW": "#D64545", "REDACTED": "#444444"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})


def make_doc(seed, severity, with_geometry=True):
    rng = np.random.RandomState(seed + 999)
    clean, _ = synth.render_clean(seed)
    img = synth.photometric_degrade(clean, severity, seed)
    angle = 0.0
    if with_geometry:
        angle = float(rng.choice([-1, 1]) * (1.5 + 5 * severity * rng.rand()))
        img = synth.geometric_degrade(img, angle, persp=0.012 * severity, seed=seed)
    return clean, img, angle


def section(t): print(f"\n{'=' * 8} {t} {'=' * (60 - len(t))}")


# calibration data
def collect(seeds, pre=True, cal=None):
    rows = []
    for sd in seeds:
        sev = 0.15 + 0.75 * ((sd * 37) % 100) / 100
        clean, img, ang = make_doc(sd, sev)
        words = mask_pii(run_ocr(preprocess(img)[0] if pre else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        ok = label_words(words, GT)
        for w, o in zip(words, ok):
            f, _ = features(w)
            rows.append({"seed": sd, "sev": sev, "ok": int(o), "raw": w["raw_conf"], "f": f, "pii": w["pii"]})
    return rows


def main(image_path=None):
    t0 = time.time()
    #  1. train calibrator on a labelled validation split 
    section("1. Calibration set (OCR on 18 degraded synthetic docs)")
    cal_rows = [r for r in collect(range(100, 118)) if not r["pii"]]
    X = [r["f"] for r in cal_rows]; y = [r["ok"] for r in cal_rows]
    cal = Calibrator().fit(X, y)
    pickle.dump(cal, open(OUT / "calibrator.pkl", "wb"))
    print(f"words={len(y)}  correct={np.mean(y):.1%}  calibrator fitted (logistic / Platt)")

    # 2. held-out test set 
    section("2. Held-out test set (12 unseen docs)")
    test_pre = [r for r in collect(range(200, 212), True) if not r["pii"]]
    test_raw = [r for r in collect(range(200, 212), False) if not r["pii"]]
    ypre = np.array([r["ok"] for r in test_pre]); yraw = np.array([r["ok"] for r in test_raw])
    p_cal = cal.predict([r["f"] for r in test_pre])
    p_raw = np.array([r["raw"] for r in test_pre])
    m = {}
    m["word_accuracy_no_preprocess"] = float(yraw.mean())
    m["word_accuracy_with_preprocess"] = float(ypre.mean())
    # confidence-vs-correctness errors  (Brier score == MSE of probability vs 0/1 outcome)
    m["calib_MSE_brier_raw"] = float(np.mean((p_raw - ypre) ** 2)); m["calib_MAE_raw"] = float(np.mean(np.abs(p_raw - ypre)))
    m["calib_MSE_brier_calibrated"] = float(np.mean((p_cal - ypre) ** 2)); m["calib_MAE_calibrated"] = float(np.mean(np.abs(p_cal - ypre)))
    m["ECE_raw"] = ece(p_raw, ypre); m["ECE_calibrated"] = ece(p_cal, ypre)
    # routing safety
    crit = np.array([bool(r["f"][2] > 0 or r["f"][1] >= 0.8) for r in test_pre])
    act = np.array([route(p, c) for p, c in zip(p_cal, crit)])
    auto = act == "AUTO"
    m["auto_fraction"] = float(auto.mean())
    m["auto_critical_error_rate"] = float(1 - ypre[auto].mean()) if auto.any() else 0.0
    m["human_review_fraction"] = float((act == "HUMAN_REVIEW").mean())
    m["error_words_sent_to_human_or_confirm"] = float(((act != "AUTO") & (ypre == 0)).sum() / max((ypre == 0).sum(), 1))
    print(json.dumps({k: round(v, 4) for k, v in m.items()}, indent=1))

    #  3. page-level CER / WER by severity 
    section("3. CER / WER vs degradation severity (raw OCR vs our preprocessing)")
    sev_levels = [0.15, 0.35, 0.55, 0.75]; cerr = {"raw": [], "pre": []}; werr = {"raw": [], "pre": []}
    for sv in sev_levels:
        c = {"raw": [], "pre": []}; w_ = {"raw": [], "pre": []}
        for sd in range(300, 304):
            _, img, _ = make_doc(sd, sv)
            for key, im in (("raw", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), ("pre", preprocess(img)[0])):
                hyp = [x["orig"] for x in run_ocr(im)]
                c[key].append(cer(GT.lower(), " ".join(hyp).lower())); w_[key].append(wer(GT.lower().split(), [h.lower() for h in hyp]))
        for k in c: cerr[k].append(np.mean(c[k])); werr[k].append(np.mean(w_[k]))
        print(f"severity {sv:.2f}:  CER raw={cerr['raw'][-1]:.3f} -> pre={cerr['pre'][-1]:.3f}   WER raw={werr['raw'][-1]:.3f} -> pre={werr['pre'][-1]:.3f}")
    m["CER_by_severity"] = {"severity": sev_levels, **{k: [float(x) for x in v] for k, v in cerr.items()}}
    m["WER_by_severity"] = {"severity": sev_levels, **{k: [float(x) for x in v] for k, v in werr.items()}}

    #  4. MSE / MAE of image restoration (geometry-preserving stages) 
    section("4. Image restoration error vs clean original (0-1 scale)")
    stage_names = ["0_degraded", "1_stamp_removed", "2_background_normalised", "3_denoised", "4_sharpened"]
    img_mse = {s: [] for s in stage_names}; img_mae = {s: [] for s in stage_names}
    for sv in (0.2, 0.5, 0.8):
        a = {s: [] for s in stage_names}; b = {s: [] for s in stage_names}
        for sd in range(400, 403):
            clean, _ = synth.render_clean(sd)
            deg = synth.photometric_degrade(clean, sv, sd)
            stg = {"0_degraded": cv2.cvtColor(deg, cv2.COLOR_BGR2GRAY), **restore_photometric(deg)}
            for s in stage_names: a[s].append(mse(stg[s], clean)); b[s].append(mae(stg[s], clean))
        for s in stage_names: img_mse[s].append(np.mean(a[s])); img_mae[s].append(np.mean(b[s]))
        print(f"severity {sv}: " + "  ".join(f"{s[:1]}:MSE={np.mean(a[s]):.4f}/MAE={np.mean(b[s]):.4f}" for s in stage_names))
    m["restoration_MSE"] = {s: [float(x) for x in v] for s, v in img_mse.items()}
    m["restoration_MAE"] = {s: [float(x) for x in v] for s, v in img_mae.items()}

    #  5. skew-angle MAE 
    section("5. Deskew accuracy")
    errs = []
    for sd in range(500, 512):
        sev = 0.2 + 0.6 * ((sd * 13) % 10) / 10
        _, img, true_ang = make_doc(sd, sev)
        est = estimate_skew(cv2.cvtColor(remove_stamp(img)[0], cv2.COLOR_BGR2GRAY))
        errs.append(abs(est + true_ang))   # correcting rotation = -true
    m["skew_MAE_degrees"] = float(np.mean(errs)); m["skew_MSE_deg2"] = float(np.mean(np.square(errs)))
    print(f"skew-angle MAE = {m['skew_MAE_degrees']:.3f} deg   MSE = {m['skew_MSE_deg2']:.3f} deg^2   (n=12)")

    #  6. DEMO DOCUMENT (the screenshots)
    section("6. Demo document")
    if image_path:
        img = cv2.imread(image_path); clean = None
        print(f"using your image: {image_path}  {img.shape[1]}x{img.shape[0]}")
    else:
        clean, img, true_ang = make_doc(7, 0.55)
        print(f"synthetic doc: severity 0.55, true skew {true_ang:+.2f} deg")
    cv2.imwrite(str(OUT / "demo_input.png"), img)
    if clean is not None: cv2.imwrite(str(OUT / "demo_clean_reference.png"), clean)
    res = process_document(img, cal)
    g = res["gate"]
    print(f"[quality gate] sharpness={g['sharpness']:.1f} contrast={g['contrast']:.1f} brightness={g['brightness']:.1f} -> {'PASS' if g['passed'] else 'FAIL: ' + '; '.join(g['reasons'])}")
    print(f"[deskew] estimated correction = {res['angle']:+.2f} deg")
    print(f"\n{'word':<18}{'raw':>6}{'calib':>7}  {'critical':<9}{'action':<17}suggestion")
    for w in res["words"]:
        print(f"{w['text']:<18}{w['raw_conf']:>6.2f}{w['calibrated_conf']:>7.2f}  {str(w['critical']):<9}{w['action']:<17}{w['suggestion'] or ''}")
    cnt = {k: sum(w['action'] == k for w in res['words']) for k in STATUS}
    print(f"\nrouting counts: {cnt}")
    print(f"PAGE-LEVEL: {'RFI -> ask hospital to resubmit a legible copy' if res['rfi'] else 'no RFI needed; low-confidence words are flagged to a human reviewer'}")
    print(f"text allowed downstream (AUTO only): {res['downstream_text']!r}")
    print(f"held as pending until a human confirms: {res['pending_confirmation_text']!r}")
    json.dump({k: v for k, v in m.items()}, open(OUT / "metrics.json", "w"), indent=1)

    
    # Fig 1: preprocessing stages
    st = {"0 original photo": cv2.cvtColor(img, cv2.COLOR_BGR2RGB), **{k: v for k, v in res["stages"].items()}}
    fig, axs = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (name, im) in zip(axs.ravel(), st.items()):
        ax.imshow(im, cmap=None if im.ndim == 3 else "gray"); ax.set_title(name.replace("_", " "), fontsize=10); ax.axis("off")
    for ax in axs.ravel()[len(st):]: ax.axis("off")
    fig.suptitle("Preprocessing pipeline: stage by stage", fontsize=14); fig.tight_layout()
    fig.savefig(OUT / "fig1_preprocessing_stages.png"); plt.close(fig)

    # Fig 1b: before / after
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axs[0].set_title("BEFORE: blurry, skewed, stamped, shadowed"); axs[1].imshow(res["image"], cmap="gray"); axs[1].set_title("AFTER: restored, deskewed, binarised")
    for a in axs: a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig1b_before_after.png"); plt.close(fig)

    # Fig 2: routing overlay
    vis = cv2.cvtColor(res["image"], cv2.COLOR_GRAY2RGB)
    for w in res["words"]:
        x, y_, ww, hh = w["box"]; col = tuple(int(STATUS[w["action"]][i:i + 2], 16) for i in (1, 3, 5))
        cv2.rectangle(vis, (x, y_), (x + ww, y_ + hh), col, 3)
        cv2.putText(vis, f"{w['calibrated_conf']:.2f}", (x, max(y_ - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    fig, ax = plt.subplots(figsize=(15, 10)); ax.imshow(vis); ax.axis("off"); ax.set_title("OCR output with calibrated confidence and routing decision")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in STATUS.items()], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(); fig.savefig(OUT / "fig2_routing_overlay.png"); plt.close(fig)

    # Fig 3: MSE / MAE restoration
    sev_lab = ["low (0.2)", "medium (0.5)", "high (0.8)"]; cols = [BLUE, ORANGE, GRAY]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, nm in ((axs[0], img_mse, "MSE"), (axs[1], img_mae, "MAE")):
        xs = np.arange(len(stage_names)); wd = 0.26
        for i, (lab, c) in enumerate(zip(sev_lab, cols)):
            ax.bar(xs + (i - 1) * wd, [data[s][i] for s in stage_names], wd - 0.02, color=c, label=f"degradation {lab}")
        ax.set_xticks(xs); ax.set_xticklabels(["degraded", "stamp\nremoved", "background\nnormalised", "denoised", "sharpened"], fontsize=9)
        ax.set_ylabel(f"{nm} vs clean original (0-1 scale)"); ax.set_title(f"Image restoration error ({nm}), lower is better"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig3_mse_mae_restoration.png"); plt.close(fig)

    # Fig 4: calibration
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    bins = np.linspace(0, 1, 11); mid = (bins[:-1] + bins[1:]) / 2
    for p, lab, c in ((p_raw, "raw Tesseract confidence", ORANGE), (p_cal, "calibrated confidence", BLUE)):
        acc = [ypre[(p > lo) & (p <= hi)].mean() if ((p > lo) & (p <= hi)).sum() >= 5 else np.nan for lo, hi in zip(bins[:-1], bins[1:])]
        axs[0].plot(mid, acc, "o-", color=c, label=lab)
    axs[0].plot([0, 1], [0, 1], "--", color=GRAY, label="perfect calibration")
    axs[0].set_xlabel("stated confidence"); axs[0].set_ylabel("actual accuracy"); axs[0].set_title(f"Reliability diagram\nECE raw={m['ECE_raw']:.3f} -> calibrated={m['ECE_calibrated']:.3f}"); axs[0].legend(frameon=False)
    axs[1].hist(p_cal[ypre == 1], bins=20, range=(0, 1), alpha=0.75, color=BLUE, label="correct words"); axs[1].hist(p_cal[ypre == 0], bins=20, range=(0, 1), alpha=0.75, color=ORANGE, label="wrong words")
    axs[1].axvline(0.6, color="k", ls=":"); axs[1].axvline(0.9, color="k", ls=":"); axs[1].set_xlabel("calibrated confidence"); axs[1].set_ylabel("words"); axs[1].set_title("Wrong words concentrate at low confidence"); axs[1].legend(frameon=False)
    names = ["MSE (Brier)", "MAE"]; raw_v = [m["calib_MSE_brier_raw"], m["calib_MAE_raw"]]; cal_v = [m["calib_MSE_brier_calibrated"], m["calib_MAE_calibrated"]]
    xs = np.arange(2); axs[2].bar(xs - 0.2, raw_v, 0.38, color=ORANGE, label="raw"); axs[2].bar(xs + 0.2, cal_v, 0.38, color=BLUE, label="calibrated")
    axs[2].set_xticks(xs); axs[2].set_xticklabels(names); axs[2].set_title("Confidence vs outcome error (lower is better)"); axs[2].legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig4_calibration.png"); plt.close(fig)

    # Fig 5: risk-coverage + routing split
    order = np.argsort(-p_cal); cum_acc = np.cumsum(ypre[order]) / np.arange(1, len(order) + 1); cov = np.arange(1, len(order) + 1) / len(order)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].plot(cov, 1 - cum_acc, color=BLUE); axs[0].set_xlabel("coverage (fraction of words auto-accepted)"); axs[0].set_ylabel("error rate among accepted words")
    axs[0].set_title("Risk-coverage curve: accept fewer words, make fewer errors"); axs[0].axvline(m["auto_fraction"], color=ORANGE, ls="--", label=f"our AUTO policy: {m['auto_fraction']:.0%} auto, {m['auto_critical_error_rate']:.1%} error"); axs[0].legend(frameon=False)
    labs = ["AUTO", "SUGGEST_CONFIRM", "HUMAN_REVIEW"]; right = [int(((act == l) & (ypre == 1)).sum()) for l in labs]; wrong = [int(((act == l) & (ypre == 0)).sum()) for l in labs]
    axs[1].bar(labs, right, color=BLUE, label="correct words"); axs[1].bar(labs, wrong, bottom=right, color=ORANGE, label="wrong words")
    axs[1].set_title("Where each routing bucket's words end up"); axs[1].set_ylabel("words (test set)"); axs[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig5_risk_coverage_routing.png"); plt.close(fig)

    # Fig 6: CER / WER vs severity
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, d, nm in ((axs[0], cerr, "CER"), (axs[1], werr, "WER")):
        ax.plot(sev_levels, d["raw"], "o-", color=ORANGE, label="raw OCR (no preprocessing)"); ax.plot(sev_levels, d["pre"], "o-", color=BLUE, label="with our preprocessing")
        ax.set_xlabel("degradation severity"); ax.set_ylabel(nm); ax.set_title(f"{nm} vs degradation (lower is better)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig6_cer_wer_vs_severity.png"); plt.close(fig)
    section(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--image"); main(ap.parse_args().image)
