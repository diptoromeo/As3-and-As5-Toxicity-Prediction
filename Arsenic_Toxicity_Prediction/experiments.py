# -*- coding: utf-8 -*-
"""
experiments.py
===============
보고서 "1. 추가 실험 권고사항" 의 4개 항목 구현:
  1) Dose-Response Curve Fitting (Hill / Logistic 회귀 vs GCN)
  2) Biomarker Attribution (gradient-based feature attribution)
  3) Threshold Sensitivity Analysis (±10%, ±20%)
  4) Cross-Species Generalizability (zero-shot / transfer-learning 평가 프레임워크)

(Bootstrap CI는 train_eval.py, Missing-Data 비율 실험은 main.py에서 missing_ratio 스윕으로 처리)
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression

from core import (
    BIOMARKERS, THRESHOLDS, build_graph, GCNBinary, perturb_thresholds,
    parse_day_blocks, label_sample,
)
from train_eval import run_cv_for_day
from torch_geometric.loader import DataLoader


# ============================================================
# 1) Dose-Response Curve Fitting
# ============================================================
def hill_eq(x, top, ec50, hill_slope, bottom=0.0):
    x = np.clip(x, 1e-6, None)
    return bottom + (top - bottom) / (1.0 + (ec50 / x) ** hill_slope)


def fit_dose_response(samples, day_name, species, out_dir):
    """
    Hill equation 및 logistic regression(농도 단변량)을 binary label에 적합시키고,
    GCN out-of-fold 예측 확률(run_cv_for_day의 y_proba)과 비교 plot.
    EC50/LC50(=Hill 적합의 ec50 파라미터, logistic의 50% 지점)을 함께 보고.
    """
    conc = np.array([s["conc"] for s in samples], dtype=float)
    y = np.array([s["y"] for s in samples], dtype=float)

    report_lines = [f"[{species} | {day_name}] Dose-Response Curve Fitting"]

    # --- Hill equation fit ---
    ec50_hill = np.nan
    hill_ok = True
    try:
        p0 = [1.0, np.median(conc[conc > 0]) if (conc > 0).any() else 1.0, 1.0, 0.0]
        bounds = ([0, 1e-3, 0.1, 0], [1.5, conc.max() * 5 + 1e-3, 20, 1])
        popt, _ = curve_fit(hill_eq, conc, y, p0=p0, bounds=bounds, maxfev=20000)
        top, ec50_hill, slope, bottom = popt
        report_lines.append(f"  Hill fit: top={top:.3f}, EC50={ec50_hill:.4f} mg/L, "
                             f"slope={slope:.3f}, bottom={bottom:.3f}")
    except Exception as e:
        hill_ok = False
        report_lines.append(f"  Hill fit failed: {e}")

    # --- Logistic regression (1D) fit -> EC50 = -b0/b1 ---
    try:
        lr = LogisticRegression()
        lr.fit(conc.reshape(-1, 1), y)
        b0, b1 = lr.intercept_[0], lr.coef_[0][0]
        ec50_logit = -b0 / b1 if b1 != 0 else np.nan
        report_lines.append(f"  Logistic fit: intercept={b0:.4f}, slope={b1:.4f}, "
                             f"EC50(50% prob)={ec50_logit:.4f} mg/L")
    except Exception as e:
        lr = None
        ec50_logit = np.nan
        report_lines.append(f"  Logistic fit failed: {e}")

    # --- GCN out-of-fold probability (5-fold, 적은 epoch으로 빠르게) ---
    cv_res = run_cv_for_day(samples, f"{day_name} (dose-response demo)", epochs=120, verbose=False)
    gcn_true = cv_res["y_true"]
    gcn_proba = cv_res["y_proba"]
    gcn_conc = conc  # run_cv_for_day는 StratifiedKFold라 순서가 섞이므로 별도 매핑 필요
    # run_cv_for_day 내부에서 fold 순서대로 섞이기 때문에, 농도-확률 매핑을 다시 만들기 위해
    # 동일한 함수를 직접 호출하는 대신 간단히 전체 데이터로 한번 더 모델을 학습/추론한다.
    gcn_conc_sorted, gcn_proba_sorted = _gcn_full_fit_predict(samples)

    # --- plot ---
    xs = np.linspace(max(conc.min(), 1e-3), 8.0, 200)
    plt.figure(figsize=(9, 6))
    plt.scatter(conc, y, color="black", alpha=0.5, label="Observed label (0/1)", zorder=3)
    if hill_ok:
        plt.plot(xs, hill_eq(xs, *popt), color="red", lw=2, label=f"Hill fit (EC50={ec50_hill:.3f})")
    if lr is not None:
        probs = 1 / (1 + np.exp(-(b0 + b1 * xs)))
        plt.plot(xs, probs, color="blue", lw=2, ls="--", label=f"Logistic fit (EC50={ec50_logit:.3f})")
    order = np.argsort(gcn_conc_sorted)
    plt.plot(np.array(gcn_conc_sorted)[order], np.array(gcn_proba_sorted)[order],
              color="green", lw=2, ls="-.", label="GCN predicted P(abnormal)")
    plt.xlabel("Concentration (mg/L)", fontsize=12, fontweight="bold")
    plt.ylabel("P(abnormal)", fontsize=12, fontweight="bold")
    plt.xlim(0, 8)
    plt.xlim(-0.15, 8.3)
    plt.ylim(-0.08, 1.12)
    plt.title(f"Dose-Response Comparison: Hill / Logistic vs GCN\n({species}, {day_name})",
              fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"dose_response_{species}_{day_name}.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    report_lines.append(f"  Plot saved: {fname}")

    summary = {
        "species": species, "day": day_name,
        "EC50_hill": ec50_hill, "EC50_logistic": ec50_logit,
    }
    return "\n".join(report_lines), summary


def _gcn_full_fit_predict(samples, epochs=150, seed=42):
    """전체 데이터로 GCN을 학습 후 같은 데이터에 대해 예측확률 반환 (dose-response 곡선 시각화용)."""
    from core import graph_mixup
    import torch.nn as nn

    bio = np.stack([s["bio"] for s in samples], axis=0)
    bio_medians = np.nanmedian(bio, axis=0).astype(np.float32)
    graphs = [build_graph(s, bio_medians) for s in samples]
    loader = DataLoader(graphs, batch_size=16, shuffle=True)
    full_loader = DataLoader(graphs, batch_size=16, shuffle=False)

    torch.manual_seed(seed)
    model = GCNBinary(in_dim=graphs[0].x.shape[1], hidden=32, dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    y_all = np.array([int(g.y.item()) for g in graphs])
    pos, neg = (y_all == 1).sum(), (y_all == 0).sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = graph_mixup(batch, alpha=0.4)
            logits = model(batch)
            loss = bce(logits, batch.y_mix.float())
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    probs, concs = [], []
    with torch.no_grad():
        for batch in full_loader:
            logits = model(batch)
            prob = torch.sigmoid(logits).cpu().numpy().tolist()
            probs.extend(prob)
    concs = [s["conc"] for s in samples]
    return concs, probs


# ============================================================
# 2) Biomarker Attribution (gradient-based)
# ============================================================
def biomarker_attribution(samples, day_name, species, out_dir, epochs=150, seed=42):
    """
    입력 node-feature gradient의 절대값 평균으로 각 biomarker의 중요도를 추정.
    (saliency / GradInput 방식: d(logit)/d(x_bio) )
    """
    bio = np.stack([s["bio"] for s in samples], axis=0)
    bio_medians = np.nanmedian(bio, axis=0).astype(np.float32)
    graphs = [build_graph(s, bio_medians) for s in samples]

    torch.manual_seed(seed)
    loader = DataLoader(graphs, batch_size=16, shuffle=True)
    full_loader = DataLoader(graphs, batch_size=1, shuffle=False)

    model = GCNBinary(in_dim=graphs[0].x.shape[1], hidden=32, dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    y_all = np.array([int(g.y.item()) for g in graphs])
    pos, neg = (y_all == 1).sum(), (y_all == 0).sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    from core import graph_mixup

    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = graph_mixup(batch, alpha=0.4)
            logits = model(batch)
            loss = bce(logits, batch.y_mix.float())
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    grad_accum = np.zeros(len(BIOMARKERS), dtype=np.float64)
    n = 0
    for data in full_loader:
        data.x.requires_grad_(True)
        logit = model(data)
        model.zero_grad()
        logit.sum().backward()
        # node index 1 = biomarker node, feature 0..10 = biomarkers (마지막은 bias)
        grad_bio = data.x.grad[1, :len(BIOMARKERS)].detach().cpu().numpy()
        grad_accum += np.abs(grad_bio)
        n += 1
    importance = grad_accum / max(n, 1)
    importance_norm = importance / (importance.sum() + 1e-12)

    order = np.argsort(-importance_norm)
    df = pd.DataFrame({
        "biomarker": [BIOMARKERS[i] for i in order],
        "mean_abs_gradient": [importance[i] for i in order],
        "relative_importance": [importance_norm[i] for i in order],
    })
    csv_path = os.path.join(out_dir, f"biomarker_attribution_{species}_{day_name}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.barh(df["biomarker"][::-1], df["relative_importance"][::-1], color="teal")
    plt.xlabel("Relative importance (|gradient|, normalized)", fontsize=12, fontweight="bold")
    plt.title(f"Biomarker Attribution ({species}, {day_name})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"biomarker_attribution_{species}_{day_name}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return df, csv_path, fig_path


# ============================================================
# 3) Threshold Sensitivity Analysis (±10%, ±20%)
# ============================================================
def threshold_sensitivity(species, day, raw_path, day_name, out_dir, epochs=120):
    """
    biomarker-specific threshold tau_{m,day}를 -20%,-10%,0%,+10%,+20% 변화시키며
    1) 라벨 분포가 어떻게 바뀌는지
    2) 5-fold CV 성능(ACC/Macro-F1/Balanced ACC)이 어떻게 바뀌는지
    를 분석한다.
    """
    pct_list = [-20, -10, 0, 10, 20]
    rows = []
    for pct in pct_list:
        th = THRESHOLDS[species][day] if pct == 0 else perturb_thresholds(species, pct)[day]
        full_th = {10: THRESHOLDS[species][10], 20: THRESHOLDS[species][20]}
        full_th[day] = th

        def relabel(conc):
            flags = [int(conc >= full_th[day][bm]) for bm in BIOMARKERS]
            return int(max(flags))

        samples_by_day = parse_day_blocks(raw_path, species)
        samples = samples_by_day[day]
        for s in samples:
            s["y"] = relabel(s["conc"])

        n_pos = sum(s["y"] for s in samples)
        n_neg = len(samples) - n_pos

        if n_pos == 0 or n_neg == 0:
            rows.append({"pct_change": pct, "n_pos": n_pos, "n_neg": n_neg,
                         "acc_mean": np.nan, "macro_f1_mean": np.nan, "balanced_acc_mean": np.nan,
                         "note": "단일 클래스만 존재 -> CV 생략"})
            continue

        res = run_cv_for_day(samples, f"{day_name} (thr {pct:+d}%)", epochs=epochs, verbose=False)
        rows.append({
            "pct_change": pct, "n_pos": n_pos, "n_neg": n_neg,
            "acc_mean": res["cv_summary"]["acc_mean"],
            "macro_f1_mean": res["cv_summary"]["mf1_mean"],
            "balanced_acc_mean": res["cv_summary"]["bacc_mean"],
            "note": "",
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"threshold_sensitivity_{species}_{day_name}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9, 6))
    plt.plot(df["pct_change"], df["acc_mean"], marker="o", label="Accuracy")
    plt.plot(df["pct_change"], df["macro_f1_mean"], marker="s", label="Macro-F1")
    plt.plot(df["pct_change"], df["balanced_acc_mean"], marker="^", label="Balanced ACC")
    plt.xlabel("Threshold change (%)", fontsize=12, fontweight="bold")
    plt.ylabel("CV mean score", fontsize=12, fontweight="bold")
    plt.title(f"Threshold Sensitivity Analysis ({species}, {day_name})", fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"threshold_sensitivity_{species}_{day_name}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return df, csv_path, fig_path


# ============================================================
# 4) Cross-Species Generalizability (zero-shot / transfer-learning 프레임워크)
# ============================================================
def cross_species_generalization(train_samples, train_name, eval_external_path,
                                  external_species_label, day, out_dir,
                                  epochs=150, finetune_epochs=30, seed=42):
    """
    올리브 플라운더 데이터(train_samples)로 학습한 GCN을
      (a) zero-shot: 추가 학습 없이 외부 종 데이터에 그대로 평가
      (b) transfer-learning: 외부 종 데이터 일부로 fine-tuning 후 재평가
    하는 프레임워크.

    외부 종(예: 제브라피시 Danio rerio, 송어류) 데이터는 본 패키지에 포함되어 있지 않으므로,
    eval_external_path가 실제로 존재하고 동일한 엑셀 레이아웃
    (10일/20일 블록, group=농도, 11개 biomarker 행)을 따를 때만 자동 평가를 수행한다.
    파일이 없으면 '실행 준비 완료' 상태의 보고서만 생성한다.
    """
    from core import graph_mixup
    import torch.nn as nn

    report = [f"[Cross-Species Generalizability] base={train_name} -> target={external_species_label}"]

    bio = np.stack([s["bio"] for s in train_samples], axis=0)
    bio_medians = np.nanmedian(bio, axis=0).astype(np.float32)
    graphs = [build_graph(s, bio_medians) for s in train_samples]
    loader = DataLoader(graphs, batch_size=16, shuffle=True)

    torch.manual_seed(seed)
    model = GCNBinary(in_dim=graphs[0].x.shape[1], hidden=32, dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    y_all = np.array([int(g.y.item()) for g in graphs])
    pos, neg = (y_all == 1).sum(), (y_all == 0).sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = graph_mixup(batch, alpha=0.4)
            logits = model(batch)
            loss = bce(logits, batch.y_mix.float())
            opt.zero_grad(); loss.backward(); opt.step()

    model_path = os.path.join(out_dir, f"base_model_{train_name}.pth")
    torch.save(model.state_dict(), model_path)
    report.append(f"  Base model trained on {train_name} (N={len(train_samples)}) saved -> {model_path}")

    if not (eval_external_path and os.path.exists(eval_external_path)):
        report.append(f"  [대기] 외부 종 데이터 경로가 없습니다: {eval_external_path}")
        report.append("  -> 동일 엑셀 포맷(10일/20일 블록, group=농도(mg/L), 11개 biomarker 행)으로")
        report.append("     데이터를 두면 이 함수가 자동으로 zero-shot/transfer-learning 평가를 수행합니다.")
        return "\n".join(report), None

    try:
        ext_by_day = parse_day_blocks(eval_external_path, train_name.split("_")[0])
        ext_samples = ext_by_day[day]
    except Exception as e:
        report.append(f"  외부 데이터 파싱 실패: {e}")
        return "\n".join(report), None

    ext_graphs = [build_graph(s, bio_medians) for s in ext_samples]
    ext_loader = DataLoader(ext_graphs, batch_size=16, shuffle=False)

    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

    def evaluate(m, loader_):
        m.eval()
        yt, yp = [], []
        with torch.no_grad():
            for b in loader_:
                logit = m(b)
                prob = torch.sigmoid(logit)
                yp.extend((prob >= 0.5).long().cpu().numpy().tolist())
                yt.extend(b.y.long().cpu().numpy().tolist())
        return (accuracy_score(yt, yp), f1_score(yt, yp, average="macro", zero_division=0),
                balanced_accuracy_score(yt, yp))

    acc0, f10, bacc0 = evaluate(model, ext_loader)
    report.append(f"  Zero-shot on {external_species_label}: ACC={acc0:.3f}, Macro-F1={f10:.3f}, BalACC={bacc0:.3f}")

    # transfer-learning: 외부 데이터 절반으로 fine-tune, 나머지 절반으로 평가
    n_ext = len(ext_graphs)
    if n_ext >= 6:
        idx = np.random.default_rng(seed).permutation(n_ext)
        half = n_ext // 2
        ft_graphs = [ext_graphs[i] for i in idx[:half]]
        eval_graphs = [ext_graphs[i] for i in idx[half:]]
        ft_loader = DataLoader(ft_graphs, batch_size=8, shuffle=True)
        eval_loader = DataLoader(eval_graphs, batch_size=8, shuffle=False)

        ft_model = GCNBinary(in_dim=graphs[0].x.shape[1], hidden=32, dropout=0.2)
        ft_model.load_state_dict(model.state_dict())
        ft_opt = torch.optim.Adam(ft_model.parameters(), lr=5e-4, weight_decay=1e-4)
        for _ in range(finetune_epochs):
            ft_model.train()
            for b in ft_loader:
                logit = ft_model(b)
                loss = bce(logit, b.y.float())
                ft_opt.zero_grad(); loss.backward(); ft_opt.step()

        acc1, f11, bacc1 = evaluate(ft_model, eval_loader)
        report.append(f"  Transfer-learning (fine-tune {finetune_epochs} epochs, "
                       f"N_ft={half}, N_eval={n_ext-half}): "
                       f"ACC={acc1:.3f}, Macro-F1={f11:.3f}, BalACC={bacc1:.3f}")
    else:
        report.append("  외부 데이터 수가 너무 적어(<6) transfer-learning 분할을 생략합니다.")

    return "\n".join(report), {"zero_shot": (acc0, f10, bacc0)}
