# -*- coding: utf-8 -*-
"""
train_eval.py
==============
Stratified 5-fold CV 학습/평가 + epoch-wise metric 기록 + Bootstrap 95% CI.
(추가 실험 권고사항 #5 "Bootstrap 신뢰구간 보고" 반영)
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    balanced_accuracy_score, roc_curve, auc,
)

from core import build_graph, graph_mixup, GCNBinary
from torch_geometric.loader import DataLoader


def bootstrap_ci(y_true, y_proba, metric_fn, n_boot=2000, alpha=0.05, seed=42, **kwargs):
    """y_true/y_proba(pooled, out-of-fold)에서 bootstrap resample로 95% CI 계산."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_proba[idx]
        if len(np.unique(yt)) < 2 and metric_fn.__name__ == "roc_auc_safe":
            continue
        try:
            stats.append(metric_fn(yt, yp, **kwargs))
        except Exception:
            continue
    stats = np.array(stats)
    lo = np.percentile(stats, 100 * alpha / 2)
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(np.mean(stats)), float(lo), float(hi)


def _acc_from_proba(y_true, y_proba, thr=0.5):
    pred = (y_proba >= thr).astype(int)
    return accuracy_score(y_true, pred)


def _f1_from_proba(y_true, y_proba, thr=0.5):
    pred = (y_proba >= thr).astype(int)
    return f1_score(y_true, pred, average="macro", zero_division=0)


def _bacc_from_proba(y_true, y_proba, thr=0.5):
    pred = (y_proba >= thr).astype(int)
    return balanced_accuracy_score(y_true, pred)


def roc_auc_safe(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        raise ValueError("single class")
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return auc(fpr, tpr)


def run_cv_for_day(samples, day_name, epochs=150, batch_size=16, lr=1e-3,
                    mixup_alpha=0.4, random_seed=42, missing_ratio=0.0,
                    n_splits=5, verbose=True, n_boot=2000):
    """
    missing_ratio: 0.0~1.0 -> 매 그래프 생성시 해당 비율만큼 biomarker 값을 강제 결측
                   처리 후 median 대체 (실험 6: Missing-Data 비율 실험)
    """
    rng = np.random.default_rng(random_seed)
    labels = [int(s["y"]) for s in samples]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    fold_results = []
    all_y_true, all_y_proba = [], []
    all_fold_train_losses, all_fold_train_accs = [], []
    all_fold_val_losses, all_fold_val_accs = [], []

    output_lines = []
    output_lines.append(f"\n{'='*80}")
    output_lines.append(f"{day_name} | N={len(samples)} | missing_ratio={missing_ratio:.2f} | "
                         f"Labeling: biomarker-threshold (max-rule)")
    output_lines.append(f"{'='*80}")

    n_bio = len(samples[0]["bio"])

    for fold, (tr_idx, te_idx) in enumerate(skf.split(samples, labels), 1):
        train_samples = [samples[i] for i in tr_idx]
        test_samples = [samples[i] for i in te_idx]

        train_bio = np.stack([s["bio"] for s in train_samples], axis=0)
        bio_medians = np.nanmedian(train_bio, axis=0).astype(np.float32)

        def make_mask(n_nodes):
            if missing_ratio <= 0:
                return None
            return rng.random(n_nodes) < missing_ratio

        train_graphs = [build_graph(s, bio_medians, make_mask(n_bio)) for s in train_samples]
        test_graphs = [build_graph(s, bio_medians, make_mask(n_bio)) for s in test_samples]

        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

        model = GCNBinary(in_dim=train_graphs[0].x.shape[1], hidden=32, dropout=0.2)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        y_train = np.array([int(g.y.item()) for g in train_graphs])
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_losses_fold, train_accs_fold = [], []
        val_losses_fold, val_accs_fold = [], []

        for _ in range(epochs):
            model.train()
            total_train_loss = 0.0
            train_preds_epoch, train_true_epoch = [], []
            for batch_train in train_loader:
                batch_mixed = graph_mixup(batch_train, alpha=mixup_alpha)
                logits = model(batch_mixed)
                loss = bce(logits, batch_mixed.y_mix.float())
                total_train_loss += loss.item() * batch_mixed.num_graphs

                opt.zero_grad()
                loss.backward()
                opt.step()

                prob = torch.sigmoid(logits)
                train_preds_epoch.extend((prob >= 0.5).long().cpu().numpy().tolist())
                train_true_epoch.extend(batch_mixed.y.long().cpu().numpy().tolist())

            train_losses_fold.append(total_train_loss / len(train_graphs))
            train_accs_fold.append(accuracy_score(train_true_epoch, train_preds_epoch))

            model.eval()
            total_val_loss = 0.0
            val_preds_epoch, val_true_epoch = [], []
            with torch.no_grad():
                for batch_val in test_loader:
                    logits_val = model(batch_val)
                    loss_val = bce(logits_val, batch_val.y.float())
                    total_val_loss += loss_val.item() * batch_val.num_graphs
                    prob_val = torch.sigmoid(logits_val)
                    val_preds_epoch.extend((prob_val >= 0.5).long().cpu().numpy().tolist())
                    val_true_epoch.extend(batch_val.y.long().cpu().numpy().tolist())

            val_losses_fold.append(total_val_loss / len(test_graphs))
            val_accs_fold.append(accuracy_score(val_true_epoch, val_preds_epoch))

        # final fold evaluation
        model.eval()
        y_true_fold, y_pred_fold, y_proba_fold = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch)
                prob = torch.sigmoid(logits)
                pred = (prob >= 0.5).long().cpu().numpy().tolist()
                y_true_fold.extend(batch.y.long().cpu().numpy().tolist())
                y_pred_fold.extend(pred)
                y_proba_fold.extend(prob.cpu().numpy().tolist())

        acc = accuracy_score(y_true_fold, y_pred_fold)
        mf1 = f1_score(y_true_fold, y_pred_fold, average="macro", zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true_fold, y_pred_fold)
        cm = confusion_matrix(y_true_fold, y_pred_fold, labels=[0, 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        output_lines.append(f"Fold {fold}: ACC={acc:.3f}, Macro-F1={mf1:.3f}, Balanced ACC={balanced_acc:.3f}")
        output_lines.append("Confusion Matrix (rows=true, cols=pred) in %:")
        output_lines.append(np.array2string(cm_percent, formatter={"float_kind": lambda x: "%.2f%%" % x}))
        fold_results.append({"acc": acc, "mf1": mf1, "balanced_acc": balanced_acc})

        all_y_true.extend(y_true_fold)
        all_y_proba.extend(y_proba_fold)
        all_fold_train_losses.append(train_losses_fold)
        all_fold_train_accs.append(train_accs_fold)
        all_fold_val_losses.append(val_losses_fold)
        all_fold_val_accs.append(val_accs_fold)

    avg_acc = np.mean([r["acc"] for r in fold_results])
    std_acc = np.std([r["acc"] for r in fold_results])
    avg_mf1 = np.mean([r["mf1"] for r in fold_results])
    std_mf1 = np.std([r["mf1"] for r in fold_results])
    avg_bacc = np.mean([r["balanced_acc"] for r in fold_results])
    std_bacc = np.std([r["balanced_acc"] for r in fold_results])

    all_y_true = np.array(all_y_true)
    all_y_proba = np.array(all_y_proba)

    # ---- Bootstrap 95% CI on pooled out-of-fold predictions ----
    boot_acc = bootstrap_ci(all_y_true, all_y_proba, _acc_from_proba, n_boot=n_boot, seed=random_seed)
    boot_mf1 = bootstrap_ci(all_y_true, all_y_proba, _f1_from_proba, n_boot=n_boot, seed=random_seed)
    boot_bacc = bootstrap_ci(all_y_true, all_y_proba, _bacc_from_proba, n_boot=n_boot, seed=random_seed)
    try:
        boot_auc = bootstrap_ci(all_y_true, all_y_proba, roc_auc_safe, n_boot=n_boot, seed=random_seed)
    except Exception:
        boot_auc = (float("nan"), float("nan"), float("nan"))

    output_lines.append(f"\n{'='*80}")
    output_lines.append(f"Overall CV Results for {day_name} (missing_ratio={missing_ratio:.2f}):")
    output_lines.append(f"  Mean ACC: {avg_acc:.3f} (+/- {std_acc:.3f})")
    output_lines.append(f"  Mean Macro-F1: {avg_mf1:.3f} (+/- {std_mf1:.3f})")
    output_lines.append(f"  Mean Balanced ACC: {avg_bacc:.3f} (+/- {std_bacc:.3f})")
    output_lines.append(f"  [Bootstrap 95% CI, pooled OOF predictions, n_boot={n_boot}]")
    output_lines.append(f"    ACC          : {boot_acc[0]:.3f}  CI=[{boot_acc[1]:.3f}, {boot_acc[2]:.3f}]")
    output_lines.append(f"    Macro-F1     : {boot_mf1[0]:.3f}  CI=[{boot_mf1[1]:.3f}, {boot_mf1[2]:.3f}]")
    output_lines.append(f"    Balanced ACC : {boot_bacc[0]:.3f}  CI=[{boot_bacc[1]:.3f}, {boot_bacc[2]:.3f}]")
    output_lines.append(f"    ROC-AUC      : {boot_auc[0]:.3f}  CI=[{boot_auc[1]:.3f}, {boot_auc[2]:.3f}]")
    output_lines.append(f"{'='*80}")

    text = "\n".join(output_lines)
    if verbose:
        print(text)

    result = {
        "text": text,
        "y_true": all_y_true,
        "y_proba": all_y_proba,
        "fold_results": fold_results,
        "cv_summary": {
            "acc_mean": avg_acc, "acc_std": std_acc,
            "mf1_mean": avg_mf1, "mf1_std": std_mf1,
            "bacc_mean": avg_bacc, "bacc_std": std_bacc,
        },
        "bootstrap_ci": {
            "acc": boot_acc, "macro_f1": boot_mf1,
            "balanced_acc": boot_bacc, "roc_auc": boot_auc,
        },
        "train_losses": np.array(all_fold_train_losses),
        "train_accs": np.array(all_fold_train_accs),
        "val_losses": np.array(all_fold_val_losses),
        "val_accs": np.array(all_fold_val_accs),
        "model_last_fold": model,   # 마지막 fold 모델 (biomarker attribution 등에서 재사용 가능)
        "bio_medians_last_fold": bio_medians,
    }
    return result
