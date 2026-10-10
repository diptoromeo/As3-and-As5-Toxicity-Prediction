# # -*- coding: utf-8 -*-
# """
# main.py
# ========
# 전체 배포 파이프라인 실행 스크립트.

# 실행:
#     python main.py

# 수행 작업:
#   A. 기본 파이프라인: As3+/As5+ x Day-10/Day-20  5-fold CV 학습/평가
#      - Loss/Accuracy 커브, ROC 커브, Confusion Matrix
#      - Bootstrap 95% CI (추가 실험 #5)
#   B. Dose-Response Curve Fitting (Hill/Logistic vs GCN)            (추가 실험 #1)
#   C. Biomarker Attribution (gradient-based)                         (추가 실험 #2)
#   D. Threshold Sensitivity Analysis (±10%, ±20%)                    (추가 실험 #3)
#   E. Missing-Data 비율 실험 (0/10/20/30%)                            (추가 실험 #6)
#   F. Cross-Species Generalizability 프레임워크 (zero-shot/transfer)  (추가 실험 #4)

# 모든 산출물은 outputs/<species>/<day>/ 아래에 저장된다.
# """
# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc

# from core import DATASETS, BIOMARKERS, parse_day_blocks
# from train_eval import run_cv_for_day
# import experiments as exp

# OUT_ROOT = "outputs"
# EPOCHS_MAIN = 100          # 메인 학습 epoch (필요시 조정)
# EPOCHS_AUX = 60           # 보조 실험(threshold/dose-response 등) epoch, 속도 위해 축소 가능
# MISSING_RATIOS = [0.0, 0.10, 0.20, 0.30]


# def ensure_dir(p):
#     os.makedirs(p, exist_ok=True)
#     return p


# def plot_loss_acc(res, species, day_name, out_dir):
#     tl, ta = res["train_losses"], res["train_accs"]
#     vl, va = res["val_losses"], res["val_accs"]
#     epochs_range = range(tl.shape[1])

#     mean_tl, std_tl = tl.mean(0), tl.std(0)
#     mean_vl, std_vl = vl.mean(0), vl.std(0)
#     mean_ta, std_ta = ta.mean(0), ta.std(0)
#     mean_va, std_va = va.mean(0), va.std(0)

#     plt.figure(figsize=(10, 5))
#     plt.plot(epochs_range, mean_tl, label="Mean Train Loss", color="blue")
#     plt.fill_between(epochs_range, mean_tl - std_tl, mean_tl + std_tl, color="blue", alpha=0.2)
#     plt.plot(epochs_range, mean_vl, label="Mean Val Loss", color="orange")
#     plt.fill_between(epochs_range, mean_vl - std_vl, mean_vl + std_vl, color="orange", alpha=0.2)
#     plt.title(f"Loss ({species}, {day_name})", fontsize=13, fontweight="bold")
#     plt.xlabel("Epoch", fontsize=11, fontweight="bold")
#     plt.ylabel("Loss", fontsize=11, fontweight="bold")
#     plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, f"loss_{species}_{day_name}.png"), dpi=150)
#     plt.close()

#     plt.figure(figsize=(10, 5))
#     plt.plot(epochs_range, mean_ta, label="Mean Train Acc", color="green")
#     plt.fill_between(epochs_range, mean_ta - std_ta, mean_ta + std_ta, color="green", alpha=0.2)
#     plt.plot(epochs_range, mean_va, label="Mean Val Acc", color="red")
#     plt.fill_between(epochs_range, mean_va - std_va, mean_va + std_va, color="red", alpha=0.2)
#     plt.title(f"Accuracy ({species}, {day_name})", fontsize=13, fontweight="bold")
#     plt.xlabel("Epoch", fontsize=11, fontweight="bold")
#     plt.ylabel("Accuracy", fontsize=11, fontweight="bold")
#     plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, f"accuracy_{species}_{day_name}.png"), dpi=150)
#     plt.close()


# def plot_roc(res, species, day_name, out_dir):
#     fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
#     auc_score = auc(fpr, tpr)
#     plt.figure(figsize=(7, 6))
#     plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={auc_score:.3f})")
#     plt.plot([0, 1], [0, 1], color="navy", lw=2, ls="--")
#     plt.xlabel("False Positive Rate", fontsize=11, fontweight="bold")
#     plt.ylabel("True Positive Rate", fontsize=11, fontweight="bold")
#     plt.title(f"ROC ({species}, {day_name})", fontsize=13, fontweight="bold")
#     plt.legend(loc="lower right"); plt.grid(True, alpha=0.4); plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, f"roc_{species}_{day_name}.png"), dpi=150)
#     plt.close()
#     return auc_score


# def run_missing_data_experiment(samples, species, day_name, out_dir):
#     rows = []
#     for ratio in MISSING_RATIOS:
#         res = run_cv_for_day(samples, f"{day_name} (missing={ratio:.0%})",
#                               epochs=EPOCHS_AUX, missing_ratio=ratio, verbose=False)
#         rows.append({
#             "missing_ratio": ratio,
#             "acc_mean": res["cv_summary"]["acc_mean"],
#             "macro_f1_mean": res["cv_summary"]["mf1_mean"],
#             "balanced_acc_mean": res["cv_summary"]["bacc_mean"],
#         })
#     import pandas as pd
#     df = pd.DataFrame(rows)
#     csv_path = os.path.join(out_dir, f"missing_data_{species}_{day_name}.csv")
#     df.to_csv(csv_path, index=False, encoding="utf-8-sig")

#     plt.figure(figsize=(8, 6))
#     plt.plot(df["missing_ratio"] * 100, df["acc_mean"], marker="o", label="Accuracy")
#     plt.plot(df["missing_ratio"] * 100, df["macro_f1_mean"], marker="s", label="Macro-F1")
#     plt.plot(df["missing_ratio"] * 100, df["balanced_acc_mean"], marker="^", label="Balanced ACC")
#     plt.xlabel("Missing data ratio (%)", fontsize=12, fontweight="bold")
#     plt.ylabel("CV mean score", fontsize=12, fontweight="bold")
#     plt.title(f"Missing-Data Robustness ({species}, {day_name})", fontsize=13, fontweight="bold")
#     plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
#     fig_path = os.path.join(out_dir, f"missing_data_{species}_{day_name}.png")
#     plt.savefig(fig_path, dpi=150)
#     plt.close()
#     return df, csv_path, fig_path


# def main():
#     ensure_dir(OUT_ROOT)
#     master_summary = {}

#     for species, info in DATASETS.items():
#         species_dir = ensure_dir(os.path.join(OUT_ROOT, species))
#         samples_by_day = parse_day_blocks(info["path"], species)
#         master_summary[species] = {}

#         for day in (10, 20):
#             day_name = f"Day-{day}"
#             day_dir = ensure_dir(os.path.join(species_dir, day_name))
#             samples = samples_by_day[day]
#             print(f"\n\n######## {species} | {day_name} | N={len(samples)} ########")

#             # ---------- A. Baseline CV + Bootstrap CI ----------
#             res = run_cv_for_day(samples, day_name, epochs=EPOCHS_MAIN)
#             with open(os.path.join(day_dir, f"cv_results_{species}_{day_name}.txt"), "w", encoding="utf-8") as f:
#                 f.write(res["text"])
#             plot_loss_acc(res, species, day_name, day_dir)
#             auc_score = plot_roc(res, species, day_name, day_dir)

#             with open(os.path.join(day_dir, f"bootstrap_ci_{species}_{day_name}.json"), "w", encoding="utf-8") as f:
#                 json.dump(res["bootstrap_ci"], f, indent=2, ensure_ascii=False)

#             # ---------- B. Dose-Response Curve Fitting ----------
#             dr_text, dr_summary = exp.fit_dose_response(samples, day_name, species, day_dir)
#             with open(os.path.join(day_dir, f"dose_response_{species}_{day_name}.txt"), "w", encoding="utf-8") as f:
#                 f.write(dr_text)

#             # ---------- C. Biomarker Attribution ----------
#             attr_df, attr_csv, attr_fig = exp.biomarker_attribution(
#                 samples, day_name, species, day_dir, epochs=EPOCHS_AUX)

#             # ---------- D. Threshold Sensitivity Analysis ----------
#             thr_df, thr_csv, thr_fig = exp.threshold_sensitivity(
#                 species, day, info["path"], day_name, day_dir, epochs=EPOCHS_AUX)

#             # ---------- E. Missing-Data Robustness ----------
#             md_df, md_csv, md_fig = run_missing_data_experiment(samples, species, day_name, day_dir)

#             master_summary[species][day_name] = {
#                 "n_samples": len(samples),
#                 "cv_acc_mean": res["cv_summary"]["acc_mean"],
#                 "cv_macro_f1_mean": res["cv_summary"]["mf1_mean"],
#                 "cv_balanced_acc_mean": res["cv_summary"]["bacc_mean"],
#                 "roc_auc": auc_score,
#                 "bootstrap_ci": res["bootstrap_ci"],
#                 "dose_response": dr_summary,
#                 "top3_biomarkers": attr_df["biomarker"].tolist()[:3],
#             }

#         # ---------- F. Cross-Species Generalizability (per species) ----------
#         cs_dir = ensure_dir(os.path.join(species_dir, "cross_species"))
#         # 외부 종(제브라피시 Danio rerio 등) 데이터는 미포함 -> 아래 경로에 동일 포맷으로
#         # 데이터를 두면 자동으로 zero-shot/transfer-learning 평가가 수행됩니다.
#         external_path = os.path.join("data", f"external_{species}_zebrafish.xlsx")
#         cs_text, cs_result = exp.cross_species_generalization(
#             samples_by_day[10], f"{species}_Day10", external_path,
#             "Zebrafish (Danio rerio) / external species", day=10, out_dir=cs_dir,
#             epochs=EPOCHS_AUX)
#         with open(os.path.join(cs_dir, f"cross_species_{species}.txt"), "w", encoding="utf-8") as f:
#             f.write(cs_text)

#     with open(os.path.join(OUT_ROOT, "master_summary.json"), "w", encoding="utf-8") as f:
#         json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

#     print("\n\n모든 실험 완료. 결과는 'outputs/' 디렉터리에 저장되었습니다.")


# if __name__ == "__main__":
#     main()



# -*- coding: utf-8 -*-
"""
main.py
========
전체 배포 파이프라인 실행 스크립트.

실행:
    python main.py

수행 작업:
  A. 기본 파이프라인: As3+/As5+ x Day-10/Day-20  5-fold CV 학습/평가
     - Loss/Accuracy 커브, ROC 커브, Confusion Matrix
     - Bootstrap 95% CI (추가 실험 #5)
  B. Dose-Response Curve Fitting (Hill/Logistic vs GCN)            (추가 실험 #1)
  C. Biomarker Attribution (gradient-based)                         (추가 실험 #2)
  D. Threshold Sensitivity Analysis (±10%, ±20%)                    (추가 실험 #3)
  E. Missing-Data 비율 실험 (0/10/20/30%)                            (추가 실험 #6)
  F. Cross-Species Generalizability 프레임워크 (zero-shot/transfer)  (추가 실험 #4)

모든 산출물은 outputs/<species>/<day>/ 아래에 저장된다.
"""
import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 작은 표본(소규모 fold)에서 한 fold의 test set이 단일 클래스만 갖게 될 때 발생하는
# 무해한 sklearn 경고(예: threshold ±20%, missing_ratio=30% 같은 극단적 조건에서 발생).
# 지표 계산에는 영향 없음(zero_division=0 처리됨) -> 출력만 깔끔하게 숨김.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

from core import DATASETS, BIOMARKERS, parse_day_blocks
from train_eval import run_cv_for_day
import experiments as exp

OUT_ROOT = "outputs"
EPOCHS_MAIN = 100          # 메인 학습 epoch (필요시 조정)
EPOCHS_AUX = 60           # 보조 실험(threshold/dose-response 등) epoch, 속도 위해 축소 가능
MISSING_RATIOS = [0.0, 0.10, 0.20, 0.30]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def plot_loss_acc(res, species, day_name, out_dir):
    tl, ta = res["train_losses"], res["train_accs"]
    vl, va = res["val_losses"], res["val_accs"]
    epochs_range = range(tl.shape[1])

    mean_tl, std_tl = tl.mean(0), tl.std(0)
    mean_vl, std_vl = vl.mean(0), vl.std(0)
    mean_ta, std_ta = ta.mean(0), ta.std(0)
    mean_va, std_va = va.mean(0), va.std(0)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, mean_tl, label="Mean Train Loss", color="blue")
    plt.fill_between(epochs_range, mean_tl - std_tl, mean_tl + std_tl, color="blue", alpha=0.2)
    plt.plot(epochs_range, mean_vl, label="Mean Val Loss", color="orange")
    plt.fill_between(epochs_range, mean_vl - std_vl, mean_vl + std_vl, color="orange", alpha=0.2)
    plt.title(f"Loss ({species}, {day_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11, fontweight="bold")
    plt.ylabel("Loss", fontsize=11, fontweight="bold")
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"loss_{species}_{day_name}.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, mean_ta, label="Mean Train Acc", color="green")
    plt.fill_between(epochs_range, mean_ta - std_ta, mean_ta + std_ta, color="green", alpha=0.2)
    plt.plot(epochs_range, mean_va, label="Mean Val Acc", color="red")
    plt.fill_between(epochs_range, mean_va - std_va, mean_va + std_va, color="red", alpha=0.2)
    plt.title(f"Accuracy ({species}, {day_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=11, fontweight="bold")
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"accuracy_{species}_{day_name}.png"), dpi=150)
    plt.close()


def plot_roc(res, species, day_name, out_dir):
    fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, ls="--")
    plt.xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    plt.title(f"ROC ({species}, {day_name})", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"roc_{species}_{day_name}.png"), dpi=150)
    plt.close()
    return auc_score


def run_missing_data_experiment(samples, species, day_name, out_dir):
    rows = []
    for ratio in MISSING_RATIOS:
        res = run_cv_for_day(samples, f"{day_name} (missing={ratio:.0%})",
                              epochs=EPOCHS_AUX, missing_ratio=ratio, verbose=False)
        rows.append({
            "missing_ratio": ratio,
            "acc_mean": res["cv_summary"]["acc_mean"],
            "macro_f1_mean": res["cv_summary"]["mf1_mean"],
            "balanced_acc_mean": res["cv_summary"]["bacc_mean"],
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"missing_data_{species}_{day_name}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 6))
    plt.plot(df["missing_ratio"] * 100, df["acc_mean"], marker="o", label="Accuracy")
    plt.plot(df["missing_ratio"] * 100, df["macro_f1_mean"], marker="s", label="Macro-F1")
    plt.plot(df["missing_ratio"] * 100, df["balanced_acc_mean"], marker="^", label="Balanced ACC")
    plt.xlabel("Missing data ratio (%)", fontsize=12, fontweight="bold")
    plt.ylabel("CV mean score", fontsize=12, fontweight="bold")
    plt.title(f"Missing-Data Robustness ({species}, {day_name})", fontsize=13, fontweight="bold")
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
    fig_path = os.path.join(out_dir, f"missing_data_{species}_{day_name}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return df, csv_path, fig_path


def main():
    ensure_dir(OUT_ROOT)
    master_summary = {}

    for species, info in DATASETS.items():
        species_dir = ensure_dir(os.path.join(OUT_ROOT, species))
        samples_by_day = parse_day_blocks(info["path"], species)
        master_summary[species] = {}

        for day in (10, 20):
            day_name = f"Day-{day}"
            day_dir = ensure_dir(os.path.join(species_dir, day_name))
            samples = samples_by_day[day]
            print(f"\n\n######## {species} | {day_name} | N={len(samples)} ########")

            # ---------- A. Baseline CV + Bootstrap CI ----------
            res = run_cv_for_day(samples, day_name, epochs=EPOCHS_MAIN)
            with open(os.path.join(day_dir, f"cv_results_{species}_{day_name}.txt"), "w", encoding="utf-8") as f:
                f.write(res["text"])
            plot_loss_acc(res, species, day_name, day_dir)
            auc_score = plot_roc(res, species, day_name, day_dir)

            with open(os.path.join(day_dir, f"bootstrap_ci_{species}_{day_name}.json"), "w", encoding="utf-8") as f:
                json.dump(res["bootstrap_ci"], f, indent=2, ensure_ascii=False)

            # ---------- B. Dose-Response Curve Fitting ----------
            dr_text, dr_summary = exp.fit_dose_response(samples, day_name, species, day_dir)
            with open(os.path.join(day_dir, f"dose_response_{species}_{day_name}.txt"), "w", encoding="utf-8") as f:
                f.write(dr_text)

            # ---------- C. Biomarker Attribution ----------
            attr_df, attr_csv, attr_fig = exp.biomarker_attribution(
                samples, day_name, species, day_dir, epochs=EPOCHS_AUX)

            # ---------- D. Threshold Sensitivity Analysis ----------
            thr_df, thr_csv, thr_fig = exp.threshold_sensitivity(
                species, day, info["path"], day_name, day_dir, epochs=EPOCHS_AUX)

            # ---------- E. Missing-Data Robustness ----------
            md_df, md_csv, md_fig = run_missing_data_experiment(samples, species, day_name, day_dir)

            master_summary[species][day_name] = {
                "n_samples": len(samples),
                "cv_acc_mean": res["cv_summary"]["acc_mean"],
                "cv_macro_f1_mean": res["cv_summary"]["mf1_mean"],
                "cv_balanced_acc_mean": res["cv_summary"]["bacc_mean"],
                "roc_auc": auc_score,
                "bootstrap_ci": res["bootstrap_ci"],
                "dose_response": dr_summary,
                "top3_biomarkers": attr_df["biomarker"].tolist()[:3],
            }

        # ---------- F. Cross-Species Generalizability (per species) ----------
        cs_dir = ensure_dir(os.path.join(species_dir, "cross_species"))
        # 외부 종(제브라피시 Danio rerio 등) 데이터는 미포함 -> 아래 경로에 동일 포맷으로
        # 데이터를 두면 자동으로 zero-shot/transfer-learning 평가가 수행됩니다.
        external_path = os.path.join("data", f"external_{species}_zebrafish.xlsx")
        cs_text, cs_result = exp.cross_species_generalization(
            samples_by_day[10], f"{species}_Day10", external_path,
            "Zebrafish (Danio rerio) / external species", day=10, out_dir=cs_dir,
            epochs=EPOCHS_AUX)
        with open(os.path.join(cs_dir, f"cross_species_{species}.txt"), "w", encoding="utf-8") as f:
            f.write(cs_text)

    with open(os.path.join(OUT_ROOT, "master_summary.json"), "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n\n모든 실험 완료. 결과는 'outputs/' 디렉터리에 저장되었습니다.")


if __name__ == "__main__":
    main()
