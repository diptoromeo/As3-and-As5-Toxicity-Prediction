# As3+/As5+ 비소 독성 예측 GCN — Full Deployment Pipeline

원본 Colab 노트북(`As³__Toxicity_Prediction...ipynb`, `As⁵__Toxicity_Prediction...ipynb`)의
파이프라인을 로컬/서버에서 바로 실행 가능한 모듈형 코드로 재구성하고,
보고서 "1. 추가 실험 권고사항"의 6개 항목을 모두 코드로 구현했습니다.

## 폴더 구조

```
deploy/
├── core.py          # BIOMARKERS, THRESHOLDS, 엑셀 파싱, 그래프 생성, GCNBinary 모델, GraphMixUp
├── train_eval.py     # Stratified 5-fold CV 학습/평가 + Bootstrap 95% CI
├── experiments.py    # 추가 실험 1,2,3,4 구현 (dose-response/attribution/threshold/cross-species)
├── main.py           # 전체 파이프라인 실행 스크립트 (E. missing-data 실험 포함)
├── data/
│   ├── as3_raw_data.xlsx   (= 업로드한 3가_비소.xlsx)
│   └── as5_raw_data.xlsx   (= 업로드한 5가_비소.xlsx)
└── outputs/           # 실행 결과 전부 저장되는 디렉터리 (아래 "결과물" 참고)
```

## 실행 방법

```bash
pip install torch torch_geometric scikit-learn scipy pandas matplotlib openpyxl --break-system-packages
cd deploy
python main.py
```

`outputs/` 디렉터리에 As3 / As5 종, Day-10 / Day-20 시점별로 하위 폴더가 생성되고
모든 결과 파일(표/그림/텍스트/JSON)이 자동 저장됩니다. `main.py`의
`EPOCHS_MAIN`(기본 100), `EPOCHS_AUX`(기본 60) 값을 늘리면 원 노트북과 동일한
epochs=200 환경으로 재현할 수 있습니다(연산 시간 증가).

## 원본 노트북과의 주요 차이점 (실데이터 호환)

업로드된 실제 엑셀 파일(`3가_비소.xlsx`, `5가_비소.xlsx`)은 노트북이 기대하던
"concentration (mg/L)" 행 대신 `group` 행에 농도값이 들어있는 레이아웃이었습니다.
`core.parse_day_blocks()`가 두 표기 모두 자동 인식하도록 일반화했습니다
(나머지 라벨링 규칙·threshold·그래프 구조·GraphMixUp·GCN 구조는 노트북 그대로 유지).

## 추가 실험 권고사항 → 구현 매핑

| 권고사항 | 구현 위치 | 결과 파일 |
|---|---|---|
| 1. 용량-반응 곡선 적합 비교 (Dose-Response) | `experiments.fit_dose_response` | `dose_response_{species}_{day}.png/.txt` (Hill·Logistic EC50 vs GCN 곡선) |
| 2. 개별 바이오마커 기여도 분석 (Attribution) | `experiments.biomarker_attribution` | `biomarker_attribution_{species}_{day}.csv/.png` (gradient 기반 saliency, 11개 바이오마커 상대 중요도) |
| 3. 임계값 민감도 분석 (Threshold Sensitivity) | `experiments.threshold_sensitivity` | `threshold_sensitivity_{species}_{day}.csv/.png` (-20%~+20% 5단계, ACC/F1/BalACC 변화) |
| 4. 종간 일반화 (Cross-Species) | `experiments.cross_species_generalization` | `outputs/{species}/cross_species/cross_species_{species}.txt` |
| 5. Bootstrap 신뢰구간 (Model Robustness) | `train_eval.run_cv_for_day` 내 `bootstrap_ci` | `bootstrap_ci_{species}_{day}.json`, `cv_results_{species}_{day}.txt` 하단 |
| 6. Missing-Data 비율 실험 | `main.run_missing_data_experiment` | `missing_data_{species}_{day}.csv/.png` (0/10/20/30% 결측 스윕) |

### 4) 종간 일반화에 대한 중요 참고사항
본 패키지에는 제브라피시(Danio rerio)·송어류 데이터가 포함되어 있지 않습니다.
`cross_species_generalization()` 함수는 올리브 플라운더로 학습한 base 모델을 저장한 뒤,
`data/external_{species}_zebrafish.xlsx` 경로에 **동일한 엑셀 포맷**
(10일/20일 블록, `group` 행=농도(mg/L), 11개 biomarker 행)의 데이터를 두면
1) zero-shot 평가와 2) 일부 데이터로 fine-tuning 한 transfer-learning 평가를
자동으로 수행하도록 설계되어 있습니다. 현재는 파일이 없어 "실행 준비 완료" 보고서만
생성됩니다(`outputs/{species}/cross_species/cross_species_{species}.txt` 확인).

## 산출물 개요 (생성 확인 완료)

각 `outputs/{As3,As5}/{Day-10,Day-20}/` 폴더에 다음이 저장됩니다.
- `cv_results_*.txt` : fold별 ACC/Macro-F1/Balanced ACC, confusion matrix, 전체 평균 ± std, Bootstrap 95% CI
- `loss_*.png`, `accuracy_*.png` : fold 평균 ± std 학습/검증 곡선
- `roc_*.png` : ROC curve (AUC)
- `bootstrap_ci_*.json` : ACC/Macro-F1/BalancedACC/AUC의 95% CI 원본 수치
- `dose_response_*.png/.txt` : Hill/Logistic EC50 vs GCN 예측 곡선 비교
- `biomarker_attribution_*.csv/.png` : 바이오마커별 gradient 기반 상대 중요도
- `threshold_sensitivity_*.csv/.png` : threshold ±10%/±20% 시 성능 변화
- `missing_data_*.csv/.png` : 결측 비율 0/10/20/30%에 따른 성능 저하 곡선
- `outputs/{species}/cross_species/` : 종간 일반화 base 모델(.pth) 및 보고서
- `outputs/master_summary.json` : 전체 species × day 핵심 지표 요약
