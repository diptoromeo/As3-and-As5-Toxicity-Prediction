# -*- coding: utf-8 -*-
"""
core.py
=======
공통 모듈: 데이터 파싱, 라벨링, 그래프 생성, GCN 모델, GraphMixUp.
As3+/As5+ 두 데이터셋과 Day-10/Day-20 두 시점에 대해 동일한 파이프라인을 재사용한다.
"""
import re
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# ============================================================
# BIOMARKERS (공통)
# ============================================================
BIOMARKERS = [
    "Hb", "Hct", "RBC count",
    "Calcium", "Magnesium", "Glucose",
    "Cholesterol", "Total protein",
    "AST", "ALT", "ALP",
]

# ============================================================
# DAY-DEPENDENT THRESHOLDS  tau_{m,day}: abnormal if conc >= tau
# (원본 노트북 As3+/As5+ 값 그대로 사용)
# ============================================================
THRESHOLDS = {
    "As3": {
        10: {"Hb": 2.0, "Hct": 2.0, "RBC count": 1.0, "Calcium": 2.0, "Magnesium": 8.0,
             "Glucose": 8.0, "Cholesterol": 2.0, "Total protein": 8.0,
             "AST": 4.0, "ALT": 4.0, "ALP": 8.0},
        20: {"Hb": 0.5, "Hct": 0.5, "RBC count": 0.5, "Calcium": 0.5, "Magnesium": 2.0,
             "Glucose": 2.0, "Cholesterol": 0.5, "Total protein": 0.5,
             "AST": 1.0, "ALT": 2.0, "ALP": 2.0},
    },
    "As5": {
        10: {"Hb": 2.0, "Hct": 4.0, "RBC count": 1.0, "Calcium": 2.0, "Magnesium": 8.0,
             "Glucose": 8.0, "Cholesterol": 8.0, "Total protein": 4.0,
             "AST": 2.0, "ALT": 8.0, "ALP": float("inf")},
        20: {"Hb": 4.0, "Hct": 2.0, "RBC count": 2.0, "Calcium": 0.5, "Magnesium": 4.0,
             "Glucose": 4.0, "Cholesterol": 2.0, "Total protein": 0.5,
             "AST": 4.0, "ALT": 4.0, "ALP": 4.0},
    },
}

DATASETS = {
    "As3": {"path": "C:/Users/romeo/Arsenic_Toxicity_Prediction/data/3가 비소.xlsx", "label": "As³⁺ (3가 비소)"},
    "As5": {"path": "C:/Users/romeo/Arsenic_Toxicity_Prediction/data/5가 비소.xlsx", "label": "As⁵⁺ (5가 비소)"},
}


# ============================================================
# Labeling
# ============================================================
def label_sample(conc: float, day: int, species: str, thresholds=None, agg="max") -> int:
    """0=정상, 1=비정상. biomarker-specific threshold tau_{m,day} 사용."""
    th = thresholds if thresholds is not None else THRESHOLDS[species][day]
    flags = [int(conc >= th[bm]) for bm in BIOMARKERS]
    if agg == "max":
        return int(max(flags))
    elif agg == "majority":
        return int(np.mean(flags) >= 0.5)
    raise ValueError("agg must be 'max' or 'majority'")


# ============================================================
# Excel parsing
# 실제 업로드된 원본 파일 레이아웃:
#   row0: "10일" / "20일"  (day marker)
#   row1: "group"          -> 농도값 (concentration, mg/L)
#   row2: "sample"         -> 샘플 인덱스 (무시)
#   row3~: 11개 biomarker
# (노트북 코드의 "concentration (mg/L)" 라벨 대신 "group" 행을 사용하도록 일반화)
# ============================================================
def parse_day_blocks(path: str, species: str, sheet_name="Sheet1", agg="max"):
    df = pd.read_excel(path, sheet_name=sheet_name)
    first_col = df.columns[0]
    col0 = df[first_col].astype(str).str.strip()

    day_starts = []
    for i, v in enumerate(col0.tolist()):
        v = str(v)
        if re.match(r"^\d+\s*일$", v):
            day_starts.append(i)
    if not day_starts:
        raise ValueError("'10일/20일' 블록을 찾을 수 없습니다. 시트 형식을 확인하세요.")

    samples_by_day = {10: [], 20: []}

    for k, start in enumerate(day_starts):
        end = day_starts[k + 1] if k + 1 < len(day_starts) else len(df)
        block = df.iloc[start:end].copy()
        block_col0 = block[first_col].astype(str).str.strip()

        day_text = block_col0.iloc[0]
        day = 10 if "10" in day_text else 20

        # concentration row: "concentration (mg/L)" 또는 "group" 둘 다 지원
        conc_row = None
        for tag in ["concentration (mg/L)", "group"]:
            hit = block[block_col0 == tag]
            if not hit.empty:
                conc_row = hit.iloc[0, 1:]
                break
        if conc_row is None:
            raise ValueError(f"'{day_text}' 블록에서 농도(concentration/group) 행을 찾을 수 없습니다.")

        sample_cols = conc_row.index

        var_map = {}
        for bm in BIOMARKERS:
            r = block[block_col0 == bm]
            if r.empty:
                raise ValueError(f"Missing biomarker row '{bm}' in {day_text} block.")
            var_map[bm] = r.iloc[0, 1:]

        for col in sample_cols:
            c = conc_row[col]
            if pd.isna(c):
                continue
            c = float(c)

            bio = []
            for bm in BIOMARKERS:
                val = var_map[bm][col]
                bio.append(float(val) if not pd.isna(val) else np.nan)
            bio = np.array(bio, dtype=np.float32)

            y = label_sample(c, day, species, agg=agg)
            samples_by_day[day].append({"conc": c, "bio": bio, "y": y})

    return samples_by_day


def load_all_datasets(agg="max"):
    """모든 species/day 데이터를 한 번에 로드."""
    out = {}
    for species, info in DATASETS.items():
        out[species] = parse_day_blocks(info["path"], species, agg=agg)
    return out


# ============================================================
# Graph build (2 nodes), feature dim = 11 biomarkers + 1 bias = 12
# ============================================================
def build_graph(sample, bio_medians, missing_mask=None):
    """
    sample: {"conc":..,"bio":..,"y":..}
    bio_medians: train-split에서 계산된 median (leakage 방지)
    missing_mask: bool array(11,) True인 위치는 강제로 결측 처리 후 median으로 대체
                  (실험 6: Missing-Data 비율 실험에서 사용)
    """
    conc = float(sample["conc"])
    bio = sample["bio"].copy()
    y = int(sample["y"])

    if missing_mask is not None:
        bio = bio.copy()
        bio[missing_mask] = np.nan

    for i in range(len(bio)):
        if np.isnan(bio[i]):
            bio[i] = bio_medians[i]

    x_as = np.zeros(11, dtype=np.float32)
    x_as[0] = conc
    x_as = np.concatenate([x_as, np.array([1.0], dtype=np.float32)], axis=0)

    x_bio = np.concatenate([bio.astype(np.float32), np.array([1.0], dtype=np.float32)], axis=0)

    x = torch.tensor(np.vstack([x_as, x_bio]), dtype=torch.float)

    edge_index = torch.tensor([[0, 1, 0, 1],
                                [0, 1, 1, 0]], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.float))


# ============================================================
# GraphMixUp (binary, soft targets)
# ============================================================
def graph_mixup(batch, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(batch.num_graphs)

    split_sizes = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
    x_per_graph = list(torch.split(batch.x, split_sizes, dim=0))

    x1_list, x2_list = [], []
    for i in range(batch.num_graphs):
        x1_list.append(x_per_graph[i])
        x2_list.append(x_per_graph[perm[i]])

    x1 = torch.cat(x1_list, dim=0)
    x2 = torch.cat(x2_list, dim=0)
    batch.x = lam * x1 + (1 - lam) * x2

    y1 = batch.y.view(-1, 1).float()
    y2 = y1[perm]
    batch.y_mix = (lam * y1 + (1 - lam) * y2).view(-1)
    return batch


# ============================================================
# GCN Binary Classifier
# ============================================================
class GCNBinary(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        g = global_mean_pool(x, batch)
        g = F.relu(self.fc1(g))
        g = F.dropout(g, p=self.dropout, training=self.training)
        logit = self.fc2(g).view(-1)
        return logit


def perturb_thresholds(species, pct):
    """threshold sensitivity 실험용: 모든 tau를 pct(%) 만큼 변경한 새 딕셔너리 반환."""
    new_th = {10: {}, 20: {}}
    for day in (10, 20):
        for bm, tau in THRESHOLDS[species][day].items():
            if np.isinf(tau):
                new_th[day][bm] = tau
            else:
                new_th[day][bm] = tau * (1.0 + pct / 100.0)
    return new_th
