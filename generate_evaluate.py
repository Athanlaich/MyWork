#!/usr/bin/env python3
"""
WSN‑DS 少数类 CTGAN 过采样与数据质量评估（改进版）。

改进点：
- 彻底移除 id 等标识列，避免泄漏和高基数噪声。
- 默认 balance_ratio = 0.3，防止过度生成。
- 针对黑洞攻击、调度攻击定制离散列。
- 默认使用 GPU 训练，可通过 --no-gpu 切换为 CPU。
- 生成量大时自动提示风险。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from ctgan import CTGAN

warnings.filterwarnings("ignore")

# ----------  常量 ----------
LABEL_MAP = {
    "Normal": 0,
    "Blackhole": 1,
    "Grayhole": 2,
    "Flooding": 3,
    "TDMA": 4,
}
CLASS_NAMES = list(LABEL_MAP.keys())
TARGET_COLUMN = "Attack type"

# 始终删除的列（无攻击行为语义，或为唯一标识）
DROP_FEATURES = ["id", "Time"]

# 全局默认离散列（Is_CH 为 0/1 二值）
CATEGORICAL_COLS = ["Is_CH"]

# 按攻击类定制的离散列（会与全局离散列合并）
DISCRETE_COLS_BY_CLASS = {
    #"Blackhole": ["DATA_S", "DATA_R"],  # 黑洞攻击下数据包常为 0，按类别处理可避免生成负数小数
    "TDMA": ["Rank"],                   # 时隙序号是整数类别
}

# 按攻击类定义“硬约束”特征值。
# 说明：生成后会强制覆盖这些列，确保合成样本满足领域先验。
CLASS_HARD_CONSTRAINTS = {
    "Blackhole": {
        "Is_CH": 1,
        "ADV_S": 1,
        "Dist_To_CH": 0,
        "JOIN_S": 0,
        "dist_CH_To_BS": 0,
        "Rank": 0,
        "DATA_S": 0,
    }
}

# 每个攻击类的硬编码训练参数（可直接手动修改）
# 说明：
# - balance_ratio: 该类目标样本数 = Normal 类样本数 * balance_ratio
# - ctgan_epochs: 该类 CTGAN 训练轮数
# - ctgan_batch_size: 该类 CTGAN batch size
# - use_gpu: 是否对该类启用 GPU
# 若某项设置为 None，会回退到命令行参数对应的默认值。
ATTACK_TRAINING_CONFIG = {
    "Blackhole": {
        "balance_ratio": 0.1,
        "ctgan_epochs": None,
        "ctgan_batch_size": None,
        "use_gpu": None,
    },
    "Grayhole": {
        "balance_ratio": 0.1,
        "ctgan_epochs": None,
        "ctgan_batch_size": None,
        "use_gpu": None,
    },
    "Flooding": {
        "balance_ratio": 0.05,
        "ctgan_epochs": None,
        "ctgan_batch_size": None,
        "use_gpu": None,
    },
    "TDMA": {
        "balance_ratio": 0.08,
        "ctgan_epochs": None,
        "ctgan_batch_size": None,
        "use_gpu": None,
        "focus_train_ratio": 0.2,
        "focus_boundary_ratio": 0.30,
        "focus_filter_multiplier": 2.0,
        "disable_focus_training": True,
        "focus_features_override": [
            "Is_CH",
            "SCH_S",
            "DATA_S",
            "DATA_R",
            "Rank",
            "Dist_To_CH",
            "dist_CH_To_BS",
            "JOIN_R",
        ],
    },
}

# 数据质量评估中的关键特征
KEY_FEATURES = [
    "Is_CH",
    "ADV_S",
    "ADV_R",
    "JOIN_S",
    "JOIN_R",
    "SCH_S",
    "SCH_R",
    "Rank",
    "DATA_S",
    "DATA_R",
    "Dist_To_BS",
    "dist_CH_To_BS",
    "Expaned Energy",
]

# IDS 重点特征（来自当前筛选的 top7）。
# 用于生成后的真实性筛选，强化关键攻击行为特征分布的贴合度。
FOCUS_TOP7_FEATURES = [
    "Is_CH",
    "SCH_S",
    "ADV_S",
    "Expaned Energy",
    "Data_Sent_To_BS",
    "DATA_R",
    "JOIN_R",
]

# 物理上不应为负的特征。CTGAN 对连续列没有边界约束，
# 这些列在生成后需要做下界裁剪，避免出现负距离/负计数/负能量。
NONNEGATIVE_FEATURES = [
    "Dist_To_CH",
    "ADV_S",
    "ADV_R",
    "JOIN_S",
    "JOIN_R",
    "SCH_S",
    "SCH_R",
    "Rank",
    "DATA_S",
    "DATA_R",
    "Data_Sent_To_BS",
    "dist_CH_To_BS",
    "Expaned Energy",
]

# 容易出现重尾/极值漂移的关键连续特征。
CRITICAL_CONTINUOUS_FEATURES = [
    "DATA_R",
    "DATA_S",
    "Data_Sent_To_BS",
    "dist_CH_To_BS",
    "Dist_To_CH",
    "Rank",
    "SCH_S",
    "JOIN_R",
]

# ----------  命令行参数 ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 CTGAN 对 WSN‑DS 少数攻击类进行过采样并评估数据质量"
    )
    parser.add_argument("--data", type=Path, default=Path("data/raw/WSN-DS.csv"),
                        help="WSN-DS 数据集 CSV 路径")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ctgan_balance"),
                        help="输出目录")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="测试集比例")
    parser.add_argument("--random-state", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--ctgan-epochs", type=int, default=300,
                        help="CTGAN 训练总轮数")
    parser.add_argument("--ctgan-batch-size", type=int, default=500,
                        help="CTGAN 批次大小")
    parser.add_argument("--ctgan-quiet", action="store_true",
                        help="关闭 CTGAN 训练过程中的 epoch 进度输出")
    parser.add_argument("--balance-ratio", type=float, default=0.3,
                        help="少数类目标数量 = 多数类数量 * balance_ratio（推荐 0.2~0.5）")
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用 GPU 加速 CTGAN 训练（默认开启，使用 --no-gpu 关闭）",
    )
    parser.add_argument("--keep-who-ch", action="store_true",
                        help="保留 'who CH' 列（极不推荐，会严重破坏生成质量）")
    parser.add_argument("--keep-send-code", action="store_true",
                        help="保留 'send_code' 列（极不推荐）")
    parser.add_argument("--models-dir", type=Path, default=Path("models"),
                        help="CTGAN 模型缓存目录")
    parser.add_argument(
        "--reuse-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用模型复用（默认开启，使用 --no-reuse-models 关闭）",
    )
    parser.add_argument(
        "--focus-filter-multiplier",
        type=float,
        default=3.0,
        help="重点特征筛选时的候选倍数：最终保留 n 条，先生成 n*倍数 条候选。",
    )
    parser.add_argument(
        "--disable-focus-filter",
        action="store_true",
        help="关闭基于 top7 重点特征的真实性筛选（默认开启）。",
    )
    parser.add_argument(
        "--disable-focus-training",
        action="store_true",
        help="关闭 top7 子空间 CTGAN 训练分支（默认开启）。",
    )
    parser.add_argument(
        "--focus-train-ratio",
        type=float,
        default=0.5,
        help="候选样本中使用 top7 训练分支替换重点特征的比例（0~1）。",
    )
    parser.add_argument(
        "--focus-boundary-ratio",
        type=float,
        default=0.15,
        help="重点筛选中保留边界样本比例（0~0.5），用于缓解样本同质化。",
    )
    parser.add_argument(
        "--data-r-log-transform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否对 DATA_R 使用 log1p 训练并在生成后逆变换（默认开启）。",
    )
    parser.add_argument(
        "--critical-quantile-clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否对关键连续特征做按类分位裁剪，抑制极端值漂移（默认开启）。",
    )
    parser.add_argument(
        "--clip-lower-quantile",
        type=float,
        default=0.005,
        help="关键特征分位裁剪下界。",
    )
    parser.add_argument(
        "--clip-upper-quantile",
        type=float,
        default=0.995,
        help="关键特征分位裁剪上界。",
    )
    return parser.parse_args()


# ----------  数据加载 ----------
def load_and_preprocess(csv_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"数据集未找到：{csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
    unexpected = set(df[TARGET_COLUMN]) - set(LABEL_MAP)
    if unexpected:
        raise ValueError(f"发现未知标签：{unexpected}")

    drop_cols = DROP_FEATURES.copy()
    if not args.keep_who_ch:
        drop_cols.append("who CH")
    if not args.keep_send_code:
        drop_cols.append("send_code")

    # 去重并仅删除实际存在的列
    drop_cols = sorted(set(drop_cols))
    existing_drop = [c for c in drop_cols if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)
    return df


# ----------  CTGAN 训练 ----------
def train_ctgan_for_class(
    class_name: str,
    real_data: pd.DataFrame,
    discrete_cols: List[str],
    epochs: int,
    batch_size: int,
    random_state: int,
    verbose: bool,
    use_gpu: bool,
) -> CTGAN:
    start_ts = time.perf_counter()
    print(f"[CTGAN] {class_name} | samples={len(real_data)} | epochs={epochs} | "
          f"batch={batch_size} | discrete={discrete_cols} | gpu={use_gpu}")
    model = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        cuda=use_gpu,
        verbose=verbose,
    )
    model.fit(real_data, discrete_cols)
    elapsed = time.perf_counter() - start_ts
    print(f"[CTGAN] {class_name} done in {elapsed:.1f}s")
    return model


def build_model_cache_path(
    models_dir: Path,
    class_name: str,
    feature_cols: List[str],
    discrete_cols: List[str],
    epochs: int,
    batch_size: int,
    random_state: int,
    use_gpu: bool,
    real_sample_count: int,
    cache_tag: str = "",
) -> Path:
    """根据关键训练参数生成缓存文件路径。"""
    signature = {
        "class_name": class_name,
        "feature_cols": feature_cols,
        "discrete_cols": sorted(discrete_cols),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "random_state": int(random_state),
        "use_gpu": bool(use_gpu),
        "real_sample_count": int(real_sample_count),
        "cache_tag": cache_tag,
    }
    sig_text = json.dumps(signature, sort_keys=True, ensure_ascii=False)
    sig_hash = hashlib.md5(sig_text.encode("utf-8")).hexdigest()[:12]
    return models_dir / f"ctgan_{class_name}_{sig_hash}.pkl"


def load_cached_model(model_path: Path) -> CTGAN | None:
    if not model_path.exists():
        return None
    try:
        model = CTGAN.load(str(model_path))
        print(f"[CTGAN Reuse] loaded model: {model_path}")
        return model
    except Exception as exc:
        print(f"[CTGAN Reuse] failed to load model {model_path}: {exc}")
        return None


def save_cached_model(model: CTGAN, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(model_path))
        print(f"[CTGAN Reuse] saved model: {model_path}")
    except Exception as exc:
        print(f"[CTGAN Reuse] failed to save model {model_path}: {exc}")


def generate_samples(model: CTGAN, num_samples: int, columns: List[str],
                     class_name: str, class_label: int,
                     nonnegative_cols: List[str],
                     data_r_log_transform: bool) -> pd.DataFrame:
    synthetic = model.sample(num_samples)
    if not isinstance(synthetic, pd.DataFrame):
        synthetic = pd.DataFrame(synthetic, columns=columns)

    # DATA_R 在训练阶段可选 log1p 变换，生成后需逆变换恢复到原始尺度。
    if data_r_log_transform and "DATA_R" in synthetic.columns:
        # 防止 GAN 在 log 空间生成极端大值造成 exp 溢出。
        data_r_log = np.clip(np.maximum(synthetic["DATA_R"], 0.0), 0.0, 20.0)
        synthetic["DATA_R"] = np.expm1(data_r_log)

    # 对物理上不应为负的特征做下界截断。
    for col in nonnegative_cols:
        if col in synthetic.columns:
            synthetic[col] = synthetic[col].clip(lower=0)

    # 对已知攻击模式施加硬约束，避免生成样本偏离真实机理。
    hard_constraints = CLASS_HARD_CONSTRAINTS.get(class_name, {})
    for col, val in hard_constraints.items():
        if col in synthetic.columns:
            synthetic[col] = val

    synthetic[TARGET_COLUMN] = class_label
    return synthetic


def select_focus_realistic_samples(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    focus_features: List[str],
    keep_n: int,
    boundary_ratio: float,
) -> pd.DataFrame:
    """按重点特征进行“密度优先 + 边界保留”筛选，兼顾逼真度与多样性。"""
    if keep_n <= 0 or synthetic_data.empty:
        return synthetic_data.head(0).copy()

    valid_focus = [c for c in focus_features if c in real_data.columns and c in synthetic_data.columns]
    if not valid_focus:
        return synthetic_data.head(min(keep_n, len(synthetic_data))).copy()

    real_focus = real_data[valid_focus].apply(pd.to_numeric, errors="coerce")
    syn_focus = synthetic_data[valid_focus].apply(pd.to_numeric, errors="coerce")

    real_focus = real_focus.replace([np.inf, -np.inf], np.nan)
    syn_focus = syn_focus.replace([np.inf, -np.inf], np.nan)

    fill_values = real_focus.median(numeric_only=True).fillna(0.0)
    real_focus = real_focus.fillna(fill_values)
    syn_focus = syn_focus.fillna(fill_values)

    mu = real_focus.mean(axis=0)
    sigma = real_focus.std(axis=0).replace(0, 1.0).fillna(1.0)

    # 评分1：标准化 L1 距离（越小越接近均值行为）。
    z_syn = (syn_focus - mu) / sigma
    l1_score = z_syn.abs().mean(axis=1)

    # 评分2：马氏距离（基于收缩协方差，稳定性更好）。
    lw = LedoitWolf()
    lw.fit(real_focus.to_numpy(dtype=float))
    syn_mahal = lw.mahalanobis(syn_focus.to_numpy(dtype=float))
    real_mahal = lw.mahalanobis(real_focus.to_numpy(dtype=float))

    l1_norm = (l1_score - l1_score.min()) / (l1_score.max() - l1_score.min() + 1e-12)
    syn_mahal_series = pd.Series(syn_mahal, index=synthetic_data.index)
    mahal_norm = (syn_mahal_series - syn_mahal_series.min()) / (
        syn_mahal_series.max() - syn_mahal_series.min() + 1e-12
    )

    # 中心样本选择：联合评分偏低者。
    combo_score = 0.5 * l1_norm + 0.5 * mahal_norm
    keep_n = min(keep_n, len(synthetic_data))
    boundary_ratio = float(np.clip(boundary_ratio, 0.0, 0.5))
    boundary_n = int(np.floor(keep_n * boundary_ratio))
    center_n = keep_n - boundary_n

    center_idx = combo_score.nsmallest(center_n).index.tolist()

    # 边界样本选择：贴近真实样本的高分位边界，不取极端离群点。
    boundary_idx: List[int] = []
    if boundary_n > 0:
        target_boundary = float(np.quantile(real_mahal, 0.9))
        rest = syn_mahal_series.drop(index=center_idx, errors="ignore")
        if not rest.empty:
            upper_limit = float(np.quantile(syn_mahal_series, 0.98))
            rest = rest[rest <= upper_limit]
            if rest.empty:
                rest = syn_mahal_series.drop(index=center_idx, errors="ignore")
            boundary_idx = (rest - target_boundary).abs().nsmallest(boundary_n).index.tolist()

    selected_idx = center_idx + boundary_idx
    if len(selected_idx) < keep_n:
        fallback_idx = combo_score.drop(index=selected_idx, errors="ignore").nsmallest(keep_n - len(selected_idx)).index
        selected_idx.extend(fallback_idx.tolist())

    selected = synthetic_data.loc[selected_idx].copy()
    return selected.reset_index(drop=True)


def maybe_transform_data_r_for_training(real_data: pd.DataFrame, enable_log: bool) -> pd.DataFrame:
    """可选对 DATA_R 做 log1p，降低重尾分布对 GAN 训练的不利影响。"""
    transformed = real_data.copy()
    if enable_log and "DATA_R" in transformed.columns:
        transformed["DATA_R"] = np.log1p(transformed["DATA_R"].clip(lower=0))
    return transformed


def clip_critical_features_by_real_quantiles(
    synthetic_data: pd.DataFrame,
    real_data: pd.DataFrame,
    feature_names: List[str],
    lower_q: float,
    upper_q: float,
) -> pd.DataFrame:
    """按真实样本分位区间裁剪关键特征，抑制 GAN 极值外推。"""
    clipped = synthetic_data.copy()
    low_q = float(np.clip(lower_q, 0.0, 0.49))
    up_q = float(np.clip(upper_q, 0.51, 1.0))

    for col in feature_names:
        if col not in clipped.columns or col not in real_data.columns:
            continue
        real_col = pd.to_numeric(real_data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if real_col.empty:
            continue
        lower = float(real_col.quantile(low_q))
        upper = float(real_col.quantile(up_q))
        if upper < lower:
            lower, upper = upper, lower
        clipped[col] = pd.to_numeric(clipped[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        clipped[col] = clipped[col].fillna(real_col.median())
        clipped[col] = clipped[col].clip(lower=lower, upper=upper)

    return clipped


# ----------  质量评估 ----------
def compute_feature_distances(real: pd.DataFrame, synthetic: pd.DataFrame,
                              features: List[str]) -> pd.DataFrame:
    from scipy.stats import wasserstein_distance

    records = []
    # 根据全局离散列 + 每类定制的离散列，动态判断列的类型
    all_discrete = set(CATEGORICAL_COLS)
    # 这里简单处理：如果列在全局离散列中，按 TV 计算，否则按连续计算
    for col in features:
        if col not in real.columns:
            continue
        if col in all_discrete:
            real_counts = real[col].value_counts(normalize=True)
            syn_counts = synthetic[col].value_counts(normalize=True)
            all_vals = sorted(set(real_counts.index) | set(syn_counts.index))
            real_prob = np.array([real_counts.get(v, 0) for v in all_vals])
            syn_prob = np.array([syn_counts.get(v, 0) for v in all_vals])
            tv = 0.5 * np.sum(np.abs(real_prob - syn_prob))
            records.append({"feature": col, "distance": tv, "type": "tv"})
        else:
            dist = wasserstein_distance(real[col].dropna(), synthetic[col].dropna())
            records.append({"feature": col, "distance": dist, "type": "wasserstein"})
    return pd.DataFrame(records)


def plot_pca_comparison(real: pd.DataFrame, synthetic: pd.DataFrame,
                        class_name: str, output_path: Path):
    features = [c for c in real.columns if c != TARGET_COLUMN]
    scaler = StandardScaler()
    combined = pd.concat([real[features], synthetic[features]])
    scaled = scaler.fit_transform(combined)
    pca = PCA(n_components=2, random_state=42)
    pca_result = np.asarray(pca.fit_transform(scaled))
    n_real = len(real)

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:n_real, 0], pca_result[:n_real, 1],
                alpha=0.5, label=f"Real {class_name}", s=10)
    plt.scatter(pca_result[n_real:, 0], pca_result[n_real:, 1],
                alpha=0.5, label=f"Synthetic {class_name}", s=10)
    plt.title(f"PCA – {class_name}")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(); plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_discriminative_score(real: pd.DataFrame, synthetic: pd.DataFrame,
                                 n_folds: int = 5) -> float:
    features = [c for c in real.columns if c != TARGET_COLUMN]
    real = real[features].copy(); synthetic = synthetic[features].copy()
    real["label"] = 1; synthetic["label"] = 0
    data = pd.concat([real, synthetic], ignore_index=True)
    X = data[features]; y = data["label"]

    scores = []
    for _ in range(n_folds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y)
        clf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                     random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        pred = np.asarray(clf.predict_proba(X_test))[:, 1]
        try:
            auc = roc_auc_score(y_test, pred)
        except ValueError:
            auc = 0.5
        scores.append(auc)
    return float(np.mean(scores))


def plot_feature_distributions(real: pd.DataFrame, synthetic: pd.DataFrame,
                               class_name: str, output_dir: Path,
                               features: Optional[List[str]] = None,
                               discrete_cols_this_class: Optional[List[str]] = None):
    if features is None:
        features = [c for c in KEY_FEATURES if c in real.columns]
    else:
        features = [c for c in features if c in real.columns]
    if not features:
        return

    if discrete_cols_this_class is None:
        discrete_cols_this_class = CATEGORICAL_COLS

    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes_list: List[Any] = np.atleast_1d(axes).ravel().tolist()

    for i, col in enumerate(features):
        ax = axes_list[i]
        if col in discrete_cols_this_class:
            real_counts = real[col].value_counts(normalize=True)
            syn_counts = synthetic[col].value_counts(normalize=True)
            all_vals = sorted(set(real_counts.index) | set(syn_counts.index))
            real_probs = np.array([real_counts.get(v, 0.0) for v in all_vals], dtype=float)
            syn_probs = np.array([syn_counts.get(v, 0.0) for v in all_vals], dtype=float)
            idx = np.arange(len(all_vals))
            width = 0.35
            ax.bar(idx - width/2, real_probs, width, label="Real")
            ax.bar(idx + width/2, syn_probs, width, label="Synthetic")
            ax.set_xticks(idx)
            ax.set_xticklabels(all_vals)
            ax.set_title(col)
        else:
            sns.kdeplot(x=real[col], ax=ax, label="Real", fill=False)
            sns.kdeplot(x=synthetic[col], ax=ax, label="Synthetic", fill=False)
            ax.set_title(col)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes_list)):
        axes_list[j].set_visible(False)
    plt.suptitle(f"Feature Distributions – {class_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"distributions_{class_name}.png", dpi=150)
    plt.close()


# ----------  主流程 ----------
def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path("data/processed")
    synthetic_dir = Path("data/synthetic")
    processed_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    df = load_and_preprocess(args.data, args)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(LABEL_MAP)

    # 分层划分
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state)

    train_df = X_train.copy(); train_df[TARGET_COLUMN] = y_train
    test_df = X_test.copy(); test_df[TARGET_COLUMN] = y_test

    # 多数类与目标数量（每类可被 ATTACK_TRAINING_CONFIG 覆盖）
    majority_class = "Normal"
    class_counts = train_df[TARGET_COLUMN].map({v: k for k, v in LABEL_MAP.items()}).value_counts()
    majority_count = class_counts[majority_class]
    default_target_count = max(1, int(majority_count * args.balance_ratio))

    minority_classes = [c for c in CLASS_NAMES if c != majority_class]
    print(f"多数类 {majority_class}: {majority_count} 样本")
    print(f"默认少数类目标数量: {default_target_count} (balance_ratio={args.balance_ratio})")
    print(f"待增强类: {minority_classes}")

    # 特征列（已移除 id、who CH、send_code，除非保留）
    feature_cols = [c for c in train_df.columns if c != TARGET_COLUMN]
    # 全局离散列
    base_discrete = [c for c in CATEGORICAL_COLS if c in feature_cols]

    synthetic_parts = []
    quality_results = {}
    total_minority = len(minority_classes)

    for idx, class_name in enumerate(minority_classes, 1):
        class_label = LABEL_MAP[class_name]
        real_class_data = train_df[train_df[TARGET_COLUMN] == class_label][feature_cols].copy()
        n_real = len(real_class_data)

        # 读取“该攻击类”的硬编码参数；None 时回退到命令行参数。
        class_cfg = ATTACK_TRAINING_CONFIG.get(class_name, {})
        class_balance_ratio = class_cfg.get("balance_ratio")
        if class_balance_ratio is None:
            class_balance_ratio = args.balance_ratio
        class_target_count = max(1, int(majority_count * float(class_balance_ratio)))

        class_epochs = class_cfg.get("ctgan_epochs")
        if class_epochs is None:
            class_epochs = args.ctgan_epochs

        class_batch_size = class_cfg.get("ctgan_batch_size")
        if class_batch_size is None:
            class_batch_size = args.ctgan_batch_size

        class_use_gpu = class_cfg.get("use_gpu")
        if class_use_gpu is None:
            class_use_gpu = args.gpu

        class_focus_train_ratio = class_cfg.get("focus_train_ratio")
        if class_focus_train_ratio is None:
            class_focus_train_ratio = args.focus_train_ratio

        class_focus_boundary_ratio = class_cfg.get("focus_boundary_ratio")
        if class_focus_boundary_ratio is None:
            class_focus_boundary_ratio = args.focus_boundary_ratio

        class_focus_filter_multiplier = class_cfg.get("focus_filter_multiplier")
        if class_focus_filter_multiplier is None:
            class_focus_filter_multiplier = args.focus_filter_multiplier

        class_disable_focus_training = class_cfg.get("disable_focus_training")
        if class_disable_focus_training is None:
            class_disable_focus_training = args.disable_focus_training

        class_focus_features_override = class_cfg.get("focus_features_override")

        n_to_generate = max(0, class_target_count - n_real)

        print(f"\n{'='*50}\n[{idx}/{total_minority}] {class_name}  "
              f"(真实 {n_real}, 目标 {class_target_count}, 需生成 {n_to_generate})")
        print(
            f"  参数: ratio={class_balance_ratio}, epochs={class_epochs}, "
            f"batch={class_batch_size}, gpu={class_use_gpu}"
        )
        print(
            f"  focus参数: train_ratio={class_focus_train_ratio}, boundary_ratio={class_focus_boundary_ratio}, "
            f"filter_multiplier={class_focus_filter_multiplier}, disable_focus_training={class_disable_focus_training}"
        )
        if n_to_generate <= 0:
            print("  已达标，跳过。")
            continue

        # 生成倍数警告
        if n_real > 0:
            ratio = n_to_generate / n_real
            if ratio > 10:
                print(f"  ⚠️ 警告：生成量是真实样本的 {ratio:.1f} 倍，质量可能下降。"
                      f"建议降低 balance-ratio。")

        # 合并该类特定离散列
        extra_disc = DISCRETE_COLS_BY_CLASS.get(class_name, [])
        extra_disc = [c for c in extra_disc if c in feature_cols]
        class_discrete = sorted(set(base_discrete + extra_disc))

        model_cache_path = build_model_cache_path(
            models_dir=models_dir,
            class_name=class_name,
            feature_cols=feature_cols,
            discrete_cols=class_discrete,
            epochs=int(class_epochs),
            batch_size=int(class_batch_size),
            random_state=args.random_state,
            use_gpu=bool(class_use_gpu),
            real_sample_count=n_real,
            cache_tag=f"data_r_log={bool(args.data_r_log_transform)}",
        )

        real_class_data_train = maybe_transform_data_r_for_training(
            real_data=real_class_data,
            enable_log=bool(args.data_r_log_transform),
        )

        ctgan: CTGAN | None = None
        if args.reuse_models:
            ctgan = load_cached_model(model_cache_path)

        if ctgan is None:
            ctgan = train_ctgan_for_class(
                class_name=class_name,
                real_data=real_class_data_train,
                discrete_cols=class_discrete,
                epochs=int(class_epochs),
                batch_size=int(class_batch_size),
                random_state=args.random_state,
                verbose=not args.ctgan_quiet,
                use_gpu=bool(class_use_gpu),
            )
            if args.reuse_models:
                save_cached_model(ctgan, model_cache_path)

        if isinstance(class_focus_features_override, list) and class_focus_features_override:
            focus_cols = [c for c in class_focus_features_override if c in feature_cols]
        else:
            focus_cols = [c for c in FOCUS_TOP7_FEATURES if c in feature_cols]
        focus_ctgan: CTGAN | None = None
        use_focus_training = (not bool(class_disable_focus_training)) and len(focus_cols) >= 2
        if use_focus_training:
            focus_discrete = sorted(set([c for c in class_discrete if c in focus_cols]))
            focus_cache_path = build_model_cache_path(
                models_dir=models_dir,
                class_name=f"{class_name}_focus",
                feature_cols=focus_cols,
                discrete_cols=focus_discrete,
                epochs=int(class_epochs),
                batch_size=int(class_batch_size),
                random_state=args.random_state,
                use_gpu=bool(class_use_gpu),
                real_sample_count=n_real,
                cache_tag=f"data_r_log={bool(args.data_r_log_transform)}",
            )
            if args.reuse_models:
                focus_ctgan = load_cached_model(focus_cache_path)
            if focus_ctgan is None:
                focus_real = real_class_data_train[focus_cols].copy()
                focus_ctgan = train_ctgan_for_class(
                    class_name=f"{class_name}_focus",
                    real_data=focus_real,
                    discrete_cols=focus_discrete,
                    epochs=int(class_epochs),
                    batch_size=int(class_batch_size),
                    random_state=args.random_state,
                    verbose=not args.ctgan_quiet,
                    use_gpu=bool(class_use_gpu),
                )
                if args.reuse_models:
                    save_cached_model(focus_ctgan, focus_cache_path)

        class_nonnegative = [col for col in NONNEGATIVE_FEATURES if col in feature_cols]
        candidate_count = n_to_generate
        if not args.disable_focus_filter:
            multiplier = max(1.0, float(class_focus_filter_multiplier))
            candidate_count = max(n_to_generate, int(np.ceil(n_to_generate * multiplier)))

        syn_df = generate_samples(
            ctgan,
            candidate_count,
            feature_cols,
            class_name,
            class_label,
            class_nonnegative,
            data_r_log_transform=bool(args.data_r_log_transform),
        )

        if bool(args.critical_quantile_clip):
            syn_df = clip_critical_features_by_real_quantiles(
                synthetic_data=syn_df,
                real_data=real_class_data,
                feature_names=[c for c in CRITICAL_CONTINUOUS_FEATURES if c in feature_cols],
                lower_q=float(args.clip_lower_quantile),
                upper_q=float(args.clip_upper_quantile),
            )

        # top7 训练分支：替换部分候选样本的重点特征，增强关键行为维度拟合能力。
        replaced_focus_rows = 0
        if use_focus_training and focus_ctgan is not None and candidate_count > 0:
            focus_ratio = float(np.clip(class_focus_train_ratio, 0.0, 1.0))
            n_focus_mix = int(np.floor(candidate_count * focus_ratio))
            if n_focus_mix > 0:
                focus_syn = focus_ctgan.sample(n_focus_mix)
                if not isinstance(focus_syn, pd.DataFrame):
                    focus_syn = pd.DataFrame(focus_syn, columns=focus_cols)
                if bool(args.data_r_log_transform) and "DATA_R" in focus_syn.columns:
                    focus_data_r_log = np.clip(np.maximum(focus_syn["DATA_R"], 0.0), 0.0, 20.0)
                    focus_syn["DATA_R"] = np.expm1(focus_data_r_log)
                replace_idx = syn_df.index[:n_focus_mix]
                for col in focus_cols:
                    syn_df.loc[replace_idx, col] = focus_syn[col].to_numpy()
                replaced_focus_rows = n_focus_mix

                if bool(args.critical_quantile_clip):
                    syn_df = clip_critical_features_by_real_quantiles(
                        synthetic_data=syn_df,
                        real_data=real_class_data,
                        feature_names=[c for c in CRITICAL_CONTINUOUS_FEATURES if c in feature_cols],
                        lower_q=float(args.clip_lower_quantile),
                        upper_q=float(args.clip_upper_quantile),
                    )

        if not args.disable_focus_filter and candidate_count > n_to_generate:
            before_filter = len(syn_df)
            syn_df = select_focus_realistic_samples(
                real_data=real_class_data,
                synthetic_data=syn_df,
                focus_features=focus_cols,
                keep_n=n_to_generate,
                boundary_ratio=float(class_focus_boundary_ratio),
            )
            print(
                f"  重点特征筛选: 候选 {before_filter} -> 保留 {len(syn_df)} "
                f"(focus_features={focus_cols})"
            )
        if replaced_focus_rows > 0:
            print(f"  top7训练分支融合: 替换重点特征样本行数 {replaced_focus_rows}/{candidate_count}")

        synthetic_parts.append(syn_df)

        # 评估
        distances = compute_feature_distances(real_class_data, syn_df, feature_cols)
        distances_df = distances.sort_values("distance", ascending=False)
        distances_df.to_csv(output_dir / f"feature_distances_{class_name}.csv", index=False)

        plot_pca_comparison(real_class_data, syn_df[feature_cols], class_name,
                            output_dir / f"pca_{class_name}.png")
        plot_feature_distributions(real_class_data, syn_df, class_name, output_dir,
                                   KEY_FEATURES, class_discrete)

        disc_score = compute_discriminative_score(real_class_data, syn_df)
        quality_results[class_name] = {
            "real_count": n_real,
            "generated_count": n_to_generate,
            "candidate_count": int(candidate_count),
            "focus_replaced_rows": int(replaced_focus_rows),
            "focus_features": focus_cols,
            "focus_train_ratio": float(class_focus_train_ratio),
            "focus_boundary_ratio": float(class_focus_boundary_ratio),
            "focus_filter_multiplier": float(class_focus_filter_multiplier),
            "disable_focus_training": bool(class_disable_focus_training),
            "mean_feature_distance": float(distances["distance"].mean()),
            "discriminative_auc": disc_score,
        }
        print(f"  平均特征距离: {distances['distance'].mean():.4f}")
        print(f"  区分度 AUC: {disc_score:.4f} (理想值 ≈0.5)")

    # 合并并保存
    if synthetic_parts:
        all_synthetic = pd.concat(synthetic_parts, ignore_index=True)
        augmented_train = pd.concat([train_df, all_synthetic], ignore_index=True)
    else:
        all_synthetic = pd.DataFrame(columns=train_df.columns)
        augmented_train = train_df.copy()

    # 新生成数据放在 data/processed
    all_synthetic.to_csv(processed_dir / "generated_samples.csv", index=False)
    test_df.to_csv(processed_dir / "test_set.csv", index=False)

    # 平衡数据放在 data/synthetic
    augmented_train.to_csv(synthetic_dir / "balanced_train.csv", index=False)

    report = {
        "balance_ratio": args.balance_ratio,
        "target_count_per_minority_default": default_target_count,
        "original_train_size": len(train_df),
        "augmented_train_size": len(augmented_train),
        "test_size": len(test_df),
        "models_dir": str(models_dir),
        "reuse_models": bool(args.reuse_models),
        "processed_data_file": str(processed_dir / "generated_samples.csv"),
        "processed_test_file": str(processed_dir / "test_set.csv"),
        "synthetic_balanced_file": str(synthetic_dir / "balanced_train.csv"),
        "attack_training_config": ATTACK_TRAINING_CONFIG,
        "focus_top7_features": FOCUS_TOP7_FEATURES,
        "focus_training_enabled": not bool(args.disable_focus_training),
        "focus_train_ratio": float(args.focus_train_ratio),
        "focus_filter_enabled": not bool(args.disable_focus_filter),
        "focus_filter_multiplier": float(args.focus_filter_multiplier),
        "focus_boundary_ratio": float(args.focus_boundary_ratio),
        "data_r_log_transform": bool(args.data_r_log_transform),
        "critical_quantile_clip": bool(args.critical_quantile_clip),
        "clip_lower_quantile": float(args.clip_lower_quantile),
        "clip_upper_quantile": float(args.clip_upper_quantile),
        "class_quality": quality_results,
    }
    with open(output_dir / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✓ 报告与图表已保存至 {output_dir.resolve()}")
    print(f"  新生成数据: {(processed_dir / 'generated_samples.csv').resolve()}")
    print(f"  平衡训练集: {(synthetic_dir / 'balanced_train.csv').resolve()}")
    print("  请查看 quality_report.json 及各 PNG 图表。")


if __name__ == "__main__":
    main()