from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, cast

import matplotlib

# 使用非交互后端，避免在终端/无 GUI 环境下出现 Tkinter 清理异常。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# 标签编码保持固定顺序，后续所有表格、混淆矩阵和热力图都按这个顺序展示。
LABEL_MAP = {
    "Normal": 0,
    "Blackhole": 1,
    "Grayhole": 2,
    "Flooding": 3,
    "TDMA": 4,
}

TARGET_COLUMN = "Attack type"
DROP_COLUMNS = ["id", "Time", "who CH", "send_code", TARGET_COLUMN]
MODEL_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest", "LightGBM", "CatBoost", "XGBoost"]
CLASS_NAMES = list(LABEL_MAP.keys())

def parse_args() -> argparse.Namespace:
    # 允许用户从命令行覆盖数据路径、输出目录和随机种子，方便重复实验。
    parser = argparse.ArgumentParser(
        description="Train and evaluate baseline intrusion detection models on WSN-DS without any imbalance handling."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/WSN-DS.csv"),
        help="Path to the WSN-DS CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wsn_ds_baseline"),
        help="Directory used to save metrics tables and plots.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset used for the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and model initialization.",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="stratified",
        choices=["stratified", "group_by_id", "time_order"],
        help=(
            "Data split strategy: stratified (random baseline), group_by_id (node-level split), "
            "time_order (first 80%% time for train, last 20%% for test)."
        ),
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="id",
        help="Group column for group_by_id split.",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default="Time",
        help="Time column for time_order split.",
    )
    parser.add_argument(
        "--do-feature-selection",
        action="store_true",
        help="Perform feature selection before model training.",
    )
    parser.add_argument(
        "--fs-method",
        type=str,
        default="xgb_importance",
        choices=["xgb_importance", "rfecv"],
        help="Feature selection method.",
    )
    parser.add_argument(
        "--fs-top-k",
        type=int,
        default=None,
        help="Top-k features to keep for xgb_importance.",
    )
    parser.add_argument(
        "--fs-threshold",
        type=str,
        default="median",
        help="Threshold for xgb_importance when fs_top_k is None: median/mean/float.",
    )
    return parser.parse_args()


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: list[str],
    output_dir: Path,
    random_state: int,
    selection_method: str,
    top_k: int | None,
    threshold_text: str,
) -> tuple[list[str], pd.DataFrame]:
    """按指定方法选择特征并返回选中特征列表与性能摘要。"""

    estimator = XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_names),
        random_state=random_state,
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        eval_metric="mlogloss",
        n_jobs=-1,
    )

    if selection_method == "xgb_importance":
        estimator.fit(X_train, y_train)
        importances = pd.Series(estimator.feature_importances_, index=X_train.columns)
        importances = importances.sort_values(ascending=False)
        imp_values = np.asarray(importances.to_numpy(dtype=float), dtype=float)

        if top_k is not None:
            keep_k = max(1, min(int(top_k), len(importances)))
            selected_features = importances.head(keep_k).index.tolist()
        else:
            if threshold_text == "median":
                threshold = float(np.median(imp_values))
            elif threshold_text == "mean":
                threshold = float(np.mean(imp_values))
            else:
                threshold = float(threshold_text)
            selected_features = importances[importances >= threshold].index.tolist()
            if not selected_features:
                selected_features = [importances.index[0]]

        plt.figure(figsize=(10, 6))
        importances.sort_values(ascending=True).plot(kind="barh", title="XGBoost Feature Importance")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance_xgb.png", dpi=300)
        plt.close()

    elif selection_method == "rfecv":
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=3,
            scoring="f1_macro",
            min_features_to_select=3,
            n_jobs=-1,
        )
        rfecv.fit(X_train, y_train)
        selected_features = X_train.columns[rfecv.support_].tolist()

        curve = pd.Series(rfecv.cv_results_["mean_test_score"])
        curve_values = np.asarray(curve.to_numpy(dtype=float), dtype=float)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(curve_values) + 1), curve_values)
        plt.xlabel("Number of features selected")
        plt.ylabel("CV Macro-F1")
        plt.title("RFECV Curve")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_selection_rfecv_curve.png", dpi=300)
        plt.close()

    else:
        raise ValueError(f"Unknown selection_method: {selection_method}")

    fs_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=len(class_names),
                    random_state=random_state,
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    eval_metric="mlogloss",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    fs_model.fit(X_train[selected_features], y_train)
    y_pred = fs_model.predict(X_test[selected_features])

    report = cast(
        dict[str, Any],
        classification_report(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
        ),
    )

    attack_recalls = [float(report[name]["recall"]) for name in class_names if name != "Normal"]
    summary = pd.DataFrame(
        [
            {
                "method": selection_method,
                "num_features": len(selected_features),
                "selected_features": ", ".join(selected_features),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
                "normal_recall": float(report["Normal"]["recall"]),
                "normal_false_alarm_rate": 1.0 - float(report["Normal"]["recall"]),
                "attack_macro_recall": float(np.mean(np.asarray(attack_recalls, dtype=float))),
            }
        ]
    )
    summary.to_csv(output_dir / "feature_selection_performance.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump({"selected_features": selected_features}, f, ensure_ascii=False, indent=2)

    print(f"[Feature Selection] Method={selection_method}, Selected={len(selected_features)}")
    print(f"[Feature Selection] Features={selected_features}")

    return selected_features, summary


def load_dataset(csv_path: Path) -> pd.DataFrame:
    # 先检查文件是否存在，避免后续在 pandas 读取时报出不够直接的错误信息。
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # WSN-DS 原始 CSV 的列名常带有前后空格，这里统一去掉，保证后续按列名取值稳定。
    df.columns = [column.strip() for column in df.columns]

    missing_columns = [column for column in DROP_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # 标签列同样可能存在隐藏空格，因此也做 strip；否则 map 编码时会产生 NaN。
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()

    unexpected_labels = sorted(set(df[TARGET_COLUMN]) - set(LABEL_MAP))
    if unexpected_labels:
        raise ValueError(f"Unexpected labels found in target column: {unexpected_labels}")

    return df


def build_models(random_state: int) -> dict[str, Pipeline]:
    # 使用统一的 Pipeline 包装三种模型。
    # 这样可以把训练流程保持一致，同时避免手工维护训练集/测试集的缩放逻辑。
    # 虽然树模型对标准化不敏感，但统一流程更利于后续横向对比。
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=6,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "LightGBM": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LGBMClassifier(
                        objective="multiclass",
                        num_class=len(LABEL_MAP),
                        random_state=random_state,
                        n_estimators=200,
                        class_weight="balanced",
                        verbosity=-1,
                    ),
                ),
            ]
        ),
        "CatBoost": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    CatBoostClassifier(
                        loss_function="MultiClass",
                        random_seed=random_state,
                        iterations=300,
                        auto_class_weights="Balanced",
                        verbose=0,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=len(LABEL_MAP),
                        eval_metric="mlogloss",
                        random_state=random_state,
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def split_dataset(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    split_strategy: str,
    group_column: str,
    time_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, int | float | str]]:
    # 该函数统一管理数据划分逻辑，确保不同策略下主流程完全一致，便于公平比较。
    split_meta: dict[str, int | float | str] = {"split_strategy": split_strategy}

    if split_strategy == "stratified":
        # 随机分层划分：类别比例稳定，但可能发生同一节点同时出现在训练和测试中的泄漏。
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            stratify=y,
            test_size=test_size,
            random_state=random_state,
        )
        split_meta.update(
            {
                "split_leakage_risk": "possible_node_overlap",
                "split_note": "random stratified split; optimistic risk exists when same node appears in both sets",
            }
        )
        return X_train, X_test, y_train, y_test, split_meta

    if split_strategy == "group_by_id":
        # 按节点分组划分：同一个 id 仅出现在训练集或测试集，评估更接近“新节点泛化”。
        if group_column not in df.columns:
            raise ValueError(f"group_column '{group_column}' not found in dataset columns.")

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        groups = df[group_column]
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        train_groups = set(groups.iloc[train_idx].tolist())
        test_groups = set(groups.iloc[test_idx].tolist())
        overlap_groups = train_groups.intersection(test_groups)

        split_meta.update(
            {
                "group_column": group_column,
                "train_unique_groups": int(len(train_groups)),
                "test_unique_groups": int(len(test_groups)),
                "overlap_groups": int(len(overlap_groups)),
                "split_leakage_risk": "low" if len(overlap_groups) == 0 else "unexpected_overlap",
            }
        )
        return X_train, X_test, y_train, y_test, split_meta

    if split_strategy == "time_order":
        # 时间顺序划分：按时间从早到晚切分，模拟“过去训练、未来测试”的在线部署场景。
        if time_column not in df.columns:
            raise ValueError(f"time_column '{time_column}' not found in dataset columns.")

        sorted_index = df.sort_values(by=time_column, ascending=True).index
        split_at = int((1.0 - test_size) * len(sorted_index))
        if split_at <= 0 or split_at >= len(sorted_index):
            raise ValueError("Invalid split index for time_order strategy. Please adjust test_size.")

        train_index = sorted_index[:split_at]
        test_index = sorted_index[split_at:]

        X_train = X.loc[train_index].copy()
        X_test = X.loc[test_index].copy()
        y_train = y.loc[train_index].copy()
        y_test = y.loc[test_index].copy()

        train_time = df.loc[train_index, time_column]
        test_time = df.loc[test_index, time_column]

        split_meta.update(
            {
                "time_column": time_column,
                "train_time_min": float(np.min(train_time)),
                "train_time_max": float(np.max(train_time)),
                "test_time_min": float(np.min(test_time)),
                "test_time_max": float(np.max(test_time)),
                "split_leakage_risk": "low_if_time_is_causal",
            }
        )
        return X_train, X_test, y_train, y_test, split_meta

    raise ValueError(f"Unsupported split_strategy: {split_strategy}")


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    # 混淆矩阵是逐类统计的基础，后面的 precision/recall/FPR/FNR 都由它推导出来。
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    records: list[dict[str, float | str]] = []

    for class_index, class_name in enumerate(class_names):
        # 二分类视角下，把“当前类”当成正类，其余所有类合并为负类。
        tp = cm[class_index, class_index]
        fn = cm[class_index, :].sum() - tp
        fp = cm[:, class_index].sum() - tp
        tn = cm.sum() - tp - fn - fp

        # 使用显式的零保护，避免在极端情况下出现除零警告。
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        records.append(
            {
                "class_name": class_name,
                "support": int(cm[class_index, :].sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
                "fnr": fnr,
            }
        )

    return pd.DataFrame(records)


def evaluate_model(
    model_name: str,
    y_true: Any,
    y_pred: Any,
    class_names: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    # classification_report 负责生成 sklearn 风格的标准报告；
    # 我们额外计算逐类 FPR/FNR，因为它们在入侵检测里同样重要。
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    per_class_df = compute_per_class_metrics(y_true, y_pred, class_names)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    report_df.to_csv(output_dir / f"{slugify(model_name)}_classification_report.csv", encoding="utf-8-sig")
    per_class_df.to_csv(output_dir / f"{slugify(model_name)}_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    cm_df.to_csv(output_dir / f"{slugify(model_name)}_confusion_matrix.csv", encoding="utf-8-sig")

    return report_df, per_class_df, cm


def plot_class_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    # 固定类别顺序，避免 value_counts 按频次排序导致图的横轴顺序不稳定。
    counts = df[TARGET_COLUMN].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int)
    ratio = counts / counts.sum()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=counts.index, y=counts.values, palette="crest")
    ax.set_title("WSN-DS Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Samples")
    ax.tick_params(axis="x", rotation=15)

    for index, value in enumerate(counts.values):
        ax.text(index, value, f"{value:,}\n{ratio.iloc[index]:.2%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=300)
    plt.close()


def plot_confusion_matrix(model_name: str, cm: np.ndarray, class_names: list[str], output_dir: Path) -> None:
    # 混淆矩阵直接展示每个真实类别被分到哪些类别，是最直观的错误分析方式。
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_dir / f"{slugify(model_name)}_confusion_matrix.png", dpi=300)
    plt.close()


def plot_summary_charts(summary_df: pd.DataFrame, output_dir: Path) -> None:
    # 第一张图：宏平均 F1。该指标对少数类更敏感，适合不平衡分类的横向比较。
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=summary_df, x="model", y="macro_f1", palette="viridis")
    ax.set_title("Macro-F1 Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)

    for patch, value in zip(ax.patches, summary_df["macro_f1"]):
        patch = cast(Any, patch)
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "macro_f1_comparison.png", dpi=300)
    plt.close()

    # 第二张图：Normal 被误报为攻击的比例。
    # 在入侵检测场景里，这个值过高会造成大量误告警，影响系统可用性。
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=summary_df, x="model", y="normal_false_alarm_rate", palette="mako")
    ax.set_title("Normal False Alarm Rate Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Normal False Alarm Rate")
    ax.set_ylim(0, 1)

    for patch, value in zip(ax.patches, summary_df["normal_false_alarm_rate"]):
        patch = cast(Any, patch)
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "normal_false_alarm_rate_comparison.png", dpi=300)
    plt.close()


def plot_per_class_heatmaps(per_class_df: pd.DataFrame, output_dir: Path) -> None:
    for metric in ["recall", "f1", "fpr"]:
        # 这里显式重排索引和列，保证热力图始终按预设类别顺序和模型顺序展示。
        # 否则 pandas pivot 后默认可能按字母序排列，和论文表格顺序不一致。
        pivot_df = per_class_df.pivot(index="class_name", columns="model", values=metric)
        pivot_df = pivot_df.reindex(index=CLASS_NAMES, columns=MODEL_ORDER)

        plt.figure(figsize=(8, 5))
        sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="YlGnBu", vmin=0, vmax=1)
        plt.title(f"Per-class {metric.upper()} Heatmap")
        plt.xlabel("Model")
        plt.ylabel("Class")
        plt.tight_layout()
        plt.savefig(output_dir / f"per_class_{metric}_heatmap.png", dpi=300)
        plt.close()


def slugify(value: str) -> str:
    # 统一输出文件名格式，避免空格带来的路径兼容问题。
    return value.lower().replace(" ", "_")


def print_console_summary(summary_df: pd.DataFrame, per_class_df: pd.DataFrame) -> None:
    # 终端摘要有意不打印 accuracy，避免在高度不平衡数据上造成误导。
    display_columns = [
        "model",
        "macro_f1",
        "weighted_f1",
        "normal_recall",
        "normal_false_alarm_rate",
        "attack_macro_recall",
        "attack_macro_f1",
    ]
    print("\n=== Model Summary (accuracy intentionally omitted) ===")
    print(summary_df[display_columns].round(4).to_string(index=False))

    print("\n=== Per-class Metrics ===")
    print(
        per_class_df[
            ["model", "class_name", "support", "precision", "recall", "f1", "fpr", "fnr"]
        ]
        .round(4)
        .to_string(index=False)
    )


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """从原始数据中构建输入特征与整数编码标签。"""
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET_COLUMN].map(LABEL_MAP)
    if y.isna().any():
        invalid_labels = sorted(df.loc[y.isna(), TARGET_COLUMN].unique().tolist())
        raise ValueError(f"Failed to encode target labels: {invalid_labels}")
    return X, y.astype(int)


def run_optional_feature_selection(
    args: argparse.Namespace,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str] | None]:
    """按参数决定是否执行特征选择，并返回更新后的训练/测试特征。"""
    selected_features: list[str] | None = None
    if args.do_feature_selection:
        selected_features, _ = select_features(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            output_dir=output_dir,
            random_state=args.random_state,
            selection_method=args.fs_method,
            top_k=args.fs_top_k,
            threshold_text=args.fs_threshold,
        )
        X_train = X_train[selected_features].copy()
        X_test = X_test[selected_features].copy()
    return X_train, X_test, selected_features


def build_run_metadata(
    args: argparse.Namespace,
    output_dir: Path,
    df: pd.DataFrame,
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    class_names: list[str],
    selected_features: list[str] | None,
    test_class_counts: pd.Series,
    zero_support_classes: list[str],
    split_meta: dict[str, int | float | str],
) -> dict[str, Any]:
    """组装实验元数据，便于复现与对照。"""
    metadata: dict[str, Any] = {
        "data_path": str(args.data),
        "output_dir": str(output_dir),
        "random_state": args.random_state,
        "test_size": args.test_size,
        "split_strategy": args.split_strategy,
        "train_size": int(len(X_train)),
        "test_size_samples": int(len(X_test)),
        "class_distribution": {
            class_name: int(count)
            for class_name, count in df[TARGET_COLUMN].value_counts().reindex(class_names).fillna(0).items()
        },
        "drop_columns": DROP_COLUMNS,
        "feature_columns": list(X.columns),
        "feature_selection": args.do_feature_selection,
        "selected_features": selected_features if selected_features is not None else list(X.columns),
        "imbalance_handling": "none",
        "test_class_distribution": {
            class_name: int(count) for class_name, count in zip(class_names, test_class_counts.tolist())
        },
        "zero_support_classes_in_test": zero_support_classes,
    }
    metadata.update(split_meta)
    return metadata


def train_and_evaluate_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """训练所有模型并返回汇总表与逐类指标表。"""
    summary_records: list[dict[str, float | str]] = []
    per_class_frames: list[pd.DataFrame] = []

    for model_name in MODEL_ORDER:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]

        _, per_class_df, cm = evaluate_model(
            model_name=model_name,
            y_true=y_test,
            y_pred=y_pred,
            class_names=class_names,
            output_dir=output_dir,
        )
        plot_confusion_matrix(model_name, cm, class_names, output_dir)

        per_class_df.insert(0, "model", model_name)
        per_class_frames.append(per_class_df)

        normal_metrics = per_class_df.loc[per_class_df["class_name"] == "Normal"].iloc[0]
        attack_metrics = per_class_df.loc[per_class_df["class_name"] != "Normal"]
        summary_records.append(
            {
                "model": model_name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
                "normal_recall": float(normal_metrics["recall"]),
                "normal_false_alarm_rate": 1.0 - float(normal_metrics["recall"]),
                "attack_macro_recall": float(attack_metrics["recall"].mean()),
                "attack_macro_f1": float(attack_metrics["f1"].mean()),
                "macro_fpr": float(per_class_df["fpr"].mean()),
                "macro_fnr": float(per_class_df["fnr"].mean()),
                "macro_precision_from_report": float(
                    precision_score(y_test, y_pred, average="macro", zero_division=0)
                ),
                "macro_recall_from_report": float(
                    recall_score(y_test, y_pred, average="macro", zero_division=0)
                ),
            }
        )

    summary_df = pd.DataFrame(summary_records)
    summary_df["model"] = pd.Categorical(summary_df["model"], categories=MODEL_ORDER, ordered=True)
    summary_df = summary_df.sort_values("model").reset_index(drop=True)

    all_per_class_df = pd.concat(per_class_frames, ignore_index=True)
    all_per_class_df["model"] = pd.Categorical(all_per_class_df["model"], categories=MODEL_ORDER, ordered=True)
    all_per_class_df["class_name"] = pd.Categorical(
        all_per_class_df["class_name"], categories=class_names, ordered=True
    )
    all_per_class_df = all_per_class_df.sort_values(["model", "class_name"]).reset_index(drop=True)
    return summary_df, all_per_class_df


def save_artifacts_and_report(
    summary_df: pd.DataFrame,
    all_per_class_df: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """统一处理结果落盘、可视化和终端摘要输出。"""
    summary_df.to_csv(output_dir / "model_summary.csv", index=False, encoding="utf-8-sig")
    all_per_class_df.to_csv(output_dir / "all_models_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    plot_summary_charts(summary_df, output_dir)
    plot_per_class_heatmaps(all_per_class_df, output_dir)
    print_console_summary(summary_df, all_per_class_df)

    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"\nArtifacts saved to: {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置统一绘图风格，保证输出图表可读性一致。
    sns.set_theme(style="whitegrid", font_scale=1.0)
    df = load_dataset(args.data)
    plot_class_distribution(df, output_dir)

    class_names = CLASS_NAMES

    X, y = prepare_features_and_target(df)

    # 支持三种划分方式：
    # 1) stratified: 传统随机分层，作为基线但可能存在节点泄漏。
    # 2) group_by_id: 按节点划分，评估对“未见节点”的泛化能力。
    # 3) time_order: 按时间先后划分，评估对未来时刻数据的泛化能力。
    X_train, X_test, y_train, y_test, split_meta = split_dataset(
        df=df,
        X=X,
        y=y,
        test_size=args.test_size,
        random_state=args.random_state,
        split_strategy=args.split_strategy,
        group_column=args.group_column,
        time_column=args.time_column,
    )

    # 某些严格划分（尤其是 time_order）可能导致少数类在测试集中缺失。
    # 这会让该类的召回/F1失去可解释性，因此显式提示并写入元数据。
    test_class_counts = y_test.value_counts().reindex(range(len(class_names))).fillna(0).astype(int)
    zero_support_classes = [class_names[idx] for idx, count in enumerate(test_class_counts) if count == 0]
    if zero_support_classes:
        print(
            "[Warning] The following classes have zero support in test set under current split strategy: "
            f"{zero_support_classes}"
        )

    X_train, X_test, selected_features = run_optional_feature_selection(
        args=args,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        output_dir=output_dir,
    )

    models = build_models(args.random_state)
    metadata = build_run_metadata(
        args=args,
        output_dir=output_dir,
        df=df,
        X=X,
        X_train=X_train,
        X_test=X_test,
        class_names=class_names,
        selected_features=selected_features,
        test_class_counts=test_class_counts,
        zero_support_classes=zero_support_classes,
        split_meta=split_meta,
    )

    summary_df, all_per_class_df = train_and_evaluate_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        output_dir=output_dir,
    )

    save_artifacts_and_report(
        summary_df=summary_df,
        all_per_class_df=all_per_class_df,
        output_dir=output_dir,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()