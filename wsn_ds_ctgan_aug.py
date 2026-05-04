from __future__ import annotations

import argparse
import importlib
import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

# 复用原有代码，但为了避免 main 自动执行，只导入函数
import wsn_ds_baseline_ids as baseline

warnings.filterwarnings("ignore")

# 全局常量（与原代码保持一致）
LABEL_MAP = baseline.LABEL_MAP
TARGET_COLUMN = baseline.TARGET_COLUMN
DROP_COLUMNS = baseline.DROP_COLUMNS
MODEL_ORDER = baseline.MODEL_ORDER
CLASS_NAMES = baseline.CLASS_NAMES

# 默认使用的前 14 个重要特征（来自 baseline 的 XGBoost 重要性排序）
DEFAULT_FOCUS_FEATURES = [
    "Is_CH", "SCH_S", "ADV_S", "Expaned Energy",
    "Data_Sent_To_BS", "DATA_R", "JOIN_R", "JOIN_S",
    "ADV_R", "Rank", "DATA_S", "dist_CH_To_BS",
    "Dist_To_CH", "SCH_R"
]

# 稀有攻击类
RARE_ATTACKS = ["Grayhole", "Blackhole", "TDMA", "Flooding"]

# 训练分类器时投入到平衡训练集中的 Normal 类倍率，可直接在代码中手动修改。
NORMAL_CLASS_MULTIPLIER = 1.0


def load_ctgan_class() -> Any:
    """兼容不同 SDV 版本的 CTGAN 导入路径。"""
    candidates = [
        ("sdv.single_table", "CTGANSynthesizer"),  # 新版本
        ("sdv.tabular", "CTGAN"),  # 旧版本
    ]
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue

    raise ImportError(
        "未找到可用的 CTGAN 实现。请安装/升级 sdv（例如: pip install -U sdv）。"
    )


def build_ctgan_instance(
    ctgan_cls: Any,
    real_attack_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    use_gpu: bool,
    verbose: bool,
) -> Any:
    """构建 CTGAN 实例，兼容新旧 SDV 接口。"""
    if getattr(ctgan_cls, "__name__", "") == "CTGANSynthesizer":
        try:
            metadata_module = importlib.import_module("sdv.metadata")
            metadata_cls = getattr(metadata_module, "SingleTableMetadata")
            metadata = metadata_cls()
            metadata.detect_from_dataframe(data=real_attack_df)
            try:
                return ctgan_cls(metadata=metadata, epochs=epochs, batch_size=batch_size, verbose=verbose, cuda=use_gpu)
            except TypeError:
                return ctgan_cls(metadata=metadata, epochs=epochs, batch_size=batch_size, verbose=verbose)
        except Exception as exc:
            raise RuntimeError(f"初始化 CTGANSynthesizer 失败: {exc}") from exc

    try:
        return ctgan_cls(
            epochs=epochs,
            batch_size=batch_size,
            discriminator_steps=1,
            log_frequency=False,
            verbose=verbose,
            cuda=use_gpu,
        )
    except TypeError:
        return ctgan_cls(
            epochs=epochs,
            batch_size=batch_size,
            discriminator_steps=1,
            log_frequency=False,
            verbose=verbose,
        )


def fit_ctgan_model(ctgan: Any, real_attack_df: pd.DataFrame, discrete_cols: List[str]) -> None:
    """拟合 CTGAN，兼容新旧 SDV fit 签名。"""
    if getattr(ctgan.__class__, "__name__", "") == "CTGANSynthesizer":
        ctgan.fit(real_attack_df)
    else:
        ctgan.fit(real_attack_df, discrete_columns=discrete_cols)


def sample_ctgan_model(ctgan: Any, n_gen: int) -> pd.DataFrame:
    """采样 CTGAN，兼容新旧 SDV sample 签名。"""
    if getattr(ctgan.__class__, "__name__", "") == "CTGANSynthesizer":
        sampled = ctgan.sample(num_rows=n_gen)
    else:
        sampled = ctgan.sample(n_gen)
    if not isinstance(sampled, pd.DataFrame):
        return pd.DataFrame(sampled)
    return sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CTGAN 数据平衡 + 质量验证 + 模型对比"
    )
    parser.add_argument("--data", type=Path, default=Path("data/raw/WSN-DS.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/wsn_ds_ctgan"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-strategy", type=str, default="stratified",
                        choices=["stratified", "group_by_id", "time_order"])
    parser.add_argument("--group-column", type=str, default="id")
    parser.add_argument("--time-column", type=str, default="Time")
    parser.add_argument("--focus-features", type=str, default=None,
                        help="逗号分隔的特征列表，默认使用 XGBoost 重要性前 14 个")
    parser.add_argument(
        "--gen-multiplier",
        type=float,
        default=1.0,
        help="每个少数类的生成倍率：n_gen = real_count * gen_multiplier",
    )
    parser.add_argument("--ctgan-epochs", type=int, default=80,
                        help="CTGAN 训练轮数")
    parser.add_argument("--ctgan-batch-size", type=int, default=500)
    parser.add_argument(
        "--ctgan-verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否显示 CTGAN epoch 训练进度（默认开启，使用 --no-ctgan-verbose 关闭）",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models/ctgan"),
                        help="CTGAN 模型保存目录")
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 GPU 训练 CTGAN（默认开启，使用 --no-gpu 关闭）",
    )
    parser.add_argument("--quality-check", action="store_true", default=True,
                        help="是否进行生成质量验证")
    parser.add_argument("--train-models", action="store_true", default=True,
                        help="是否用平衡数据训练模型并对比")
    parser.add_argument(
        "--reuse-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否优先复用已训练的 CTGAN 模型（默认开启，使用 --no-reuse-models 关闭）",
    )
    return parser.parse_args()


def save_ctgan_model(model: Any, model_path: Path) -> None:
    """保存 CTGAN 模型，优先使用库自带 save，其次回退到 pickle。"""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_fn = getattr(model, "save", None)
        if callable(save_fn):
            save_fn(str(model_path))
        else:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        print(f"    模型已保存: {model_path}")
    except Exception as exc:
        print(f"    模型保存失败 ({model_path}): {exc}")

def load_ctgan_model(ctgan_cls: Any, model_path: Path) -> Any | None:
    """加载 CTGAN 模型，优先使用类方法 load，其次回退到 pickle。"""
    if not model_path.exists():
        return None
    try:
        load_fn = getattr(ctgan_cls, "load", None)
        if callable(load_fn):
            model = load_fn(str(model_path))
            print(f"    复用已保存模型: {model_path}")
            return model
    except Exception:
        pass

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"    复用已保存模型: {model_path}")
        return model
    except Exception as exc:
        print(f"    模型加载失败，将重新训练 ({model_path}): {exc}")
        return None


def compute_effective_ctgan_batch_size(requested_batch_size: int, real_count: int, pac: int = 10) -> int:
    """计算实际可用于 CTGAN 的 batch_size，保证能被 pac 整除。"""
    batch_size = min(int(requested_batch_size), int(real_count))
    return max(int(pac), (batch_size // int(pac)) * int(pac))


def build_ctgan_model_path(models_dir: Path, attack: str, epochs: int, batch_size: int) -> Path:
    """构建带训练轮数和 batch 大小标识的 CTGAN 模型文件名，便于复用与区分版本。"""
    return models_dir / f"ctgan_{attack}_e{int(epochs)}_b{int(batch_size)}.pkl"


def build_legacy_ctgan_model_paths(models_dir: Path, attack: str, epochs: int) -> List[Path]:
    """兼容历史 CTGAN 模型命名。"""
    return [
        models_dir / f"ctgan_{attack}_e{int(epochs)}.pkl",
        models_dir / f"ctgan_{attack}.pkl",
    ]


def resample_class_by_multiplier(
    X_class: pd.DataFrame,
    y_class: pd.Series,
    multiplier: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """按倍率重采样单个类别，使样本量约为原始数量的 multiplier 倍。"""
    base_count = len(X_class)
    target_count = max(0, int(round(base_count * float(multiplier))))
    if target_count == 0 or base_count == 0:
        return X_class.iloc[0:0].copy(), y_class.iloc[0:0].copy()

    sampled_index = X_class.sample(
        n=target_count,
        replace=target_count > base_count,
        random_state=random_state,
    ).index
    return X_class.loc[sampled_index].reset_index(drop=True), y_class.loc[sampled_index].reset_index(drop=True)


def summarize_balanced_label_distribution(y_balanced: pd.Series, output_dir: Path) -> pd.DataFrame:
    """统计并保存平衡训练集各类别的数量与占比。"""
    label_to_name = {value: key for key, value in LABEL_MAP.items()}
    counts = y_balanced.value_counts().sort_index()
    total = int(counts.sum())
    distribution_df = pd.DataFrame(
        {
            "label": counts.index.astype(int),
            "class_name": [label_to_name.get(int(label), f"label_{int(label)}") for label in counts.index],
            "count": counts.values.astype(int),
            "ratio": (counts / total).astype(float),
            "percentage": ((counts / total) * 100.0).astype(float),
        }
    )
    distribution_df.to_csv(output_dir / "balanced_class_distribution.csv", index=False)

    print("平衡训练集类别分布:")
    for row in distribution_df.itertuples(index=False):
        print(f"  {row.class_name}: count={row.count}, ratio={row.ratio:.4f}, percentage={row.percentage:.2f}%")

    return distribution_df


def prepare_focus_features(args: argparse.Namespace, X: pd.DataFrame) -> List[str]:
    """确定最终使用的特征列表"""
    if args.focus_features is None:
        return DEFAULT_FOCUS_FEATURES
    features = [f.strip() for f in args.focus_features.split(",")]
    missing = set(features) - set(X.columns)
    if missing:
        raise ValueError(f"指定特征不在数据集中: {missing}")
    return features


def build_balanced_eval_models(random_state: int) -> Dict[str, Any]:
    """为平衡数据评估构建无类别权重模型。"""
    models = baseline.build_models(random_state)

    # 训练集已做增强平衡后，关闭类别权重，避免对合成样本再次放大损失权重。
    override_params = {
        "Decision Tree": {"model__class_weight": None},
        "LightGBM": {"model__class_weight": None},  # LightGBM 的 balanced 是根据训练数据自动计算权重，适合增强后数据
        "CatBoost": {"model__auto_class_weights": "None"},
    }
    for model_name, params in override_params.items():
        if model_name not in models:
            continue
        try:
            models[model_name].set_params(**params)
        except Exception as exc:
            print(f"警告: 无法为 {model_name} 覆盖类别权重参数: {exc}")

    return models


def ctgan_balance_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    focus_features: List[str],
    target_column: str,
    gen_multiplier: float,
    rare_attacks: List[str],
    ctgan_epochs: int,
    ctgan_batch_size: int,
    use_gpu: bool,
    ctgan_verbose: bool,
    reuse_models: bool,
    models_dir: Path,
    random_state: int,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, pd.DataFrame]]:
    """
    对训练集中的稀有攻击类进行 CTGAN 生成，合并后返回平衡训练集。
    返回: 平衡后的特征矩阵, 平衡后的标签, 各类合成样本字典(用于质量验证)
    """
    ctgan_cls = load_ctgan_class()

    normal_mask = y_train == LABEL_MAP["Normal"]
    X_normal = X_train[normal_mask]
    y_normal = y_train[normal_mask]
    X_normal_resampled, y_normal_resampled = resample_class_by_multiplier(
        X_class=X_normal,
        y_class=y_normal,
        multiplier=NORMAL_CLASS_MULTIPLIER,
        random_state=random_state,
    )

    print(
        f"Normal 类倍率: {NORMAL_CLASS_MULTIPLIER} | original={len(X_normal)}, used={len(X_normal_resampled)}"
    )

    balanced_X_parts = [X_normal_resampled]
    balanced_y_parts = [y_normal_resampled]
    synthetic_data_dict = {}  # 存放每类生成的合成样本 DataFrame

    # 连续列：除了 Is_CH 以外的所有 focus_features
    all_features = focus_features + [target_column]
    discrete_cols = ["Is_CH"]  # 只有这一个真正的二值离散特征

    multiplier = max(0.0, float(gen_multiplier))
    total_attacks = len(rare_attacks)
    for idx, attack in enumerate(rare_attacks, 1):
        attack_label = LABEL_MAP[attack]
        real_mask = y_train == attack_label
        X_attack = X_train[real_mask]
        y_attack = y_train[real_mask]
        real_count = len(X_attack)
        n_gen = int(real_count * multiplier) if real_count > 0 else 0

        print(f"\n[{idx}/{total_attacks}] {attack} | real={real_count}, multiplier={multiplier}, need_generate={n_gen}")

        real_attack_df = X_attack.copy()
        real_attack_df[target_column] = attack

        if n_gen > 0:
            print(f"  为 {attack} 生成 {n_gen} 条合成样本...")
            effective_batch_size = compute_effective_ctgan_batch_size(
                requested_batch_size=ctgan_batch_size,
                real_count=real_count,
            )
            model_path = build_ctgan_model_path(
                models_dir=models_dir,
                attack=attack,
                epochs=ctgan_epochs,
                batch_size=effective_batch_size,
            )
            ctgan = None
            if reuse_models:
                ctgan = load_ctgan_model(ctgan_cls=ctgan_cls, model_path=model_path)
                if ctgan is None:
                    # 向后兼容旧命名，避免历史模型无法复用。
                    for legacy_model_path in build_legacy_ctgan_model_paths(
                        models_dir=models_dir,
                        attack=attack,
                        epochs=ctgan_epochs,
                    ):
                        ctgan = load_ctgan_model(ctgan_cls=ctgan_cls, model_path=legacy_model_path)
                        if ctgan is not None:
                            break

            if ctgan is None:
                ctgan = build_ctgan_instance(
                    ctgan_cls=ctgan_cls,
                    real_attack_df=real_attack_df,
                    epochs=ctgan_epochs,
                    batch_size=effective_batch_size,
                    use_gpu=use_gpu,
                    verbose=ctgan_verbose,
                )
                fit_ctgan_model(ctgan=ctgan, real_attack_df=real_attack_df, discrete_cols=discrete_cols)
                save_ctgan_model(ctgan, model_path)
            synthetic = sample_ctgan_model(ctgan=ctgan, n_gen=n_gen)

            # 后处理：计数/流量类特征保留连续值，仅做非负约束；二值特征保持离散。
            count_like_features = [
                "SCH_S", "ADV_S", "JOIN_R", "DATA_R", "Data_Sent_To_BS",
                "ADV_R", "JOIN_S", "SCH_R", "DATA_S",
            ]
            for feat in count_like_features:
                if feat in synthetic.columns:
                    synthetic[feat] = synthetic[feat].clip(lower=0)
            if "Is_CH" in synthetic.columns:
                synthetic["Is_CH"] = (synthetic["Is_CH"] >= 0.5).astype(int)

            synthetic[target_column] = attack
            synthetic_data_dict[attack] = synthetic

            X_syn = synthetic.drop(columns=[target_column])
            y_syn = pd.Series(LABEL_MAP[attack], index=X_syn.index)
            balanced_X_parts.append(X_syn)
            balanced_y_parts.append(y_syn)
        else:
            print(f"  {attack} 无需生成（已达到目标数量）")

        # 无论是否生成，都加入真实样本
        balanced_X_parts.append(X_attack)
        balanced_y_parts.append(y_attack)

    X_balanced = pd.concat(balanced_X_parts, ignore_index=True)
    y_balanced = pd.concat(balanced_y_parts, ignore_index=True)

    # 打乱顺序
    combined = pd.concat([X_balanced, y_balanced.rename("label")], axis=1)
    combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
    X_balanced = combined.drop(columns=["label"])
    y_balanced = combined["label"].astype(int)

    summarize_balanced_label_distribution(y_balanced=y_balanced, output_dir=output_dir)

    return X_balanced, y_balanced, synthetic_data_dict


def quality_verification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    synthetic_data_dict: Dict[str, pd.DataFrame],
    focus_features: List[str],
    target_column: str,
    output_dir: Path,
) -> None:
    """生成质量验证：PCA 图、KS 检验、描述统计对比"""
    print("\n========== 生成质量验证 ==========")
    quality_dir = output_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    def plot_feature_distribution_comparison(
        attack_name: str,
        real_df: pd.DataFrame,
        syn_df_feat: pd.DataFrame,
        features: List[str],
        save_path: Path,
    ) -> None:
        """为单个攻击类绘制 14 个特征的真实/合成分布对比大图。"""
        n_features = len(features)
        n_cols = 4
        n_rows = int(np.ceil(n_features / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 4.6 * n_rows), constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()

        real_color = "#1f77b4"
        syn_color = "#d62728"

        for idx, feat in enumerate(features):
            ax = axes[idx]
            real_vals = pd.to_numeric(real_df[feat], errors="coerce").dropna().to_numpy()
            syn_vals = pd.to_numeric(syn_df_feat[feat], errors="coerce").dropna().to_numpy()

            if len(real_vals) == 0 or len(syn_vals) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
                ax.set_title(feat)
                ax.set_axis_off()
                continue

            combined_vals = np.concatenate([real_vals, syn_vals])
            unique_count = np.unique(combined_vals).size

            if unique_count <= 12:
                unique_vals = np.unique(combined_vals)
                if unique_vals.size == 1:
                    step = 1.0
                else:
                    diffs = np.diff(unique_vals)
                    positive_diffs = diffs[diffs > 0]
                    step = float(np.min(positive_diffs)) if positive_diffs.size > 0 else 1.0

                bin_edges = np.concatenate([
                    [unique_vals[0] - step / 2],
                    (unique_vals[:-1] + unique_vals[1:]) / 2,
                    [unique_vals[-1] + step / 2],
                ])
                real_counts = pd.Series(real_vals).value_counts(normalize=True)
                syn_counts = pd.Series(syn_vals).value_counts(normalize=True)
                real_density = np.array([real_counts.get(value, 0.0) for value in unique_vals])
                syn_density = np.array([syn_counts.get(value, 0.0) for value in unique_vals])

                ax.stairs(
                    values=real_density,
                    edges=bin_edges,
                    color=real_color,
                    linewidth=2.6,
                    alpha=0.95,
                    linestyle="-",
                    label="Real",
                )
                ax.stairs(
                    values=syn_density,
                    edges=bin_edges,
                    color=syn_color,
                    linewidth=2.6,
                    alpha=0.95,
                    linestyle="--",
                    label="Synthetic",
                )
                ax.fill_between(
                    unique_vals,
                    real_density,
                    step="mid",
                    color=real_color,
                    alpha=0.12,
                )
                ax.fill_between(
                    unique_vals,
                    syn_density,
                    step="mid",
                    color=syn_color,
                    alpha=0.12,
                )
                ax.scatter(unique_vals, real_density, color=real_color, s=16, alpha=0.9, zorder=3)
                ax.scatter(unique_vals, syn_density, color=syn_color, s=16, alpha=0.9, marker="s", zorder=3)
            else:
                bins = min(60, max(20, int(np.sqrt(len(combined_vals)))))
                hist_range = (combined_vals.min(), combined_vals.max())
                if np.isclose(hist_range[0], hist_range[1]):
                    hist_range = (hist_range[0] - 0.5, hist_range[1] + 0.5)
                real_hist, bin_edges = np.histogram(real_vals, bins=bins, range=hist_range, density=True)
                syn_hist, _ = np.histogram(syn_vals, bins=bin_edges, density=True)
                centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                ax.plot(
                    centers,
                    real_hist,
                    color=real_color,
                    linewidth=2.8,
                    alpha=0.88,
                    linestyle="-",
                    label="Real",
                )
                ax.plot(
                    centers,
                    syn_hist,
                    color=syn_color,
                    linewidth=2.8,
                    alpha=0.88,
                    linestyle="--",
                    label="Synthetic",
                )
                ax.fill_between(centers, real_hist, color=real_color, alpha=0.12)
                ax.fill_between(centers, syn_hist, color=syn_color, alpha=0.12)

            ax.set_title(feat, fontsize=11)
            ax.grid(alpha=0.18, linewidth=0.6)
            ax.tick_params(labelsize=9)

        for ax in axes[n_features:]:
            ax.set_axis_off()

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=11)
        fig.suptitle(f"Feature Distribution Comparison: {attack_name}", fontsize=16, y=1.02)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 针对每个稀有类
    for attack, syn_df in synthetic_data_dict.items():
        attack_label = LABEL_MAP[attack]
        real_mask = y_train == attack_label
        real_df = X_train[real_mask].copy()
        # 保留 focus_features
        real_df = real_df[focus_features]
        syn_df_feat = syn_df[focus_features].copy()

        # 描述统计对比
        desc_real = real_df.describe().T
        desc_syn = syn_df_feat.describe().T
        comparison = pd.concat(
            [desc_real.add_suffix("_real"), desc_syn.add_suffix("_syn")], axis=1
        )
        comparison.to_csv(quality_dir / f"{attack}_describe_comparison.csv")

        # KS 检验（针对每个数值特征）
        ks_results = {}
        for feat in focus_features:
            real_vals = real_df[feat].dropna()
            syn_vals = syn_df_feat[feat].dropna()
            if len(real_vals) > 0 and len(syn_vals) > 0:
                stat, p = ks_2samp(real_vals, syn_vals)
                ks_results[feat] = {"KS_statistic": stat, "p_value": p}
        ks_df = pd.DataFrame(ks_results).T
        ks_df.to_csv(quality_dir / f"{attack}_ks_test.csv")

        plot_feature_distribution_comparison(
            attack_name=attack,
            real_df=real_df,
            syn_df_feat=syn_df_feat,
            features=focus_features,
            save_path=quality_dir / f"feature_distribution_{attack}.png",
        )

        # PCA 可视化：合并真实与合成，标注来源
        combined = pd.concat([real_df, syn_df_feat], keys=["Real", "Synthetic"], names=["Source"])
        combined = combined.reset_index(level=0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(combined[focus_features])
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(scaled)
        combined["PC1"] = pca_result[:, 0]
        combined["PC2"] = pca_result[:, 1]

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=combined, x="PC1", y="PC2", hue="Source", alpha=0.6, palette="dark")
        plt.title(f"PCA: Real vs Synthetic ({attack})")
        plt.tight_layout()
        plt.savefig(quality_dir / f"pca_{attack}.png", dpi=300)
        plt.close()
        print(f"  {attack}: PCA 图、分布对比图及指标已保存")


def build_same_model_comparison_table(
    original_summary_df: pd.DataFrame,
    balanced_summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构建同一模型在平衡前后训练结果的宽表与长表。"""
    metric_specs = [
        ("accuracy", "Accuracy", True),
        ("macro_precision_from_report", "Macro Precision", True),
        ("macro_recall_from_report", "Macro Recall", True),
        ("macro_f1", "Macro F1", True),
        ("weighted_f1", "Weighted F1", True),
        ("attack_macro_recall", "Attack Macro Recall", True),
        ("attack_macro_f1", "Attack Macro F1", True),
        ("macro_fpr", "Macro FPR", False),
        ("macro_fnr", "Macro FNR", False),
        ("normal_false_alarm_rate", "Normal False Alarm Rate", False),
    ]

    merged_df = pd.merge(
        original_summary_df.add_suffix("_original"),
        balanced_summary_df.add_suffix("_balanced"),
        left_on="model_original",
        right_on="model_balanced",
        how="inner",
    )
    merged_df = merged_df.rename(columns={"model_original": "model"}).drop(columns=["model_balanced"])

    long_records: list[dict[str, str | float | bool]] = []
    for metric_name, metric_label, higher_is_better in metric_specs:
        original_col = f"{metric_name}_original"
        balanced_col = f"{metric_name}_balanced"
        delta_col = f"{metric_name}_delta"
        benefit_col = f"{metric_name}_benefit_delta"
        improved_col = f"{metric_name}_improved"

        merged_df[delta_col] = merged_df[balanced_col] - merged_df[original_col]
        if higher_is_better:
            merged_df[benefit_col] = merged_df[delta_col]
        else:
            merged_df[benefit_col] = merged_df[original_col] - merged_df[balanced_col]
        merged_df[improved_col] = merged_df[benefit_col] > 1e-12

        for row in merged_df.itertuples(index=False):
            original_value = float(getattr(row, original_col))
            balanced_value = float(getattr(row, balanced_col))
            delta_value = float(getattr(row, delta_col))
            benefit_value = float(getattr(row, benefit_col))
            long_records.append(
                {
                    "model": str(row.model),
                    "metric": metric_name,
                    "metric_label": metric_label,
                    "stage": "Original Train",
                    "value": original_value,
                    "delta": delta_value,
                    "benefit_delta": benefit_value,
                    "higher_is_better": higher_is_better,
                    "improved": bool(benefit_value > 1e-12),
                }
            )
            long_records.append(
                {
                    "model": str(row.model),
                    "metric": metric_name,
                    "metric_label": metric_label,
                    "stage": "Balanced Train",
                    "value": balanced_value,
                    "delta": delta_value,
                    "benefit_delta": benefit_value,
                    "higher_is_better": higher_is_better,
                    "improved": bool(benefit_value > 1e-12),
                }
            )

    long_df = pd.DataFrame(long_records)
    return merged_df, long_df


def plot_same_model_comparison(long_df: pd.DataFrame, output_dir: Path) -> None:
    """绘制平衡前后同一模型的关键指标对比图与提升热力图。"""
    plot_dir = output_dir / "same_model_comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)

    selected_metric_labels = [
        "Accuracy",
        "Macro Recall",
        "Macro F1",
        "Macro FPR",
        "Normal False Alarm Rate",
    ]
    plot_df = long_df[long_df["metric_label"].isin(selected_metric_labels)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for idx, metric_label in enumerate(selected_metric_labels):
        ax = axes[idx]
        metric_df = plot_df[plot_df["metric_label"] == metric_label]
        sns.barplot(
            data=metric_df,
            x="model",
            y="value",
            hue="stage",
            palette=["#4c72b0", "#dd8452"],
            ax=ax,
        )
        ax.set_title(metric_label)
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
        if idx == 0:
            ax.legend(title="Training Set")
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    axes[-1].set_axis_off()
    fig.suptitle("Same Model Performance: Original vs Balanced Training Data", fontsize=16)
    fig.savefig(plot_dir / "same_model_before_after_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    heatmap_df = (
        long_df[["model", "metric_label", "benefit_delta"]]
        .drop_duplicates()
        .pivot(index="model", columns="metric_label", values="benefit_delta")
    )
    metric_order = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1",
        "Attack Macro Recall",
        "Attack Macro F1",
        "Macro FPR",
        "Macro FNR",
        "Normal False Alarm Rate",
    ]
    heatmap_df = heatmap_df.reindex(index=MODEL_ORDER, columns=metric_order)

    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_df, annot=True, fmt=".4f", cmap="RdYlGn", center=0.0)
    plt.title("Balanced Training Improvement Heatmap (Positive Means Better)")
    plt.xlabel("Metric")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(plot_dir / "same_model_improvement_heatmap.png", dpi=300)
    plt.close()


def export_lightgbm_detailed_comparison(
    original_summary_df: pd.DataFrame,
    balanced_summary_df: pd.DataFrame,
    original_per_class_df: pd.DataFrame,
    balanced_per_class_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """导出 LightGBM 平衡前后总体指标与逐类指标对比。"""
    comparison_dir = output_dir / "same_model_comparison" / "lightgbm_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    overall_metric_specs = [
        ("accuracy", "Accuracy", True),
        ("macro_precision_from_report", "Macro Precision", True),
        ("macro_recall_from_report", "Macro Recall", True),
        ("macro_f1", "Macro F1", True),
        ("weighted_f1", "Weighted F1", True),
        ("attack_macro_recall", "Attack Macro Recall", True),
        ("attack_macro_f1", "Attack Macro F1", True),
        ("macro_fpr", "Macro FPR", False),
        ("macro_fnr", "Macro FNR", False),
        ("normal_false_alarm_rate", "Normal False Alarm Rate", False),
    ]
    per_class_metric_specs = [
        ("precision", "Precision", True),
        ("recall", "Recall", True),
        ("f1", "F1", True),
        ("fpr", "FPR", False),
        ("fnr", "FNR", False),
    ]

    original_lightgbm = original_summary_df.loc[original_summary_df["model"] == "LightGBM"]
    balanced_lightgbm = balanced_summary_df.loc[balanced_summary_df["model"] == "LightGBM"]
    if original_lightgbm.empty or balanced_lightgbm.empty:
        print("警告: 未找到 LightGBM 汇总指标，跳过 LightGBM 专项对比导出。")
        return

    overall_records: list[dict[str, str | float | bool]] = []
    for metric_name, metric_label, higher_is_better in overall_metric_specs:
        original_value = float(original_lightgbm.iloc[0][metric_name])
        balanced_value = float(balanced_lightgbm.iloc[0][metric_name])
        delta = balanced_value - original_value
        benefit_delta = delta if higher_is_better else original_value - balanced_value
        overall_records.append(
            {
                "metric": metric_name,
                "metric_label": metric_label,
                "original_value": original_value,
                "balanced_value": balanced_value,
                "delta": delta,
                "benefit_delta": benefit_delta,
                "higher_is_better": higher_is_better,
                "improved": bool(benefit_delta > 1e-12),
            }
        )
    overall_df = pd.DataFrame(overall_records)
    overall_df.to_csv(comparison_dir / "lightgbm_overall_before_after.csv", index=False)

    original_per_class_lightgbm = original_per_class_df.loc[original_per_class_df["model"] == "LightGBM"].copy()
    balanced_per_class_lightgbm = balanced_per_class_df.loc[balanced_per_class_df["model"] == "LightGBM"].copy()
    if original_per_class_lightgbm.empty or balanced_per_class_lightgbm.empty:
        print("警告: 未找到 LightGBM 逐类指标，跳过 LightGBM 逐类对比导出。")
        return

    merged_per_class = pd.merge(
        original_per_class_lightgbm.add_suffix("_original"),
        balanced_per_class_lightgbm.add_suffix("_balanced"),
        left_on="class_name_original",
        right_on="class_name_balanced",
        how="inner",
    )
    merged_per_class = merged_per_class.rename(columns={"class_name_original": "class_name"}).drop(
        columns=["class_name_balanced", "model_original", "model_balanced"]
    )

    per_class_records: list[dict[str, str | float | bool]] = []
    for metric_name, metric_label, higher_is_better in per_class_metric_specs:
        original_col = f"{metric_name}_original"
        balanced_col = f"{metric_name}_balanced"
        for row in merged_per_class.itertuples(index=False):
            original_value = float(getattr(row, original_col))
            balanced_value = float(getattr(row, balanced_col))
            delta = balanced_value - original_value
            benefit_delta = delta if higher_is_better else original_value - balanced_value
            per_class_records.append(
                {
                    "class_name": str(row.class_name),
                    "metric": metric_name,
                    "metric_label": metric_label,
                    "support_original": int(getattr(row, "support_original")),
                    "support_balanced": int(getattr(row, "support_balanced")),
                    "original_value": original_value,
                    "balanced_value": balanced_value,
                    "delta": delta,
                    "benefit_delta": benefit_delta,
                    "higher_is_better": higher_is_better,
                    "improved": bool(benefit_delta > 1e-12),
                }
            )

    per_class_df = pd.DataFrame(per_class_records)
    per_class_df.to_csv(comparison_dir / "lightgbm_per_class_before_after.csv", index=False)
    per_class_df.loc[per_class_df["class_name"] != "Normal"].to_csv(
        comparison_dir / "lightgbm_attack_per_class_before_after.csv", index=False
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=overall_df, x="metric_label", y="benefit_delta", palette="crest")
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    plt.title("LightGBM Overall Metric Improvement After Balancing")
    plt.xlabel("Metric")
    plt.ylabel("Improvement Value")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(comparison_dir / "lightgbm_overall_improvement.png", dpi=300)
    plt.close()

    heatmap_df = per_class_df.pivot(index="class_name", columns="metric_label", values="benefit_delta")
    heatmap_df = heatmap_df.reindex(index=CLASS_NAMES, columns=[spec[1] for spec in per_class_metric_specs])
    plt.figure(figsize=(10, 4.8))
    sns.heatmap(heatmap_df, annot=True, fmt=".4f", cmap="RdYlGn", center=0.0)
    plt.title("LightGBM Per-class Improvement Heatmap")
    plt.xlabel("Metric")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(comparison_dir / "lightgbm_per_class_improvement_heatmap.png", dpi=300)
    plt.close()


def run_balanced_evaluation(
    X_train_original: pd.DataFrame,
    y_train_original: pd.Series,
    X_train_balanced: pd.DataFrame,
    y_train_balanced: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: List[str],
    random_state: int,
    output_dir: Path,
) -> None:
    """用平衡训练集训练模型并在真实测试集评估，与原始不平衡结果对比"""
    print("\n========== 训练平衡模型并评估 ==========")
    same_model_dir = output_dir / "same_model_comparison"
    original_dir = same_model_dir / "original_train_models"
    balanced_dir = same_model_dir / "balanced_train_models"
    original_dir.mkdir(parents=True, exist_ok=True)
    balanced_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "balanced_models").mkdir(parents=True, exist_ok=True)
    (output_dir / "balanced_plots").mkdir(parents=True, exist_ok=True)

    original_models = build_balanced_eval_models(random_state)
    original_summary_df, original_per_class_df = baseline.train_and_evaluate_models(
        models=original_models,
        X_train=X_train_original,
        y_train=y_train_original,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        output_dir=original_dir,
    )
    original_summary_df.to_csv(same_model_dir / "original_train_model_summary.csv", index=False)
    original_per_class_df.to_csv(same_model_dir / "original_train_per_class_metrics.csv", index=False)

    models = build_balanced_eval_models(random_state)
    summary_df, per_class_df = baseline.train_and_evaluate_models(
        models=models,
        X_train=X_train_balanced,
        y_train=y_train_balanced,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        output_dir=balanced_dir,
    )
    # 保存平衡后的结果
    summary_df.to_csv(output_dir / "balanced_model_summary.csv", index=False)
    per_class_df.to_csv(output_dir / "balanced_per_class_metrics.csv", index=False)
    summary_df.to_csv(same_model_dir / "balanced_train_model_summary.csv", index=False)
    per_class_df.to_csv(same_model_dir / "balanced_train_per_class_metrics.csv", index=False)

    comparison_wide_df, comparison_long_df = build_same_model_comparison_table(
        original_summary_df=original_summary_df,
        balanced_summary_df=summary_df,
    )
    comparison_wide_df.to_csv(same_model_dir / "same_model_before_after_summary.csv", index=False)
    comparison_long_df.to_csv(same_model_dir / "same_model_before_after_long.csv", index=False)
    plot_same_model_comparison(comparison_long_df, output_dir)
    export_lightgbm_detailed_comparison(
        original_summary_df=original_summary_df,
        balanced_summary_df=summary_df,
        original_per_class_df=original_per_class_df,
        balanced_per_class_df=per_class_df,
        output_dir=output_dir,
    )

    improved_counts = comparison_long_df[
        ["model", "metric_label", "benefit_delta"]
    ].drop_duplicates()
    improved_counts = improved_counts.groupby("model")["benefit_delta"].apply(lambda values: int((values > 1e-12).sum()))
    print("同一模型在平衡前后关键指标提升个数:")
    for model_name, improved_count in improved_counts.items():
        print(f"  {model_name}: {improved_count} / 10 metrics improved")

    baseline.plot_summary_charts(summary_df, output_dir / "balanced_plots")
    baseline.plot_per_class_heatmaps(per_class_df, output_dir / "balanced_plots")

    # 若存在原始不平衡结果，加载进行对比
    original_summary_path = output_dir.parent / "wsn_ds_baseline" / "model_summary.csv"
    if original_summary_path.exists():
        original_df = pd.read_csv(original_summary_path)
        compare_df = pd.merge(
            original_df.add_suffix("_original"), summary_df.add_suffix("_balanced"),
            left_on="model_original", right_on="model_balanced",
            how="inner"
        )

        shared_metric_columns = [
            column for column in summary_df.columns
            if column != "model" and f"{column}_original" in compare_df.columns and f"{column}_balanced" in compare_df.columns
        ]
        for column in shared_metric_columns:
            original_col = f"{column}_original"
            balanced_col = f"{column}_balanced"
            delta_col = f"{column}_delta"
            change_pct_col = f"{column}_change_pct"
            trend_col = f"{column}_trend"

            compare_df[delta_col] = compare_df[balanced_col] - compare_df[original_col]
            original_abs = compare_df[original_col].abs()
            compare_df[change_pct_col] = np.where(
                original_abs > 1e-12,
                compare_df[delta_col] / original_abs * 100.0,
                np.nan,
            )
            compare_df[trend_col] = np.select(
                [compare_df[delta_col] > 1e-12, compare_df[delta_col] < -1e-12],
                ["increased", "decreased"],
                default="unchanged",
            )

        compare_df.to_csv(output_dir / "comparison_balanced_vs_original.csv", index=False)
        print("已保存与原始 baseline 的对比表。")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print("加载数据...")
    df = baseline.load_dataset(args.data)
    X, y = baseline.prepare_features_and_target(df)

    # 2. 确定要用的特征
    focus_features = prepare_focus_features(args, X)
    print(f"使用特征 ({len(focus_features)}): {focus_features}")
    X = X[focus_features]  # 限制特征集

    # 3. 划分训练/测试集（保持测试集真实）
    print(f"划分策略: {args.split_strategy}")
    X_train, X_test, y_train, y_test, split_meta = baseline.split_dataset(
        df=df,
        X=X,
        y=y,
        test_size=args.test_size,
        random_state=args.random_state,
        split_strategy=args.split_strategy,
        group_column=args.group_column,
        time_column=args.time_column,
    )

    # 4. 生成倍率设置（按每个少数类原始样本数乘倍率）
    gen_multiplier = max(0.0, float(args.gen_multiplier))
    print(f"生成倍率模式：n_gen = real_count * {gen_multiplier}")

    # 5. CTGAN 生成并平衡训练集
    print("\n开始 CTGAN 生成...")
    X_balanced, y_balanced, synthetic_dict = ctgan_balance_training_data(
        X_train=X_train,
        y_train=y_train,
        focus_features=focus_features,
        target_column=TARGET_COLUMN,
        gen_multiplier=gen_multiplier,
        rare_attacks=RARE_ATTACKS,
        ctgan_epochs=args.ctgan_epochs,
        ctgan_batch_size=args.ctgan_batch_size,
        use_gpu=bool(args.gpu),
        ctgan_verbose=bool(args.ctgan_verbose),
        reuse_models=bool(args.reuse_models),
        models_dir=args.models_dir,
        random_state=args.random_state,
        output_dir=output_dir,
    )
    print(f"平衡后训练集大小: {len(X_balanced)}")

    # 6. 质量验证
    if args.quality_check and len(synthetic_dict) > 0:
        quality_verification(
            X_train=X_train,
            y_train=y_train,
            synthetic_data_dict=synthetic_dict,
            focus_features=focus_features,
            target_column=TARGET_COLUMN,
            output_dir=output_dir,
        )

    # 7. 训练模型并评估（可选）
    if args.train_models:
        run_balanced_evaluation(
            X_train_original=X_train,
            y_train_original=y_train,
            X_train_balanced=X_balanced,
            y_train_balanced=y_balanced,
            X_test=X_test,
            y_test=y_test,
            class_names=CLASS_NAMES,
            random_state=args.random_state,
            output_dir=output_dir,
        )

    # 8. 保存合成数据和元信息
    metadata = {
        "generation_mode": "per_class_multiplier",
        "gen_multiplier": float(gen_multiplier),
        "normal_class_multiplier": float(NORMAL_CLASS_MULTIPLIER),
        "ctgan_epochs": int(args.ctgan_epochs),
        "ctgan_verbose": bool(args.ctgan_verbose),
        "models_dir": str(args.models_dir),
        "ctgan_use_gpu": bool(args.gpu),
        "reuse_models": bool(args.reuse_models),
        "focus_features": [str(feature) for feature in focus_features],
        "split_strategy": str(args.split_strategy),
        "random_state": int(args.random_state),
    }
    with open(output_dir / "ctgan_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 保存平衡后的完整训练集（可选）
    balanced_full = X_balanced.copy()
    balanced_full["label"] = y_balanced.values
    balanced_full.to_csv(output_dir / "balanced_train.csv", index=False)
    print(f"\n所有结果已保存至: {output_dir.resolve()}")


if __name__ == "__main__":
    main()