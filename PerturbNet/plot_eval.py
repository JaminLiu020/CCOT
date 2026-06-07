"""
plot_perturbnet_eval.py
=======================
离线绘图脚本：加载 notebooks/results/perturbnet_condot_sciplex/sciplex_comp2condot_rdkit2d/eval/data
中的数据，分别为不同阶段（ood/test）、不同 subset（full/deg50）、不同药物生成：
  - UMAP 图：保存到 eval/plot/umap/<stage_subset>/<drug>_combined_umap.pdf
  - 热力差异图：保存到 eval/plot/heatmap/<stage_subset>/<drug>_diff_heatmap.pdf

画法参数与 condot/scripts/plot.py + heatmap.py 完全对齐。

修复说明（v2）：
  [Fix1] 热力差异图颜色条固定：先遍历所有待画药物计算 global_diff_max，
         所有图共用同一固定 vmin/vmax，颜色条范围不再随药物变化。
  [Fix2] UMAP 聚簇一致性：把所有待画药物的 real data 合并后 fit 一次 UMAP，
         各药物再 transform，确保跨图坐标空间统一（簇位置不随药物而翻转/旋转）。

用法（在 PerturbNet 仓库根目录下执行）：
    conda run -n rapids_singlecell python notebooks/plot_perturbnet_eval.py

可配置项见下方 CONFIG 区域。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
import seaborn as sns
from pathlib import Path
from umap import UMAP


# ============================================================
# CONFIG
# ============================================================
_SCRIPT_DIR = Path(__file__).resolve().parent        # notebooks/
_REPO_ROOT  = _SCRIPT_DIR.parent                     # PerturbNet/

DATA_DIR = _REPO_ROOT / "notebooks" / "results" / "perturbnet_condot_sciplex" / \
           "sciplex_comp2condot_rdkit2d" / "eval" / "data"

PLOT_DIR = _REPO_ROOT / "notebooks" / "results" / "perturbnet_condot_sciplex" / \
           "sciplex_comp2condot_rdkit2d" / "eval" / "plot"

# 只处理 ood_deg50
STAGES  = ["ood"]
SUBSETS = ["deg50"]

# 药物过滤：None = 不过滤（画全部）；list[str] = 名字含任意一关键词才画
UMAP_DRUG_FILTER = ["TAK-901"]      # 只画名字含 TAK-901 的药物的 UMAP
HEAT_DRUG_FILTER = ["Hesperadin"]   # 只画名字含 Hesperadin 的药物的热力图

# 细胞类型（顺序决定颜色：蓝/红/橙，与 condot/scripts/plot.py 完全一致）
CELL_TYPES       = ["A549", "K562", "MCF7"]
CELL_TYPE_COLORS = ["blue", "red", "orange"]

# UMAP 超参数（与 condot/scripts/plot.py 完全一致）
UMAP_PARAMS = dict(
    n_components=2,
    random_state=42,
    spread=2.0,
    min_dist=0.4,
    n_neighbors=15,
    n_jobs=1,
)

# 字体设置（与 condot/scripts/heatmap.py 完全一致）
def _pick_english_serif_font():
    candidates = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Serif"

def _pick_chinese_font():
    # 与 condot/scripts/heatmap.py 完全一致，宋体优先
    candidates = ["SimSun", "Noto Serif CJK SC", "AR PL UMing CN", "WenQuanYi Zen Hei"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None

_english_font_name = _pick_english_serif_font()
plt.rcParams["font.family"]        = _english_font_name
plt.rcParams["font.serif"]         = [_english_font_name]
plt.rcParams["axes.unicode_minus"] = False

_chinese_font_name  = _pick_chinese_font()
_chinese_font_props = (
    font_manager.FontProperties(family=_chinese_font_name)
    if _chinese_font_name
    else None
)


# ============================================================
# 辅助：药物名过滤
# ============================================================
def _drug_matches(drug_name: str, filter_keywords) -> bool:
    """filter_keywords=None 则全部通过；否则名字中含任意一个关键词就通过。"""
    if filter_keywords is None:
        return True
    return any(kw in drug_name for kw in filter_keywords)


# ============================================================
# 数据加载
# ============================================================
def load_stage_subset_data(data_dir: Path, stage: str, subset: str):
    """
    加载 data/<stage>_<subset>/ 下所有药物的 .npz 文件。

    返回：
        drug_data: dict[drug_name -> {
            'real'              : np.ndarray  (n_cells, n_genes)
            'pred'              : np.ndarray
            'source'            : np.ndarray
            'labels'            : np.ndarray  整数 0/1/2 对应 CELL_TYPES
            'cell_types_present': list[str]
        }]
    """
    stage_dir = data_dir / f"{stage}_{subset}"
    if not stage_dir.exists():
        return {}

    cell_type_to_label = {ct: i for i, ct in enumerate(CELL_TYPES)}
    drug_data = {}

    for drug_dir in sorted(stage_dir.iterdir()):
        if not drug_dir.is_dir():
            continue
        drug_name = drug_dir.name

        real_list, pred_list, src_list, label_list, ct_present = [], [], [], [], []

        for ct in CELL_TYPES:
            cell_file = drug_dir / f"{ct}.npz"
            if not cell_file.exists():
                continue
            try:
                npz  = np.load(str(cell_file))
                real = npz["real"].astype(np.float32)
                pred = npz["pred"].astype(np.float32)
                src  = npz["source"].astype(np.float32)
                lbl  = np.full(real.shape[0], cell_type_to_label[ct], dtype=np.int32)
                real_list.append(real)
                pred_list.append(pred)
                src_list.append(src)
                label_list.append(lbl)
                ct_present.append(ct)
            except Exception as e:
                print(f"  [WARN] Failed to load {cell_file}: {e}")

        if real_list:
            drug_data[drug_name] = {
                "real":               np.vstack(real_list),
                "pred":               np.vstack(pred_list),
                "source":             np.vstack(src_list),
                "labels":             np.concatenate(label_list),
                "cell_types_present": ct_present,
            }

    return drug_data


# ============================================================
# UMAP 绘图（对齐 condot/scripts/plot.py）
#
# [Fix2] shared_reducer 由外部传入，已在所有待画药物的 real 数据上
#        统一 fit，保证跨图坐标空间一致（聚簇位置不随药物变化）。
# ============================================================
def plot_combined_umap(real, real_labels, pred, pred_labels,
                       drug_name, savepath: Path, shared_reducer):
    """
    对齐 condot plot.py 的 plot_combined_scatter：
    - 使用共享 reducer（已在所有待画药物 real 数据上 fit）做 transform
    - pred（transport）实心 alpha=0.8，real（target）透明 alpha=0.1
    - 颜色：蓝/红/橙 对应 A549/K562/MCF7
    - 图例 fontsize=18，marker size=16
    - 无坐标轴刻度
    """
    cmap = ListedColormap(CELL_TYPE_COLORS)

    real_2d = shared_reducer.transform(real)
    pred_2d = shared_reducer.transform(pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    # pred (transport)：实心，对齐 condot 绘制顺序（先 transport 后 target）
    ax.scatter(pred_2d[:, 0], pred_2d[:, 1],
               c=pred_labels, cmap=cmap, s=30, alpha=0.8,
               edgecolor="k", linewidths=0.3)

    # real (target)：透明，叠在上层
    ax.scatter(real_2d[:, 0], real_2d[:, 1],
               c=real_labels, cmap=cmap, s=30, alpha=0.1,
               edgecolor="k", linewidths=0.3)

    ax.set_xticks([])
    ax.set_yticks([])

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=CELL_TYPE_COLORS[i], markersize=16,
                   label=CELL_TYPES[i])
        for i in range(len(CELL_TYPES))
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=18)

    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(savepath), format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  [UMAP]    saved -> {savepath.relative_to(_REPO_ROOT)}")


# ============================================================
# 热力差异图（对齐 condot/scripts/heatmap.py）
#
# [Fix1] vmin/vmax 由外部传入全局统一值，跨所有药物固定颜色条范围。
# ============================================================
def plot_diff_heatmap(real, pred, labels, drug_name,
                      savepath: Path, global_vmin: float, global_vmax: float):
    """
    对齐 condot heatmap.py 的 create_heatmaps（差异热力图部分）：
    - 每细胞类型计算 mean(pred) - mean(real) -> diff matrix (n_ct, n_genes)
    - cmap="RdBu_r"，全局统一 vmin/vmax（symmetric），center=0
    - x/y 轴标签中文（宋体），font size 26/28，无 x 轴刻度，无顶部标题
    保存为 PDF。
    """
    label_to_ct = {i: ct for i, ct in enumerate(CELL_TYPES)}
    present_labels = sorted(np.unique(labels))
    ct_names = [label_to_ct[l] for l in present_labels]

    real_means = np.array([real[labels == l].mean(axis=0) for l in present_labels])
    pred_means = np.array([pred[labels == l].mean(axis=0) for l in present_labels])
    diff = pred_means - real_means

    plt.rcParams.update({"font.size": 26})

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        diff,
        cmap="RdBu_r",
        vmin=global_vmin, vmax=global_vmax, center=0,
        xticklabels=False,
        yticklabels=ct_names,
        ax=ax,
    )
    # 中文轴标签，字体参照 condot/scripts/heatmap.py
    ax.set_xlabel("基因", fontsize=28, fontproperties=_chinese_font_props)
    ax.set_ylabel("细胞类型", fontsize=28, fontproperties=_chinese_font_props)
    # 不设置标题

    plt.tight_layout()
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(savepath), format="pdf", bbox_inches="tight")
    plt.close()
    plt.rcParams.update({"font.size": 10})   # 还原，避免污染后续图
    print(f"  [HEATMAP] saved -> {savepath.relative_to(_REPO_ROOT)}")


# ============================================================
# 主流程
# ============================================================
def main():
    print(f"Data  dir  : {DATA_DIR}")
    print(f"Output dir : {PLOT_DIR}")
    print(f"UMAP  filter : {UMAP_DRUG_FILTER}")
    print(f"Heatmap filter: {HEAT_DRUG_FILTER}")
    print()

    umap_root    = PLOT_DIR / "umap"
    heatmap_root = PLOT_DIR / "heatmap"

    for stage in STAGES:
        for subset in SUBSETS:
            tag = f"{stage}_{subset}"
            print(f"=== Processing stage={stage}, subset={subset} ===")

            all_drug_data = load_stage_subset_data(DATA_DIR, stage, subset)
            if not all_drug_data:
                print(f"  [SKIP] No data found under {DATA_DIR / tag}")
                continue

            # 过滤出各自需要的药物子集
            umap_drugs = {
                k: v for k, v in all_drug_data.items()
                if _drug_matches(k, UMAP_DRUG_FILTER)
            }
            heat_drugs = {
                k: v for k, v in all_drug_data.items()
                if _drug_matches(k, HEAT_DRUG_FILTER)
            }

            # ─────────────────────────────────────────────────────────
            # [Fix2] UMAP：合并所有待画药物的 real data 统一 fit，
            #         保证跨图坐标空间一致
            # ─────────────────────────────────────────────────────────
            if umap_drugs:
                drug_names_umap = list(umap_drugs.keys())
                print(f"\n  [UMAP] Drugs to plot: {drug_names_umap}")
                print(f"  [UMAP] Fitting shared reducer on combined real data ...")

                all_real_for_fit = np.vstack([b["real"] for b in umap_drugs.values()])
                shared_reducer = UMAP(**UMAP_PARAMS)
                shared_reducer.fit(all_real_for_fit)
                print(f"  [UMAP] Fit done: {all_real_for_fit.shape[0]} cells x "
                      f"{all_real_for_fit.shape[1]} genes\n")

                for drug_name, bundle in umap_drugs.items():
                    real   = bundle["real"]
                    pred   = bundle["pred"]
                    labels = bundle["labels"]
                    print(f"  Drug (UMAP): {drug_name} | "
                          f"real={real.shape[0]}, pred={pred.shape[0]}, "
                          f"genes={real.shape[1]}, "
                          f"cell_types={bundle['cell_types_present']}")
                    umap_path = umap_root / tag / f"{drug_name}_combined_umap.pdf"
                    try:
                        plot_combined_umap(
                            real=real, real_labels=labels,
                            pred=pred, pred_labels=labels,
                            drug_name=drug_name,
                            savepath=umap_path,
                            shared_reducer=shared_reducer,
                        )
                    except Exception as e:
                        print(f"  [ERROR] UMAP failed for {drug_name}: {e}")
            else:
                print("  [UMAP] No drugs matched the filter, skipping.")

            # ─────────────────────────────────────────────────────────
            # [Fix1] Heatmap：先扫描所有待画药物计算 global_diff_max，
            #         再统一绘图，颜色条范围固定
            # ─────────────────────────────────────────────────────────
            if heat_drugs:
                drug_names_heat = list(heat_drugs.keys())
                print(f"\n  [HEATMAP] Drugs to plot: {drug_names_heat}")
                print(f"  [HEATMAP] Computing global diff_max across all matched drugs ...")

                global_diff_max = 0.0
                for drug_name, bundle in heat_drugs.items():
                    real   = bundle["real"]
                    pred   = bundle["pred"]
                    labels = bundle["labels"]
                    present_labels = sorted(np.unique(labels))
                    real_means = np.array([real[labels == l].mean(axis=0) for l in present_labels])
                    pred_means = np.array([pred[labels == l].mean(axis=0) for l in present_labels])
                    diff = pred_means - real_means
                    drug_max = float(max(abs(diff.min()), abs(diff.max())))
                    print(f"    {drug_name}: local diff_abs_max = {drug_max:.6f}")
                    global_diff_max = max(global_diff_max, drug_max)

                # 固定颜色条范围为 -0.5 ~ 0.5（用户指定，覆盖自动计算值）
                global_vmin = -0.5
                global_vmax =  0.5
                print(f"  [HEATMAP] Fixed colorbar (user-specified): [{global_vmin}, {global_vmax}]\n")

                for drug_name, bundle in heat_drugs.items():
                    real   = bundle["real"]
                    pred   = bundle["pred"]
                    labels = bundle["labels"]
                    print(f"  Drug (Heatmap): {drug_name} | "
                          f"real={real.shape[0]}, genes={real.shape[1]}, "
                          f"cell_types={bundle['cell_types_present']}")
                    heatmap_path = heatmap_root / tag / f"{drug_name}_diff_heatmap.pdf"
                    try:
                        plot_diff_heatmap(
                            real=real, pred=pred,
                            labels=labels,
                            drug_name=drug_name,
                            savepath=heatmap_path,
                            global_vmin=global_vmin,
                            global_vmax=global_vmax,
                        )
                    except Exception as e:
                        print(f"  [ERROR] Heatmap failed for {drug_name}: {e}")
            else:
                print("  [HEATMAP] No drugs matched the filter, skipping.")

            print()

    print("All plots done.")


if __name__ == "__main__":
    main()
