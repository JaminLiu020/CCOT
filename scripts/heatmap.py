import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import pandas as pd
from pathlib import Path

# Configuration
base_dir_candidates = [
    "/home/jamin/condot/old_result/nine_drugs/977genes/drug_pert/visualization_exp",
    "/home/jamin/condot/results/nine_drugs/977genes/drug_pert/visualization_exp",
]
base_dir = next((path for path in base_dir_candidates if os.path.exists(path)), base_dir_candidates[0])
# models = ["CCOT", "CELLOT", "CondOT_cell_type", "pretrained_chemCPA", "CondOT_dose", "chemCPA"]
models = ["CCOT", "CELLOT", "CondOT_cell_type", "pretrained_chemCPA"]
cell_types = ["A549", "K562", "MCF7"]
test_sets = ["ood_DEGs"]  # Focus on out-of-distribution
drugs_to_visualize = ["Hesperadin", "TAK-901", "Dacinostat", "Givinostat", "Belinostat", "Quisinostat", "Alvespimycin", "Tanespimycin", "Flavopiridol"]  # Selected representative drugs

# Create output directories
output_dir = Path("visualization_results")
output_dir.mkdir(exist_ok=True)
(output_dir / "heatmap").mkdir(exist_ok=True)


def _pick_chinese_font():
    candidates = ["SimSun", "Noto Serif CJK SC", "AR PL UMing CN", "WenQuanYi Zen Hei"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def _pick_english_serif_font():
    candidates = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Serif"


_english_font_name = _pick_english_serif_font()
plt.rcParams["font.family"] = _english_font_name
plt.rcParams["font.serif"] = [_english_font_name]
plt.rcParams["axes.unicode_minus"] = False
_chinese_font_name = _pick_chinese_font()
_chinese_font_props = (
    font_manager.FontProperties(family=_chinese_font_name)
    if _chinese_font_name
    else None
)


def load_data_for_drug(model, test_set, drug):
    """Load real and predicted data for all cell types for a specific drug"""
    all_real = []
    all_pred = []
    cell_type_labels = []

    for cell_type in cell_types:
        file_path = os.path.join(base_dir, model, test_set, drug, f"{cell_type}.npz")
        if os.path.exists(file_path):
            data = np.load(file_path)
            real = data['real']
            pred = data['pred']

            # Append data and labels
            all_real.append(real)
            all_pred.append(pred)
            cell_type_labels.extend([cell_type] * len(real))

    if all_real:
        return np.vstack(all_real), np.vstack(all_pred), cell_type_labels
    return None, None, None


def create_violin_plots(top_n_genes=5):
    """Create violin plots for top N genes by variance"""
    for drug in drugs_to_visualize:
        for model in models:
            real_data, pred_data, cell_labels = load_data_for_drug(model, test_sets[0], drug)

            if real_data is None:
                print(f"No data found for {model} on {drug}")
                continue

            # Find genes with highest variance
            gene_var = np.var(real_data, axis=0)
            top_gene_indices = np.argsort(gene_var)[-top_n_genes:]

            # Prepare data for plotting
            plt.figure(figsize=(15, 10))

            for i, gene_idx in enumerate(top_gene_indices):
                plt.subplot(1, top_n_genes, i + 1)

                # Create DataFrame for seaborn
                df = pd.DataFrame({
                    'Expression': np.concatenate([real_data[:, gene_idx], pred_data[:, gene_idx]]),
                    'Type': ['Real'] * len(real_data) + ['Predicted'] * len(pred_data),
                    'Cell Type': cell_labels * 2
                })

                # Create violin plot
                sns.violinplot(x='Cell Type', y='Expression', hue='Type', data=df, split=True)
                plt.title(f"Gene {gene_idx}")
                plt.xticks(rotation=45)
                if i == 0:
                    plt.ylabel('Expression Level')
                else:
                    plt.ylabel('')

            plt.tight_layout()
            plt.suptitle(f"{model} prediction on {drug} - Top {top_n_genes} genes by variance", y=1.05)
            plt.savefig(output_dir / f"violin"/ f"{model}_{drug}_violin_plot.pdf", bbox_inches='tight')
            plt.close()


def create_heatmaps():
    """Create separate heatmaps with consistent color scales for better comparison"""

    # Process each drug separately to establish common color scales
    for drug in drugs_to_visualize:
        # Collect data for all models for this drug
        all_model_data = {}

        for model in models:
            real_data, pred_data, cell_labels = load_data_for_drug(model, test_sets[0], drug)
            if real_data is not None:
                # Calculate average expression per gene for each cell type
                cell_types_unique = np.unique(cell_labels)
                real_means = []
                pred_means = []

                for ct in cell_types_unique:
                    mask = np.array(cell_labels) == ct
                    real_means.append(np.mean(real_data[mask], axis=0))
                    pred_means.append(np.mean(pred_data[mask], axis=0))

                all_model_data[model] = {
                    'real': np.array(real_means),
                    'pred': np.array(pred_means),
                    'diff': np.array(pred_means) - np.array(real_means),
                    'cell_types': cell_types_unique
                }

        if not all_model_data:
            continue

        # Determine common scales for each type of plot
        real_min = min([data['real'].min() for data in all_model_data.values()])
        real_max = max([data['real'].max() for data in all_model_data.values()])

        pred_min = min([data['pred'].min() for data in all_model_data.values()])
        pred_max = max([data['pred'].max() for data in all_model_data.values()])

        # Use the larger of the two ranges for both real and predicted for consistency
        expr_min = min(real_min, pred_min)
        expr_max = max(real_max, pred_max)

        # For difference heatmaps, find symmetric scale
        diff_max = max([abs(data['diff'].min()) for data in all_model_data.values()] +
                       [abs(data['diff'].max()) for data in all_model_data.values()])
        diff_min = -diff_max

        # Set larger font size
        plt.rcParams.update({'font.size': 26})

        # Create heatmaps for each model with consistent scales
        for model, data in all_model_data.items():
            # Real data heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(data['real'], cmap="viridis", vmin=expr_min, vmax=expr_max,
                        xticklabels=False, yticklabels=data['cell_types'], cbar_kws={'label': 'Expression'})
            plt.xlabel("基因", fontsize=28, fontproperties=_chinese_font_props)
            plt.ylabel("细胞类型", fontsize=28, fontproperties=_chinese_font_props)
            plt.tight_layout()
            plt.savefig(output_dir / "heatmap" / f"{model}_{drug}_real_heatmap_cn.pdf", bbox_inches='tight')
            plt.close()

            # Predicted data heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(data['pred'], cmap="viridis", vmin=expr_min, vmax=expr_max,
                        xticklabels=False, yticklabels=data['cell_types'], cbar_kws={'label': 'Expression'})
            plt.xlabel("基因", fontsize=28, fontproperties=_chinese_font_props)
            plt.ylabel("细胞类型", fontsize=28, fontproperties=_chinese_font_props)
            plt.tight_layout()
            plt.savefig(output_dir / "heatmap" / f"{model}_{drug}_pred_heatmap_cn.pdf", bbox_inches='tight')
            plt.close()

            # Difference heatmap
            plt.figure(figsize=(10, 6))
            # sns.heatmap(data['diff'], cmap="RdBu_r", vmin=diff_min, vmax=diff_max, center=0,
            #             xticklabels=False, yticklabels=data['cell_types'], cbar_kws={'label': 'Prediction - Real'})
            sns.heatmap(data['diff'], cmap="RdBu_r", vmin=diff_min, vmax=diff_max, center=0,
                        xticklabels=False, yticklabels=data['cell_types'])
            plt.xlabel("基因", fontsize=28, fontproperties=_chinese_font_props)
            plt.ylabel("细胞类型", fontsize=28, fontproperties=_chinese_font_props)
            plt.tight_layout()
            plt.savefig(output_dir / "heatmap" / f"{model}_{drug}_diff_heatmap_cn.pdf", bbox_inches='tight')
            plt.close()


# def create_heatmaps():
#     """Create heatmaps comparing real and predicted expression patterns"""
#     for drug in drugs_to_visualize:
#         for model in models:
#             real_data, pred_data, cell_labels = load_data_for_drug(model, test_sets[0], drug)
#
#             if real_data is None:
#                 continue
#
#             # Calculate average expression per gene for visualization clarity
#             cell_types_unique = np.unique(cell_labels)
#             real_means = []
#             pred_means = []
#
#             for ct in cell_types_unique:
#                 mask = np.array(cell_labels) == ct
#                 real_means.append(np.mean(real_data[mask], axis=0))
#                 pred_means.append(np.mean(pred_data[mask], axis=0))
#
#             real_means = np.array(real_means)
#             pred_means = np.array(pred_means)
#
#             # Create heatmap
#             plt.figure(figsize=(14, 8))
#
#             # Real data heatmap
#             plt.subplot(1, 2, 1)
#             sns.heatmap(real_means, cmap="viridis", center=0,
#                         xticklabels=False, yticklabels=cell_types_unique)
#             plt.title("Real Expression")
#             plt.xlabel("Genes")
#             plt.ylabel("Cell Type")
#
#             # Predicted data heatmap
#             plt.subplot(1, 2, 2)
#             sns.heatmap(pred_means, cmap="viridis", center=0,
#                         xticklabels=False, yticklabels=cell_types_unique)
#             plt.title("Predicted Expression")
#             plt.xlabel("Genes")
#
#             plt.tight_layout()
#             plt.suptitle(f"{model} on {drug}: Gene Expression Patterns", y=1.05)
#             plt.savefig(output_dir / f"{model}_{drug}_heatmap.pdf", bbox_inches='tight')
#             plt.close()


if __name__ == "__main__":
    create_heatmaps()
    print("Visualization complete!")