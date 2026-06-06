import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
import numpy as np


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

# --- File Path Settings ---
# Manually specify the absolute path to your Excel data file here.
# Example: input_file_path = r'C:\Users\YourUser\Documents\your_data\id_test.xlsx'
# Use a raw string (r'') or double backslashes (\\) for Windows paths.
input_candidates = [
    r'/home/jamin/condot/old_result/nine_drugs/977genes/drug_pert/hyperpara_sensitivity_analysis/DEGs_id_test.xlsx',
    r'/home/jamin/condot/results/nine_drugs/977genes/drug_pert/hyperpara_sensitivity_analysis/DEGs_id_test.xlsx',
    r'/home/jamin/condot/scripts/scripts\visualization_results\hyperpara_sensitivity_analysis/id_test.xlsx',
]
input_file_path = next((path for path in input_candidates if os.path.exists(path)), input_candidates[0])

# Set the base output directory
output_base_dir = r'/home/jamin/condot/scripts/visualization_results/hyperpara_sensitivity_analysis' # Use raw string to avoid escape issues

# Force output into DEGs_id_test to match target artifact directory.
output_subdir_name = 'DEGs_id_test'

output_dir = os.path.join(output_base_dir, output_subdir_name)

# --- Create Output Directory ---
# Create the directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
else:
    print(f"Output directory already exists: {output_dir}")


# --- Read Data ---
try:
    # Read data from the specified Excel file
    # If your data is not in the first sheet, add sheet_name='YourSheetName'
    df = pd.read_excel(input_file_path)
    print("Successfully loaded data from:", input_file_path)
    print("Columns in the data:", df.columns.tolist()) # Print column names for verification

except FileNotFoundError:
    print(f"Error: Input file not found at {input_file_path}")
    print("Please ensure the specified file path is correct.")
    exit() # Exit if the file is not found
except ImportError:
    print("Error: 'openpyxl' library not found.")
    print("Please install it using: pip install openpyxl")
    exit()
except Exception as e:
    print(f"An error occurred while reading the Excel file: {e}")
    exit()


# --- Define Metrics, their Column Names, and LaTeX Labels ---
# !!! IMPORTANT !!!
# Please modify the keys (e.g., 'MMD', 'L2PS') and the first two elements
# of the tuples (e.g., 'MMD_mean', 'MMD_std') to match the EXACT column names
# in your Excel file.
# The third element of the tuple is the LaTeX label for plotting.
metric_cols = {
    'MMD': ('MMD_mean', 'MMD_std', r'MMD'), # MMD
    'L2PS': ('L2PS_mean', 'L2PS_std', r'$\ell_2(PS)$'), # \ell_2(PS)
    'R2': ('R2_mean', 'R2_std', r'$R^2$'), # R^2
    'Ed': ('Ed_mean', 'Ed_std', r'$E_d$'), # E_d
    'FID': ('FID_mean', 'FID_std', r'FID'), # FID - Added for FID plotting
    # If there are other metrics, add them here following the same pattern:
    # 'Metric5': ('Metric5_mean', 'Metric5_std', r'Metric 5 Label'),
}

# Determine the hyperparameter column name
# Assuming the first column is the beta value. Modify if needed.
beta_col = df.columns[0]


# --- Set Matplotlib Parameters for LaTeX Rendering ---
# Use Matplotlib's mathtext engine for simple math symbols
# This usually doesn't require a full LaTeX installation
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm' # Use Computer Modern font for a LaTeX-like look
_english_font_name = _pick_english_serif_font()
plt.rcParams['font.family'] = _english_font_name
plt.rcParams['font.serif'] = [_english_font_name]
plt.rcParams['axes.unicode_minus'] = False
_chinese_font_name = _pick_chinese_font()
_chinese_font_props = (
    font_manager.FontProperties(family=_chinese_font_name)
    if _chinese_font_name
    else None
)

# --- Plot Sensitivity for Each Metric ---
for metric_name, (mean_col, std_col, latex_label) in metric_cols.items():
    # Check if the required columns exist in the DataFrame
    if mean_col not in df.columns or std_col not in df.columns:
        print(
            f"Warning: Columns for metric {metric_name} ({mean_col}, {std_col}) not found in data. Skipping this metric.")
        continue

    # Create figure and axes for the plot
    fig, ax = plt.subplots(figsize=(8, 5))  # Adjust figure size as needed

    # Plot the mean line (确保所有数据点都用于绘制折线)
    ax.plot(df[beta_col], df[mean_col], label=metric_name, marker='o', linestyle='-')

    # Plot the standard deviation shaded area
    lower_bound = df[mean_col] - df[std_col]
    upper_bound = df[mean_col] + df[std_col]
    ax.fill_between(df[beta_col], lower_bound, upper_bound, color=ax.lines[-1].get_color(), alpha=0.15)

    # --- Set Axis Labels and Title ---
    ax.set_xlabel(r'超参数 $\beta$', fontsize=18, fontproperties=_chinese_font_props)
    ax.set_ylabel(f'{latex_label}', fontsize=18)
    # ax.set_title(f'Sensitivity of {latex_label} to $\\beta$', fontsize=14)

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- 修改X轴刻度 ---
    # 1. 获取X轴数据的实际范围
    #    我们先确保 beta_col 列是数值类型，以便正确获取最大最小值
    x_data_numeric = pd.to_numeric(df[beta_col], errors='coerce').dropna()

    if not x_data_numeric.empty:
        min_x_val = x_data_numeric.min()
        max_x_val = x_data_numeric.max()

        # 2. 确定刻度的起始点和结束点 (我们希望是整数)
        #    横轴上限固定为15，避免显示到20。
        tick_min = np.floor(min_x_val)
        tick_max = min(np.ceil(max_x_val), 15)

        # 3. 生成新的X轴刻度位置，间隔为1
        #    np.arange(start, stop, step) 中 stop 参数是不包含在内的。
        #    所以，为了确保 tick_max 被包含，可以使用 tick_max + 1 (或 tick_max + desired_interval)。
        desired_interval = 1.0
        new_xticks = np.arange(tick_min, tick_max + desired_interval, desired_interval)

        ax.set_xticks(new_xticks)
        ax.set_xlim(tick_min, 15)

        # (可选) 如果您希望X轴刻度标签强制显示为整数 (例如 "1" 而不是 "1.0")
        # from matplotlib.ticker import FuncFormatter
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}'))
    else:
        # 如果 beta_col 数据有问题，打印警告并可能回退到默认行为或不设置特定xticks
        print(
            f"Warning: Could not determine custom x-axis ticks for {metric_name} due to problematic data in '{beta_col}'.")
        # 可以选择让 Matplotlib 自动决定刻度，或者保留原来的 ax.set_xticks(df[beta_col]) (但不推荐，因为它会导致重叠)
        # 如果不调用 ax.set_xticks()，Matplotlib 会自动尝试选择合适刻度。

    # 自动调整绘图布局以防止标签重叠
    plt.tight_layout()

    # --- Save the Plot ---
    safe_metric_name = metric_name.replace('$', '').replace('\\', '').replace('{', '').replace('}', '').replace('^',
                                                                                                                '').replace(
        '_', '').replace('(', '').replace(')', '').replace(' ', '_')
    output_filename = f'{safe_metric_name}_sensitivity_cn.pdf'
    output_filepath = os.path.join(output_dir, output_filename)

    plt.savefig(output_filepath, format='pdf', bbox_inches='tight')
    print(f"Saved {metric_name} sensitivity plot to: {output_filepath}")

    plt.close(fig)

print("\nAll plots generated and saved.")

# --- Notes on Metric Scaling and Combining Plots ---
# As mentioned before, since you indicated metrics have different scales,
# plotting them individually is the clearest approach.
# If you manually check the data (e.g., using df.head(), df.describe())
# and find metrics with similar scales that you wish to plot together,
# you can modify the loop structure above to group and plot them on the same axes.
# However, plotting individually is the default and safest method for differing scales.
