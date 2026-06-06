import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import umap
# import seaborn as sns
import gc
# import warnings
from pathlib import Path


def load_data_from_drug_folder(drug_folder_path):
    """
    从药物文件夹中加载所有细胞类型的数据，并整合为source和real数据集。
    """
    # 获取药物名称（文件夹名）
    drug_name = os.path.basename(drug_folder_path)
    print(f"处理药物: {drug_name}")

    # 存储所有细胞类型的数据
    all_source_data = []
    all_real_data = []
    all_cell_types = []

    # 遍历药物文件夹中的所有npz文件
    for file_name in os.listdir(drug_folder_path):
        if file_name.endswith('.npz'):
            # 从文件名提取细胞类型
            cell_type = file_name.split('.')[0]
            file_path = os.path.join(drug_folder_path, file_name)

            print(f"  加载细胞类型: {cell_type}")

            try:
                # 加载npz文件
                data = np.load(file_path)

                # 提取real和source数据
                if 'real' in data and 'source' in data:
                    real_data = data['real']
                    source_data = data['source']

                    # 添加数据到列表
                    all_real_data.append(real_data)
                    all_source_data.append(source_data)

                    # 创建细胞类型标签
                    cell_types = [cell_type] * len(real_data)
                    all_cell_types.extend(cell_types)
                else:
                    print(f"  警告: {file_name} 中未找到 'real' 或 'source' 数据")
            except Exception as e:
                print(f"  错误: 无法加载 {file_path}: {str(e)}")

    # 将所有数据整合为numpy数组
    if all_source_data and all_real_data:
        source_data = np.vstack(all_source_data)
        real_data = np.vstack(all_real_data)
        return source_data, real_data, all_cell_types, drug_name
    else:
        print(f"  错误: 在药物 {drug_name} 的文件夹中未找到有效数据")
        return None, None, None, drug_name


def create_umap_embeddings(source_data, real_data):
    """
    使用UMAP创建数据的低维嵌入，避免n_jobs警告
    """
    # 创建UMAP模型
    umap_model = umap.UMAP(
        spread = 3.0,
        n_neighbors=10,
        min_dist=0.5,
        n_components=2,
        random_state=42,
        # random_state=77,
        n_jobs=1  # 设置n_jobs为1以避免警告
    )

    # # 将source和real数据合并以进行联合嵌入
    # combined_data = np.vstack([source_data, real_data])
    #
    # # 拟合和转换数据
    # combined_embedding = umap_model.fit_transform(combined_data)
    #
    # # 分离嵌入
    # n_source = len(source_data)
    # source_embedding = combined_embedding[:n_source]
    # target_embedding = combined_embedding[n_source:]
    source_embedding = umap_model.fit_transform(source_data)
    target_embedding = umap_model.fit_transform(real_data)

    return source_embedding, target_embedding


def visualize_transport_map(source_embedding, target_embedding, cell_types, drug_name, save_path, format="pdf"):
    """
    创建并保存源数据和目标数据的传输映射可视化，按要求优化了字体大小和图表布局
    """
    # 创建图形
    fig = plt.figure(figsize=(14, 7))

    # 获取唯一的细胞类型并分配颜色
    # unique_cell_types = sorted(list(set(cell_types)))
    # colors = sns.color_palette("husl", len(unique_cell_types))
    # cell_type_color_map = {cell_type: colors[i] for i, cell_type in enumerate(unique_cell_types)}

    # 获取唯一的细胞类型
    unique_cell_types = sorted(list(set(cell_types)))

    # 自定义RGB颜色 (需要将RGB值归一化到0-1的范围)
    custom_colors = [
        (230 / 255, 111 / 255, 81 / 255),  # 红色 (A549)
        (42 / 255, 157 / 255, 140 / 255),  # 绿色 (K562)
        (82 / 255, 143 / 255, 173 / 255)  # 蓝色 (MCF7)
    ]

    # 创建颜色映射
    cell_type_color_map = {cell_type: custom_colors[i] for i, cell_type in enumerate(unique_cell_types)}

    # 创建细胞类型标签映射（修改为Cell type A, Cell type B等）
    cell_type_labels = {}
    for i, ct in enumerate(unique_cell_types):
        cell_type_labels[ct] = f"Cell type {chr(65 + i)}"  # A, B, C, ...

    # 按细胞类型分组数据点
    cell_type_indices = {cell_type: [] for cell_type in unique_cell_types}
    for i, ct in enumerate(cell_types):
        cell_type_indices[ct].append(i)

    # 创建子图
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # 设置子图位置，给顶部留出更多空间
    ax1_pos = [0.09, 0.05, 0.38, 0.85]  # [left, bottom, width, height]
    ax2_pos = [0.56, 0.05, 0.38, 0.85]
    ax1.set_position(ax1_pos)
    ax2.set_position(ax2_pos)

    # # 创建图例元素 - 使用新的Cell type标签
    # legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=cell_type_labels[ct],
    #                               markerfacecolor=cell_type_color_map[ct], markersize=15)  # 增大标记点大小
    #                    for ct in unique_cell_types]
    #
    # # 将图例放置在Source图(ax1)内部左上角
    # legend = ax1.legend(
    #     handles=legend_elements,
    #     # loc='best',
    #     loc='upper left',
    #     bbox_to_anchor=(-0.65, 1.1),  # 调整位置到左上角内侧
    #     frameon=True,
    #     framealpha=0.9,
    #     fontsize=22,
    #     # title='Cell Types',
    #     title_fontsize=22,
    #     markerscale=1.5  # 增加图例标记的大小
    # )

    # 在源空间绘制点
    for cell_type, indices in cell_type_indices.items():
        ax1.scatter(
            source_embedding[indices, 0],
            source_embedding[indices, 1],
            c=[cell_type_color_map[cell_type]],
            alpha=0.8,
            s=40,
            edgecolors='white',
            linewidths=0.5
        )

    # 在目标空间绘制点
    for cell_type, indices in cell_type_indices.items():
        ax2.scatter(
            target_embedding[indices, 0],
            target_embedding[indices, 1],
            c=[cell_type_color_map[cell_type]],
            alpha=0.8,
            s=40,
            edgecolors='white',
            linewidths=0.5
        )

    # 设置图表标题 - 手动放置在子图上方的中央位置
    # ax1.text(0.5, 1.05, "Source", fontsize=24, ha='center', transform=ax1.transAxes)
    # ax1.text(0.5, -0.16, "Source", fontsize=24, ha='center', transform=ax1.transAxes)
    # ax2.text(0.5, -0.16, "Target", fontsize=24, ha='center', transform=ax2.transAxes)

    # 关闭坐标轴
    ax1.axis('off')
    ax2.axis('off')

    # 画美观的圆角边框
    for ax, pos in zip([ax1, ax2], [ax1_pos, ax2_pos]):
        p_bbox = FancyBboxPatch(
            (-0.05, -0.05),
            width=1.1,
            height=1.1,
            boxstyle="round,pad=0.05",
            ec="black",
            fc="none",
            linewidth=2.0,
            linestyle='--',
            clip_on=False,
            transform=ax.transAxes
        )
        ax.add_patch(p_bbox)

    # 确保坐标范围是固定的，避免自动调整
    min_x1, max_x1 = source_embedding[:, 0].min(), source_embedding[:, 0].max()
    min_y1, max_y1 = source_embedding[:, 1].min(), source_embedding[:, 1].max()
    min_x2, max_x2 = target_embedding[:, 0].min(), target_embedding[:, 0].max()
    min_y2, max_y2 = target_embedding[:, 1].min(), target_embedding[:, 1].max()

    # 为坐标范围添加一些边距
    margin = 0.1
    x1_range = max_x1 - min_x1
    y1_range = max_y1 - min_y1
    x2_range = max_x2 - min_x2
    y2_range = max_y2 - min_y2

    ax1.set_xlim(min_x1 - margin * x1_range, max_x1 + margin * x1_range)
    ax1.set_ylim(min_y1 - margin * y1_range, max_y1 + margin * y1_range)
    ax2.set_xlim(min_x2 - margin * x2_range, max_x2 + margin * x2_range)
    ax2.set_ylim(min_y2 - margin * y2_range, max_y2 + margin * y2_range)

    # 直接绘制源和目标之间的连线，不使用tight_layout
    # 获取子图在图中的位置
    ax1_bbox = ax1.get_position()
    ax2_bbox = ax2.get_position()

    # 正确计算点的图形坐标
    def get_figure_coordinate(ax, bbox, x, y):
        # 使用轴的数据范围计算相对位置
        x_rel = (x - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
        y_rel = (y - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])

        # 转换为图形坐标
        x_fig = bbox.x0 + x_rel * bbox.width
        y_fig = bbox.y0 + y_rel * bbox.height

        return x_fig, y_fig

    # 按细胞类型分组绘制连线
    for cell_type, indices in cell_type_indices.items():
        color = cell_type_color_map[cell_type]

        for idx in indices:
            # 获取源和目标点的数据坐标
            source_x, source_y = source_embedding[idx]
            target_x, target_y = target_embedding[idx]

            # 转换为图形坐标
            source_x_fig, source_y_fig = get_figure_coordinate(ax1, ax1_bbox, source_x, source_y)
            target_x_fig, target_y_fig = get_figure_coordinate(ax2, ax2_bbox, target_x, target_y)

            # 绘制连线
            line = plt.Line2D(
                [source_x_fig, target_x_fig],
                [source_y_fig, target_y_fig],
                transform=fig.transFigure,
                color=color,
                alpha=0.2,
                linewidth=0.5,
                zorder=0
            )
            fig.add_artist(line)

    # 创建保存目录（如果不存在）
    os.makedirs(save_path, exist_ok=True)

    # 保存图像
    # 根据指定格式保存图像
    file_extension = format.lower()
    save_file = os.path.join(save_path, f"{drug_name}_combined_umap_embedding.{file_extension}")
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    print(f"图像已保存至: {save_file}")

    # 关闭图像以释放内存
    plt.close()


def process_drug_folder(drug_folder_path, base_save_path, image_format="pdf"):
    """
    处理单个药物文件夹，加载数据、创建UMAP嵌入并可视化
    """
    # 加载药物数据
    source_data, real_data, cell_types, drug_name = load_data_from_drug_folder(drug_folder_path)

    if source_data is not None and real_data is not None:
        try:
            # 创建UMAP嵌入
            source_embedding, target_embedding = create_umap_embeddings(source_data, real_data)

            # 可视化并保存传输映射
            drug_save_path = os.path.join(base_save_path, drug_name)
            visualize_transport_map(source_embedding, target_embedding, cell_types, drug_name, drug_save_path, format=image_format)

            print(f"已完成药物 {drug_name} 的处理\n")

            # 清除变量以释放内存
            del source_data, real_data, cell_types
            del source_embedding, target_embedding
            gc.collect()

        except Exception as e:
            print(f"处理药物 {drug_name} 时出错: {str(e)}")


def process_all_drug_folders(base_folder, image_format="pdf"):
    """
    处理基础文件夹下的所有药物文件夹
    """
    # 确保基础文件夹存在
    if not os.path.exists(base_folder):
        print(f"错误: 文件夹 {base_folder} 不存在")
        return

    # 设置保存路径
    base_save_path = "/home/jamin/condot/scripts/visualization_results/transport_map"

    # 确保保存目录存在
    Path(base_save_path).mkdir(parents=True, exist_ok=True)

    # 获取所有药物文件夹
    drug_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    print(f"找到 {len(drug_folders)} 个药物文件夹")

    # 遍历并处理每个药物文件夹
    for drug_folder in drug_folders:
        drug_folder_path = os.path.join(base_folder, drug_folder)
        process_drug_folder(drug_folder_path, base_save_path, image_format)

        # 在处理完每个药物后强制清理内存
        gc.collect()



if __name__ == "__main__":
    # 抑制UMAP警告
    # warnings.filterwarnings("ignore", category=UserWarning, module="umap")

    # 这里需要提供包含药物文件夹的基础目录路径
    base_folder = "/home/jamin/condot/results/nine_drugs/977genes/drug_pert/visualization_exp/CCOT/ood_DEGs"  # 请替换为实际路径
    image_format = "svg"  # 可选值: "pdf", "svg", "eps"
    # 处理所有药物文件夹
    process_all_drug_folders(base_folder, image_format=image_format)

    print("所有药物处理完毕！")