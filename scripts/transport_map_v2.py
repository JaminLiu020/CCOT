import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# 设置随机种子确保结果可重复
np.random.seed(42)


# 生成三簇实心椭圆形数据
def generate_cluster_data(centers, params, n_samples=1000):
    """
    生成具有实心椭圆形分布的数据簇，确保数据均匀分布在整个椭圆内部

    参数:
        centers: 每个簇的中心位置列表 [(x1,y1), (x2,y2), (x3,y3)]
        params: 每个簇的形状参数列表 [(a1,b1,angle1), (a2,b2,angle2), (a3,b3,angle3)]
                a和b是椭圆的长轴和短轴，angle是椭圆的旋转角度（弧度）
        n_samples: 每个簇的样本数量

    返回:
        all_data: 所有数据点的坐标
        labels: 对应的标签
    """
    data = []
    labels = []

    for i, (center, (a, b, angle)) in enumerate(zip(centers, params)):
        # 使用更均匀的方法生成椭圆 - 确保中心区域也有点

        # 生成均匀分布的圆形点，从中心到边缘
        # 使用平方根分布策略实现区域均匀分布，从0开始确保中心有点
        r = np.sqrt(np.random.uniform(0.0, 1.0, n_samples))  # 平方根使点在整个圆内均匀分布
        theta = np.random.uniform(0, 2 * np.pi, n_samples)  # 均匀角度

        # 转换为直角坐标
        x_circle = r * np.cos(theta)
        y_circle = r * np.sin(theta)

        # 拉伸成椭圆（应用长短轴）
        x_ellipse = a * x_circle
        y_ellipse = b * y_circle

        # 旋转椭圆
        x = center[0] + (x_ellipse * np.cos(angle) - y_ellipse * np.sin(angle))
        y = center[1] + (x_ellipse * np.sin(angle) + y_ellipse * np.cos(angle))

        # 添加轻微的高斯扰动，使分布更自然
        x += np.random.normal(0, a * 0.05, n_samples)
        y += np.random.normal(0, b * 0.05, n_samples)

        cluster_data = np.column_stack((x, y))
        data.append(cluster_data)
        labels.extend([f"Cell type {chr(65 + i)}"] * n_samples)

    # 合并所有簇的数据
    all_data = np.vstack(data)
    return all_data, labels


# 生成source数据 - 三个不同形状的椭圆
source_centers = [(-8, 6), (3, -5), (8, 7)]
# 参数格式: (a, b, angle) - a是长轴，b是短轴，angle是旋转角度（弧度）
source_params = [
    (3.5, 2.2, np.pi / 4),  # 第一个簇：扁平椭圆，45度旋转
    (3.0, 1.8, np.pi / 2),  # 第二个簇：中等椭圆，90度旋转
    (4.0, 2.5, np.pi / 6)  # 第三个簇：大椭圆，30度旋转
]
source_data, source_labels = generate_cluster_data(source_centers, source_params)

# 生成target数据 - 也是三个椭圆，但形状和位置不同
target_centers = [(5, 5), (-7, -6), (0, 8)]
# 不同的形状参数
target_params = [
    (3.2, 2.0, np.pi / 3),  # 第一个簇：60度旋转
    (3.8, 2.3, np.pi),  # 第二个簇：180度旋转
    (3.0, 1.5, np.pi / 2.5)  # 第三个簇：72度旋转
]
target_data, target_labels = generate_cluster_data(target_centers, target_params)


# 绘制transport map
def visualize_transport_map(source_data, target_data, labels):
    # 创建图形
    fig = plt.figure(figsize=(14, 7))

    # 获取唯一的标签
    unique_labels = sorted(set(labels))
    colors = sns.color_palette("husl", len(unique_labels))
    label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    # 按标签分组数据点
    source_by_label = {label: [] for label in unique_labels}
    target_by_label = {label: [] for label in unique_labels}

    for i, label in enumerate(labels):
        source_by_label[label].append(source_data[i])
        target_by_label[label].append(target_data[i])

    # 将列表转换为numpy数组
    for label in unique_labels:
        source_by_label[label] = np.array(source_by_label[label])
        target_by_label[label] = np.array(target_by_label[label])

    # 创建子图
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # 设置子图位置，避免重叠
    ax1_pos = [0.08, 0.05, 0.38, 0.85]
    ax2_pos = [0.58, 0.05, 0.38, 0.85]
    ax1.set_position(ax1_pos)
    ax2.set_position(ax2_pos)

    # 在源空间绘制点 - 减小点的大小从60到40
    for label, data in source_by_label.items():
        ax1.scatter(
            data[:, 0],
            data[:, 1],
            c=[label_color_map[label]],
            alpha=0.8,
            s=40,  # 减小点的大小
            edgecolors='white',
            linewidths=0.5
        )

    # 在目标空间绘制点 - 减小点的大小从60到40
    for label, data in target_by_label.items():
        ax2.scatter(
            data[:, 0],
            data[:, 1],
            c=[label_color_map[label]],
            alpha=0.8,
            s=40,  # 减小点的大小
            edgecolors='white',
            linewidths=0.5
        )

    # 添加图例 - 保持原来的大小
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=label_color_map[label], markersize=15)
                       for label in unique_labels]

    legend = ax1.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(0.05, 0.95),
        frameon=True,
        framealpha=0.9,
        fontsize=22,
        title='Cell Types',
        title_fontsize=22,
        markerscale=1.5
    )

    # 设置图表标题
    ax1.text(0.5, 1.08, "Source", fontsize=24, ha='center', transform=ax1.transAxes)
    ax2.text(0.5, 1.08, "Target", fontsize=24, ha='center', transform=ax2.transAxes)

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
            clip_on=False,
            transform=ax.transAxes
        )
        ax.add_patch(p_bbox)

    # 获取子图在图中的位置
    ax1_bbox = ax1.get_position()
    ax2_bbox = ax2.get_position()

    # 调整坐标范围以适应所有数据点
    ax1.set_xlim(np.min(source_data[:, 0]) - 1, np.max(source_data[:, 0]) + 1)
    ax1.set_ylim(np.min(source_data[:, 1]) - 1, np.max(source_data[:, 1]) + 1)
    ax2.set_xlim(np.min(target_data[:, 0]) - 1, np.max(target_data[:, 0]) + 1)
    ax2.set_ylim(np.min(target_data[:, 1]) - 1, np.max(target_data[:, 1]) + 1)

    # 绘制连接线 - 增加线宽从0.5到0.8
    for label in unique_labels:
        color = label_color_map[label]
        source_points = source_by_label[label]
        target_points = target_by_label[label]

        # 为了避免图表过于复杂，只绘制部分连线（比如每10个点）
        step = 10  # 每10个点绘制一条连线
        for i in range(0, len(source_points), step):
            if i < len(source_points):
                source_x, source_y = source_points[i]
                target_x, target_y = target_points[i]

                # 计算图形坐标
                x1_range = ax1.get_xlim()[1] - ax1.get_xlim()[0]
                y1_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
                x2_range = ax2.get_xlim()[1] - ax2.get_xlim()[0]
                y2_range = ax2.get_ylim()[1] - ax2.get_ylim()[0]

                # 转换为相对位置
                source_x_rel = (source_x - ax1.get_xlim()[0]) / x1_range
                source_y_rel = (source_y - ax1.get_ylim()[0]) / y1_range
                target_x_rel = (target_x - ax2.get_xlim()[0]) / x2_range
                target_y_rel = (target_y - ax2.get_ylim()[0]) / y2_range

                # 转换为图形坐标
                source_x_fig = ax1_bbox.x0 + source_x_rel * ax1_bbox.width
                source_y_fig = ax1_bbox.y0 + source_y_rel * ax1_bbox.height
                target_x_fig = ax2_bbox.x0 + target_x_rel * ax2_bbox.width
                target_y_fig = ax2_bbox.y0 + target_y_rel * ax2_bbox.height

                # 绘制连线 - 增加线宽，保持相同的透明度
                line = plt.Line2D(
                    [source_x_fig, target_x_fig],
                    [source_y_fig, target_y_fig],
                    transform=fig.transFigure,
                    color=color,
                    alpha=0.15,
                    linewidth=1.0,  # 增加线宽
                    zorder=0
                )
                fig.add_artist(line)

    # plt.savefig('synthetic_transport_map.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('synthetic_transport_map.svg', bbox_inches='tight', dpi=300)
    plt.show()


# 绘制图形
visualize_transport_map(source_data, target_data, source_labels)