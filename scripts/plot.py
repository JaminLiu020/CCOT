import os
import numpy as np
import torch
from umap import UMAP
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_scatter(target, target_labels, transport, transport_labels, control_data, condition, savedir, mode='global'):
    # map_model = './datasets/scrna-norman/umap_model.pkl'
    # savedir = os.path.join(savedir, 'umap')
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir, exist_ok=True)

    # 确保张量在 CPU 上
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(transport, torch.Tensor):
        transport = transport.cpu().numpy()

    target = np.asarray(target)
    transport = np.asarray(transport)

    # 合并数据和标签
    data = np.concatenate((target, transport), axis=0)
    # labels = np.concatenate((target_labels, transport_labels), axis=0)

    # 应用 UMAP 进行降维
    # default spread 1.0
    # default min_dist 0.1
    # default n_neighbors 15
    # reducer = UMAP(n_components=2, random_state=42, spread=3.0, min_dist=0.8, n_neighbors=5)
    reducer = UMAP(n_components=2, random_state=42, spread=2.0, min_dist=0.4, n_neighbors=15, n_jobs=1)
    # embedding = reducer.fit_transform(data)
    reducer.fit(target)
    target = reducer.transform(target)
    transport = reducer.transform(transport)
    # control_data = reducer.transform(control_data)

    # target = embedding[:target.shape[0]]
    # transport = embedding[target.shape[0]:]

    # 自定义颜色映射
    colors = ['blue', 'red', 'orange']
    cmap = ListedColormap(colors)

    # # 自定义形状映射
    # markers = ['o', 's']  # 'o' for target, 's' for transport

    # annnotate_scatter(target, target_labels, condition, cmap, savedir, mode, object='target')
    # annnotate_scatter(transport, transport_labels, condition, cmap, savedir, mode, object='predict')
    # annnotate_scatter(control_data, transport_labels, condition, cmap, savedir, mode, object='source')
    # 调用绘图函数，合并 target 和 transport 数据
    plot_combined_scatter(target, target_labels, transport, transport_labels, condition, savedir, mode)

    return


def annnotate_scatter(data, labels, condition, cmap, savedir, mode, object=None):
    """
    c=labels：指定每个数据点的颜色，颜色根据 labels 数组映射。
    cmap='coolwarm'：选择颜色映射。
    s=5：设置数据点的大小。
    plt.colorbar(label='Label')：显示颜色条，并添加标签说明。
    """
    # 绘制图表
    fig, ax = plt.subplots(figsize=(8, 6))

    # 调整透明度和边框
    alpha = 0.6
    edgecolor = 'k'
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, s=30, alpha=alpha,
                         edgecolor=edgecolor)
    # 去除坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])
    # 添加坐标轴标签
    # ax.text(0.5, -0.05, 'UMAP1', transform=ax.transAxes, ha='center', va='center')
    # ax.text(-0.05, 0.5, 'UMAP2', transform=ax.transAxes, ha='center', va='center', rotation='vertical')

    # 添加样例点及其标注
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='A549'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='K562'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='MCF7')
    ]
    ax.legend(handles=legend_elements, loc='best')
    # 添加图表标题
    # ax.set_title(f'{object} data under {condition} doseage', fontsize=16)

    plt.savefig(os.path.join(savedir, f'{mode}_{condition}_{object}_umap_embedding.pdf'), format='pdf')
    plt.close()
    return


def plot_combined_scatter(target, target_labels, transport, transport_labels, condition, savedir, mode):
    """
    在同一张图上绘制 target 和 transport 的散点图
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # 自定义颜色映射
    colors = ['blue', 'red', 'orange']  # 蓝色, 红色, 橙色
    cmap = ListedColormap(colors)

    # Plot transport with solid colors
    ax.scatter(transport[:, 0], transport[:, 1], c=transport_labels, cmap=cmap, s=30, alpha=0.8, label='Transport',
               edgecolor='k')

    # Plot target with lighter colors (using alpha)
    ax.scatter(target[:, 0], target[:, 1], c=target_labels, cmap=cmap, s=30, alpha=0.1, label='Target', edgecolor='k')

    # 去除坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])

    # 添加坐标轴标签
    # ax.text(0.5, -0.05, 'UMAP1', transform=ax.transAxes, ha='center', va='center')
    # ax.text(-0.05, 0.5, 'UMAP2', transform=ax.transAxes, ha='center', va='center', rotation='vertical')

    # 添加样例点及其标注
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=16, label='A549'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=16, label='K562'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=16, label='MCF7')
    ]
    # 将图例移到图像右上角的外部，避免遮挡数据点
    ax.legend(handles=legend_elements, loc='best', fontsize=18)

    # 保存图像
    plt.savefig(os.path.join(savedir, f'{mode}_{condition}_combined_umap_embedding.pdf'), format='pdf')
    plt.close()

    return


def process_eval_data(eval_dir):
    # 创建用于存储全局数据的字典
    global_data = defaultdict(dict)

    # 获取data目录
    data_dir = os.path.join(eval_dir, 'data')

    # 确保plot目录存在
    plot_dir = os.path.join(eval_dir, 'plot')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    # 细胞类型到标签的映射
    cell_type_to_label = {
        'A549': 0,
        'K562': 1,
        'MCF7': 2
    }

    # 遍历测试集文件夹（test/ood等）
    # for test_set in os.listdir(data_dir):
    for test_set in ['ood_DEGs']:
        test_set_dir = os.path.join(data_dir, test_set)
        if not os.path.isdir(test_set_dir):
            continue

        # 确保测试集对应的plot目录存在
        test_set_plot_dir = os.path.join(plot_dir, test_set)
        if not os.path.exists(test_set_plot_dir):
            os.makedirs(test_set_plot_dir, exist_ok=True)

        # 初始化这个测试集的全局数据累积变量
        real_data_all = []
        pred_data_all = []
        source_data_all = []
        real_labels_all = []
        pred_labels_all = []
        source_labels_all = []

        # 遍历药物文件夹
        for drug_name in os.listdir(test_set_dir):
            drug_dir = os.path.join(test_set_dir, drug_name)
            if not os.path.isdir(drug_dir):
                continue

            # 确保药物对应的plot目录存在
            # drug_plot_dir = os.path.join(test_set_plot_dir, drug_name)
            # if not os.path.exists(drug_plot_dir):
            #     os.makedirs(drug_plot_dir, exist_ok=True)
            drug_plot_dir = test_set_plot_dir

            # 初始化当前药物的数据容器
            real_data = []
            pred_data = []
            source_data = []
            real_labels = []
            pred_labels = []
            source_labels = []

            # 遍历细胞类型文件（A549.npz, K562.npz, MCF7.npz）
            for cell_type in cell_type_to_label.keys():
                cell_file = os.path.join(drug_dir, f"{cell_type}.npz")

                if os.path.exists(cell_file):
                    try:
                        # 加载npz文件
                        load_data = np.load(cell_file)

                        # 提取数据
                        cell_real_data = load_data['real']
                        cell_pred_data = load_data['pred']
                        cell_source_data = load_data['source']

                        # 创建对应的标签
                        cell_label = cell_type_to_label[cell_type]
                        cell_real_labels = np.full(cell_real_data.shape[0], cell_label)
                        cell_pred_labels = np.full(cell_pred_data.shape[0], cell_label)
                        cell_source_labels = np.full(cell_source_data.shape[0], cell_label)

                        # 添加到当前药物的数据容器中
                        real_data.append(cell_real_data)
                        pred_data.append(cell_pred_data)
                        source_data.append(cell_source_data)
                        real_labels.append(cell_real_labels)
                        pred_labels.append(cell_pred_labels)
                        source_labels.append(cell_source_labels)

                    except Exception as e:
                        print(f"Error loading {cell_file}: {e}")

            # 合并当前药物的所有细胞类型数据
            if real_data and pred_data and source_data:
                real_data = np.vstack(real_data)
                pred_data = np.vstack(pred_data)
                source_data = np.vstack(source_data)
                real_labels = np.concatenate(real_labels)
                pred_labels = np.concatenate(pred_labels)
                source_labels = np.concatenate(source_labels)

                # 为当前药物画UMAP图
                plot_scatter(
                    target=real_data,
                    target_labels=real_labels,
                    transport=pred_data,
                    transport_labels=pred_labels,
                    control_data=source_data,
                    condition=drug_name,
                    savedir=drug_plot_dir,
                    mode='local'
                )

                # 将当前药物的数据添加到全局数据累积变量中
                real_data_all.append(real_data)
                pred_data_all.append(pred_data)
                source_data_all.append(source_data)
                real_labels_all.append(real_labels)
                pred_labels_all.append(pred_labels)
                source_labels_all.append(source_labels)

                # 清空当前药物的数据容器，为下一个药物准备
                real_data = None
                pred_data = None
                source_data = None
                real_labels = None
                pred_labels = None
                source_labels = None

        # 为当前测试集创建全局数据（合并所有药物的数据）
        if real_data_all and pred_data_all and source_data_all:
            real_data_all = np.vstack(real_data_all)
            pred_data_all = np.vstack(pred_data_all)
            source_data_all = np.vstack(source_data_all)
            real_labels_all = np.concatenate(real_labels_all)
            pred_labels_all = np.concatenate(pred_labels_all)
            source_labels_all = np.concatenate(source_labels_all)

            # 保存全局数据用于后续分析
            global_data[test_set] = {
                'real_data': real_data_all,
                'pred_data': pred_data_all,
                'source_data': source_data_all,
                'real_labels': real_labels_all,
                'pred_labels': pred_labels_all,
                'source_labels': source_labels_all
            }

            # # 为整个测试集画UMAP图
            # plot_scatter(
            #     target=real_data_all,
            #     target_labels=real_labels_all,
            #     transport=pred_data_all,
            #     transport_labels=pred_labels_all,
            #     control_data=source_data_all,
            #     condition='all_drugs',
            #     savedir=test_set_plot_dir,
            #     mode='global'
            # )
    # np.savez(os.path.join(data_dir, 'global_data.npz'), **global_data)
    return global_data


# 示例使用方法
if __name__ == "__main__":
    eval_dir = "/home/jamin/condot/results/nine_drugs/977genes/drug_pert/contrast_experiment/chemCPA/eval"
    global_data = process_eval_data(eval_dir)

    # global_data 包含了所有测试集的数据，可以用于后续的置换检验
    print("Processing completed. Global data is ready for permutation tests.")
