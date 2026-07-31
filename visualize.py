import copy

import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text
import os
import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point2D
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
import matplotlib.cm as cm
from dataset import *
from torch.utils.data import DataLoader
import matplotlib.colors as colors
from sklearn.metrics.pairwise import cosine_similarity


class ModelWrapperForLIG(torch.nn.Module):
    def __init__(self, model, collated_batch_template):
        super().__init__()
        self.model = model
        self.template = collated_batch_template

    # Captum 会像这样调用: forward(interpolated_input_features, *additional_forward_args)
    def forward(self, atom_indices_tensor, task_name, mode):
        batch_copy = copy.deepcopy(self.template)
        # 将 Captum 传入的张量放回模板
        batch_copy.x = atom_indices_tensor
        # 1. 正常调用模型，得到包含所有任务预测的字典
        predictions_dict = self.model(batch_copy, task_name, mode)

        # 2. 从字典中，根据传入的 task_name，抽取出我们当前关心的那一个任务的预测张量
        target_prediction_tensor = predictions_dict[task_name]

        # 3. 只返回这个张量给 Captum
        return target_prediction_tensor


def visualize_task_prompt_similarity(model, save_dir):
    """
    【升级版】使用层次聚类热图(clustermap)来可视化任务相似度。
    """
    print("\n--- Generating Hierarchical Clustered Heatmap for Task Prompts ---")

    # 1. 提取数据 (代码不变)
    try:
        prompts_tensor = model.encoder.ema_prompts.squeeze(0).detach().cpu().numpy()
        task_names = model.task_name
    except AttributeError:
        print("Error: Could not retrieve prompts from model.")
        return

    # 2. 计算相似度矩阵 (代码不变)
    similarity_matrix = cosine_similarity(prompts_tensor)
    df_similarity = pd.DataFrame(similarity_matrix, index = task_names, columns = task_names)

    # 3. 使用 seaborn.clustermap 进行绘图
    # 这是一个功能非常强大的函数，它会自动完成聚类、排序和绘图
    g = sns.clustermap(
        df_similarity,
        method = 'average',  # 聚类算法
        metric = 'cosine',  # 计算距离的度量
        cmap = "coolwarm",  # 色板
        figsize = (20, 20),  # 图像尺寸
        linewidths = .5,
        annot = False  # 在59x59的图上不建议显示数值
    )

    # 调整字体大小和旋转
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation = 90, size = 8)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation = 0, size = 8)

    g.fig.suptitle('Hierarchical Clustered Heatmap of Task Prompt Similarity', fontsize = 20)

    # 4. 保存图像
    output_path = os.path.join(save_dir, 'task_prompt_clustermap.svg')
    plt.savefig(output_path, dpi = 200, bbox_inches = 'tight')
    plt.close()

    print(f"--- Clustered heatmap saved to {output_path} ---")

def get_species_from_task(task_name, species_list):
    """
    从任务名称中提取物种，并将'man'和'women'合并到'human'。
    """
    # --- 核心修改：优先处理特殊映射 ---
    if task_name.startswith('man_') or task_name.startswith('women_'):
        return 'human'

    # 然后再进行通用查找
    for species in species_list:
        # 在通用查找中跳过 'man' 和 'women'，因为它们已被处理
        if species in ['man', 'women']:
            continue
        if task_name.startswith(species.replace(" ", "_")):
            return species
    return 'other'

def visualize_atom_attributions(smiles_str, atom_scores, file_name):
    mol = Chem.MolFromSmiles(smiles_str)
    if not mol: return

    # 归一化分数以便映射到颜色
    norm = matplotlib.colors.Normalize(vmin = min(atom_scores), vmax = max(atom_scores))
    cmap = cm.get_cmap('bwr')  # Blue-White-Red colormap

    atom_colors = {}
    for i, score in enumerate(atom_scores):
        atom_colors[i] = cmap(norm(score))

    d = rdMolDraw2D.MolDraw2DCairo(400, 400)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms = range(len(atom_scores)),
                                       highlightAtomColors = atom_colors)
    d.FinishDrawing()
    d.WriteDrawingText(file_name)


def draw_molecule_with_atom_scores(smiles_str, atom_scores, file_name, cmap_name = 'bwr', title = ""):
    """
    一个通用的绘图函数，根据提供的原子分数高亮分子图。
    """
    mol = Chem.MolFromSmiles(smiles_str)
    if not mol:
        print(f"Error: RDKit could not parse SMILES: {smiles_str}")
        return

    AllChem.Compute2DCoords(mol)

    if mol.GetNumAtoms() != len(atom_scores):
        print(f"Error in drawing: Atom count ({mol.GetNumAtoms()}) and score count ({len(atom_scores)}) mismatch.")
        return

    flat_scores = atom_scores.flatten()
    if len(flat_scores) == 0:
        print("Warning: atom_scores is empty, skipping visualization.")
        return

    if cmap_name in ['bwr', 'coolwarm', 'seismic', 'RdBu_r']:
        abs_max = max(abs(min(flat_scores)), abs(max(flat_scores)))
        norm = colors.Normalize(vmin = -abs_max, vmax = abs_max)
    else:
        norm = colors.Normalize(vmin = min(flat_scores), vmax = max(flat_scores))

    cmap = cm.get_cmap(cmap_name)
    atom_colors = {i: cmap(norm(score)) for i, score in enumerate(flat_scores)}

    d = rdMolDraw2D.MolDraw2DCairo(500, 500)
    d.drawOptions().addAtomIndices = True
    d.drawOptions().circleAtoms = True

    # --- 核心修复：为 DrawString 提供坐标 ---
    if title:
        # MolDraw2D 的坐标系原点(0,0)在左下角
        # 我们在画布的左上方设置标题位置
        # d.width() 和 d.height() 可以获取画布尺寸
        pos = Point2D(d.width() * 0.05, d.height() * 0.9)

        # 将标题按换行符分割，逐行绘制
        for i, line in enumerate(title.split('\n')):
            d.DrawString(line, Point2D(pos.x, pos.y - i * 25))  # 每行向下偏移25个单位

    d.DrawMolecule(mol, highlightAtoms = range(len(flat_scores)), highlightAtomColors = atom_colors)

    d.FinishDrawing()
    d.WriteDrawingText(file_name)

def visualize_functional_group_attention(smiles_str, attention_map):
    """
    可视化官能团之间的聚合注意力。

    :param smiles_str: 分子的SMILES字符串。
    :param attention_map: 从模型中提取的注意力图 (形状: [heads, N, N])。
    """
    mol = Chem.MolFromSmiles(smiles_str)
    if not mol: return

    # 1. 定义并识别官能团 (这是一个示例，您可以自定义)
    functional_groups = {
        'Carboxylic Acid (-COOH)': '[CX3](=O)[OX2H1]',
        'Cyclohexene Ring': 'C1CCCCC=C1',
        'Gem-dimethyl': '[C;X4](C)(C)',
        # 可以添加更多 SMARTS...
    }

    group_indices = {}
    for name, smarts in functional_groups.items():
        patt = Chem.MolFromSmarts(smarts)
        hits = mol.GetSubstructMatches(patt)
        if hits:
            # 将所有匹配的原子索引合并到一个集合中，避免重复
            all_atoms_in_group = set()
            for hit in hits:
                all_atoms_in_group.update(hit)
            group_indices[name] = list(all_atoms_in_group)

    # 2. 聚合注意力分数
    # 我们以第0个注意力头为例
    attn_head_0 = attention_map[0]  # 形状: [N, N]
    group_names = list(group_indices.keys())
    num_groups = len(group_names)
    aggregated_attention = np.zeros((num_groups, num_groups))

    for i, name_i in enumerate(group_names):
        for j, name_j in enumerate(group_names):
            indices_i = group_indices[name_i]
            indices_j = group_indices[name_j]
            # 提取子矩阵并求和
            sub_matrix = attn_head_0[np.ix_(indices_i, indices_j)]
            aggregated_attention[i, j] = sub_matrix.sum()

    # 3. 绘制热图
    df_agg = pd.DataFrame(aggregated_attention, index = group_names, columns = group_names)

    plt.figure(figsize = (10, 8))
    sns.heatmap(df_agg, annot = True, fmt = ".2f", cmap = "viridis")
    plt.title("Aggregated Attention between Functional Groups (Head 0)")
    plt.xticks(rotation = 45, ha = 'right')
    plt.yticks(rotation = 0)
    plt.tight_layout()
    plt.savefig("functional_group_attention.png")
    plt.close()

    print("--- Functional group attention heatmap saved. ---")

def visualize_attention_overlay(smiles_str, attention_map, query_atom_idx):
    """
    在2D分子图上叠加从一个查询原子出发的注意力连线。

    :param smiles_str: 分子的SMILES字符串。
    :param attention_map: 注意力图 (形状: [heads, N, N])。
    :param query_atom_idx: 我们关心的“查询”原子的索引。
    """
    mol = Chem.MolFromSmiles(smiles_str)
    if not mol: return

    # --- 核心修复：在绘图前生成2D坐标 ---
    AllChem.Compute2DCoords(mol)
    # 以第0个注意力头为例
    attn_head_0 = attention_map[0]  # 形状: [N, N]

    # 获取从查询原子到所有其他原子的注意力分数
    attention_scores = attn_head_0[query_atom_idx, :]

    # 归一化分数以便映射到颜色和线宽
    norm = colors.Normalize(vmin = attention_scores.min(), vmax = attention_scores.max())
    cmap = cm.get_cmap('bwr')

    d = rdMolDraw2D.MolDraw2DCairo(500, 500,)
    d.drawOptions().addAtomIndices = True  # 显示原子索引
    d.drawOptions().circleAtoms = False

    # 准备绘制分子，并高亮查询原子
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms = [query_atom_idx])

    # 叠加注意力连线
    mol_conf = mol.GetConformer()
    # 叠加注意力连线
    for i in range(mol.GetNumAtoms()):
        if i == query_atom_idx: continue

        score = attention_scores[i]
        if score < 0.1: continue

        color = cmap(norm(score))
        line_width = 0.5 + norm(score) * 2.5

        # --- 核心修改：创建并传递 Point2D 对象 ---

        # 1. 获取原子的 Point3D/Point2D 对象
        p1_obj = mol_conf.GetAtomPosition(query_atom_idx)
        p2_obj = mol_conf.GetAtomPosition(i)

        # 2. 从中提取 x, y 坐标，并创建 DrawLine 函数期望的 Point2D 对象
        p1_2d = Point2D(p1_obj.x, p1_obj.y)
        p2_2d = Point2D(p2_obj.x, p2_obj.y)

        # 3. 将 Point2D 对象传递给 DrawLine
        d.SetLineWidth(line_width)
        d.SetColour(color)
        d.DrawLine(p1_2d, p2_2d)

    d.FinishDrawing()
    d.WriteDrawingText("attention_overlay.png")

    print("--- Attention overlay visualization saved. ---")


def visualize_cross_task_attribution(
        trainer,
        preprocessed_data_dir,
        source_task,
        molecule_index,
        target_tasks,
        save_dir
):
    """
    对一个固定的分子，使用Captum归因分析，可视化不同目标任务的原子贡献度。
    """
    print(f"\n--- Starting Cross-Task Attribution Visualization ---")
    print(f"Source Molecule: Index {molecule_index} from task '{source_task}'")
    print(f"Target Tasks for Analysis: {target_tasks}")

    # 1. 加载唯一的源分子数据
    source_task_dir = os.path.join(preprocessed_data_dir, source_task)
    source_dataset = PreprocessedDatasetWrapper(source_task_dir)
    if molecule_index >= len(source_dataset):
        print(f"Error: Index {molecule_index} is out of bounds for source task '{source_task}'.")
        return

    single_mol_data = source_dataset[molecule_index]
    smiles = single_mol_data.smiles

    # 2. 准备一个可被复用的 collated_batch 和 Captum 工具
    collator = DataCollator(spatial_pos_max_clip = trainer.args.spatial_pos_clip, device = trainer.device)
    collated_batch = collator([single_mol_data])

    wrapped_model = ModelWrapperForLIG(trainer.model, collated_batch)
    layer_to_attribute = trainer.model.encoder.atoms_emb.atom_encoder
    lig = LayerIntegratedGradients(wrapped_model, layer_to_attribute)
    baseline = torch.zeros_like(collated_batch.x)

    # 3. 循环遍历所有目标任务，为每个任务生成一张归因图
    for target_task in target_tasks:
        if target_task not in trainer.model.task_name:
            print(f"Warning: Target task '{target_task}' not in model's known tasks. Skipping.")
            continue

        print(f"  - Calculating attribution for target task: '{target_task}'...")

        try:
            # a. 执行归因计算
            attributions = lig.attribute(
                inputs = collated_batch.x,
                baselines = baseline,
                additional_forward_args = (target_task, 'test'),
                n_steps = 200
            )

            # b. 聚合分数
            atom_contributions = attributions.sum(dim = (-1, -2)).squeeze().cpu().numpy()

            # c. 调用通用绘图函数
            output_filename = os.path.join(save_dir, f'attribution_mol_{molecule_index}_for_{target_task}.png')
            image_title = f"Attribution for Task:\n{target_task}"  # 为图片添加标题
            visualize_atom_attributions(smiles, atom_contributions, output_filename, )
            print(f"    -> Saved attribution map to {output_filename}")

        except Exception as e:
            print(f"    An error occurred while visualizing for task '{target_task}': {e}")

# def visualize_task_prompt_similarity(model, save_dir):
#     """
#     计算并可视化模型中所有任务提示向量(Prompts)之间的余弦相似度。
#
#     :param model: 训练好的、包含 prompts 参数的模型对象。
#     :param save_dir: 保存生成图像的目录路径。
#     """
#     print("\n--- Generating Task Prompt Similarity Heatmap ---")
#
#     # 1. 从模型中提取 prompts 和任务名称
#     # 这个路径是根据您提供的 Graphormer_prompt.py 文件确定的
#     try:
#         prompts_tensor = model.encoder.ema_prompts.squeeze(0).detach().cpu().numpy()
#         task_names = model.task_name
#     except AttributeError:
#         print("Error: Could not find 'encoder.prompts' or 'task_name' in the model. Aborting similarity visualization.")
#         return
#
#     # 2. 计算任务向量之间的余弦相似度矩阵
#     # similarity_matrix 的形状将是 [num_tasks, num_tasks]
#     similarity_matrix = cosine_similarity(prompts_tensor)
#
#     # 3. 使用 Seaborn 和 Matplotlib 绘制热图
#     # 对于任务很多的情况（如ToxAcute），需要较大的图尺寸和较小的字体
#     num_tasks = len(task_names)
#     fig_size = max(12, int(num_tasks * 0.5))  # 动态调整图像大小
#     font_size = max(6, int(120 / num_tasks))  # 动态调整字体大小
#
#     plt.figure(figsize = (fig_size, fig_size))
#     heatmap = sns.heatmap(
#         similarity_matrix,
#         xticklabels = task_names,
#         yticklabels = task_names,
#         annot = True,  # 在每个格子上显示数值
#         fmt = ".2f",  # 将数值格式化为两位小数
#         cmap = "coolwarm",  # 使用蓝-白-红发散色板，1(红色)代表最相似
#         linewidths = .5,
#         annot_kws = {"size": font_size}  # 动态设置注释字体大小
#     )
#
#     heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation = 45, horizontalalignment = 'right',
#                             fontsize = font_size + 1)
#     heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation = 0, fontsize = font_size + 1)
#
#     plt.title("Cosine Similarity between Task Prompts", size = fig_size)
#     plt.tight_layout()  # 自动调整布局以防标签重叠
#
#     # 4. 保存图像
#     output_path = os.path.join(save_dir, 'task_prompt_similarity.png')
#     plt.savefig(output_path, dpi = 200)  # 使用较高的DPI以保证清晰度
#     plt.close()  # 关闭图像，释放内存
#
#     print(f"--- Task prompt similarity heatmap saved to {output_path} ---")


def visualize_dynamic_task_attention(cross_attention_map, all_task_names, target_task, save_dir):
    """
    可视化对于一个特定分子，在预测一个目标任务时，其他所有任务的注意力权重。

    :param cross_attention_map: 交叉注意力的图 (形状: [heads, num_tasks, num_tasks])。
    :param all_task_names: 模型中所有任务的名称列表。
    :param target_task: 我们想分析的目标任务名称。
    :param save_dir: 保存图像的目录。
    """
    print(f"\n--- Generating Dynamic Task-Task Attention for target: '{target_task}' ---")

    try:
        task_idx = all_task_names.index(target_task)
    except ValueError:
        print(f"Error: Target task '{target_task}' not found in the model's task list.")
        return

    # 1. 提取第0个注意力头，以及目标任务作为查询的那一行注意力分数
    # attn_head_0 形状: [num_tasks, num_tasks]
    # 每一行 i 代表了任务 i 的 prompt 对所有其他任务上下文的注意力
    attn_head_0 = cross_attention_map[0]
    attention_scores = attn_head_0[task_idx, :]  # 形状: [num_tasks,]

    # 2. 创建 DataFrame 以便绘制条形图
    df = pd.DataFrame({
        'Task': all_task_names,
        'Attention Score': attention_scores
    }).sort_values(by = 'Attention Score', ascending = False)

    # 3. 绘制条形图
    plt.figure(figsize = (12, 18))
    sns.barplot(x = 'Attention Score', y = 'Task', data = df, palette = 'viridis')

    plt.title(f"Dynamic Attention to Other Tasks for Predicting '{target_task}'")
    plt.xlabel("Attention Score")
    plt.ylabel("Source Task Context")
    plt.tight_layout()

    output_path = os.path.join(save_dir, f'dynamic_task_attention_{target_task}.png')
    plt.savefig(output_path, dpi = 150)
    plt.close()

    print(f"--- Dynamic task attention bar chart saved to {output_path} ---")


def visualize_species_similarity_aggregated(model, save_dir, args):
    """
    计算并可视化物种之间的平均相似度。
    """
    print("\n--- Generating Aggregated Heatmap of Prompt Similarity by Species ---")

    # 1. 提取所需数据
    try:
        prompts = model.encoder.ema_prompts.squeeze(0).detach().cpu().numpy()
        task_names = np.array(model.task_name)
        species_list = args.species_list
    except AttributeError:
        print("Error: Could not retrieve necessary attributes from the model.")
        return

    # 2. 为每个任务生成物种标签 (使用我们之前修正过的、能合并human的函数)
    # 假设 get_species_from_task 函数已在您的文件中定义
    task_species = np.array([get_species_from_task(tn, species_list) for tn in task_names])

    # 获取合并后的、唯一的物种列表并排序
    unique_species = sorted(list(set(task_species)))
    num_species = len(unique_species)

    # 3. 计算完整的 [59, 59] 任务相似度矩阵
    similarity_matrix = cosine_similarity(prompts)

    # 4. 聚合计算物种间的平均相似度
    agg_matrix = np.zeros((num_species, num_species))
    for i in range(num_species):
        for j in range(num_species):
            species_i = unique_species[i]
            species_j = unique_species[j]

            # a. 找到分别属于这两个物种的所有任务的索引
            indices_i = np.where(task_species == species_i)[0]
            indices_j = np.where(task_species == species_j)[0]

            # b. 从完整的相似度矩阵中，根据索引提取出对应的子矩阵
            sub_matrix = similarity_matrix[np.ix_(indices_i, indices_j)]

            # c. 计算这个子矩阵的平均值，作为物种i和物种j之间的平均相似度
            if sub_matrix.size > 0:
                agg_matrix[i, j] = np.mean(sub_matrix)

    # 5. 使用 Pandas 和 Seaborn 绘制热图
    df_agg = pd.DataFrame(agg_matrix, index = unique_species, columns = unique_species)

    plt.figure(figsize = (14, 12))
    sns.heatmap(
        df_agg,
        annot = True,  # 在格子上显示数值
        fmt = ".3f",  # 将数值格式化为三位小数
        cmap = "viridis",  # 使用视觉上更清晰的色板
        linewidths = .5,
        annot_kws = {"size": 12}
    )
    plt.title("Aggregated Cosine Similarity between Species Prompts", size = 16)
    plt.xticks(rotation = 45, ha = 'right', size = 12)
    plt.yticks(rotation = 0, size = 12)
    plt.tight_layout()

    output_path = os.path.join(save_dir, 'species_aggregated_similarity.png')
    plt.savefig(output_path, dpi = 200)
    plt.close()

    print(f"--- Aggregated species heatmap saved to {output_path} ---")


def get_ordered_species_palette(unique_species_from_data):
    """
    根据预定义的生物学顺序，为物种生成一个有序的调色板。
    此版本会自动处理 'man' 和 'women' 到 'human' 的合并。
    """
    # 1. 定义一个符合生物学逻辑的物种主顺序
    master_species_order = [
        # 人类相关
        'human', 'man', 'women',  # 仍然保留man/women以确定排序位置
        # 常见哺乳动物
        'mammal (species unspecified)',
        'dog', 'cat',
        'rabbit', 'guinea pig', 'rat', 'mouse',
        # 鸟类
        'bird-wild', 'duck', 'quail', 'chicken',
        # 两栖类
        'frog'
    ]

    # --- 核心修改：在排序前进行合并 ---

    # a. 创建一个重映射后的物种集合，例如 {'human', 'dog', 'rat', ...}
    remapped_species = set()
    for species in unique_species_from_data:
        if species in ['man', 'women']:
            remapped_species.add('human')
        else:
            remapped_species.add(species)

    # b. 根据主顺序，对合并后的物种列表进行排序
    order_map = {species: i for i, species in enumerate(master_species_order)}
    sorted_species = sorted(list(remapped_species), key = lambda s: order_map.get(s, 999))

    # c. 为这个排好序的、合并后的物种列表生成调色板
    palette = sns.color_palette("husl", n_colors = len(sorted_species))

    # d. 创建并返回最终的颜色映射字典
    species_color_map = {species: color for species, color in zip(sorted_species, palette)}

    return species_color_map


def visualize_prompt_polar_map(model, save_dir, args):

    """

    角度(θ): 每个prompt向量与均值向量的夹角。
    半径(r): 每个prompt向量到均值向量的欧式距离。
    """
    print("\n--- Generating Polar Map via Direct High-Dimensional Calculation ---")

    # 1. 提取数据
    try:
        all_data = model.encoder.ema_prompts.squeeze(0).detach().cpu().numpy()
        task_names = model.task_name
        species_list = args.species_list
    except AttributeError:
        print("Error: Could not find necessary attributes in the model.")
        return

    # --- 核心计算部分 ---

    # 2. 计算角度 θ
    reference_vector = np.mean(all_data, axis=0)
    dot_products = np.dot(all_data, reference_vector)
    reference_norm = np.linalg.norm(reference_vector)
    data_norms = np.linalg.norm(all_data, axis=1)

    # 防止除以零
    data_norms[data_norms == 0] = 1e-9

    cosine_values = dot_products / (data_norms * reference_norm)
    thetas = np.arccos(np.clip(cosine_values, -1.0, 1.0)) # 角度，单位是弧度

    # 3. 计算半径 r (改进后的算法)
    # r = 每个向量到均值向量的欧式距离
    radii = np.linalg.norm(all_data - reference_vector, axis=1)

    # 4. 准备绘图
    def get_species_from_task(task_name, species_list):
        if not species_list: return 'unknown'
        for species in species_list:
            if task_name.startswith(species.replace(" ", "_")): return species
        return 'other'
    species_labels = [get_species_from_task(t, species_list) for t in task_names]

    # 5. 绘图 (与之前的绘图代码类似)
    fig, ax = plt.subplots(figsize=(18, 18), subplot_kw={'projection': 'polar'})

    unique_species = sorted(list(set(species_labels)))

    # 假设 get_ordered_species_palette 已定义
    species_color_map = get_ordered_species_palette(unique_species)
    colors = [species_color_map.get(s) for s in species_labels]

    sizes = 50 + (radii / np.max(radii) * 400) if np.max(radii) > 0 else 50

    ax.scatter(thetas, radii, c=colors, s=sizes, alpha=0.7, edgecolors='black')

    ax.set_title("Polar Map of Task Prompts (Direct Calculation)", size=24, pad=40)
    ax.grid(True)
    # ax.set_yticklabels([]) # 半径有真实意义，可以显示刻度

    legend = ax.legend(
        handles=[plt.Line2D([0], [0], marker='o', color='w', label=species, markerfacecolor=color, markersize=15) for species, color in species_color_map.items()],
        title="Species", loc='upper left', bbox_to_anchor=(1.1, 1.05), fontsize=16, title_fontsize=18
    )

    output_path = os.path.join(save_dir, 'task_prompt_polar_map_direct.png')
    plt.savefig(output_path, dpi=200, bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close()

    print(f"--- Direct calculation polar map saved to {output_path} ---")


class PromptAttributionWrapper(torch.nn.Module):
    def __init__(self, model, mol_rep_tensor):
        super().__init__()
        self.model = model
        self.mol_rep = mol_rep_tensor

    def forward(self, prompts_tensor):
        # 这个forward方法接收一个由Captum传入的、插值后的prompts张量
        # 模型的 mol_rep 部分是固定的

        # 调用模型的后半部分：Task Prompt Encoder
        # 注意：这里的 mol_rep 需要根据 task_prompt_encoder 的期望输入进行扩展
        B, T, H = prompts_tensor.shape
        expanded_mol_rep = self.mol_rep.expand(B, T, H)

        task_encoder_output = self.model.encoder.task_prompt_encoder(expanded_mol_rep, prompts_tensor)

        # 应用所有任务的解码器，得到所有任务的预测值
        # task_encoder_output 在模型内部已经被转置为 [Tasks, Batch, Hidden]
        predictions_list = [
            self.model.decoders[task_name](task_encoder_output[i])
            for i, task_name in enumerate(self.model.task_name)
        ]

        # Captum需要一个张量作为输出，我们将所有预测值堆叠起来
        return torch.stack(predictions_list, dim = 0)


def visualize_prompt_contribution(model, collated_batch, target_task, save_dir):
    """
    使用Captum计算并可视化，在预测一个目标任务时，所有任务的Prompts分别做出的贡献。

    :param model: 训练好的模型对象。
    :param collated_batch: 包含单个目标分子的、已经collated的批处理数据。
    :param target_task: 我们要分析其预测结果的目标任务名称。
    :param save_dir: 保存图像的目录。
    """
    print(f"\n--- Generating Prompt Contribution Attribution for target: '{target_task}' ---")

    # 2. 准备归因所需的各个部分
    model.eval()
    with torch.no_grad():
        # a. 先计算出当前分子的固定表示向量
        mol_rep_tensor = model.encoder.atoms_emb(collated_batch)  # Shape: [1, hidden_dim]
        # b. 提取原始的、未经插值的prompts
        original_prompts = model.encoder.prompts  # Shape: [1, num_tasks, hidden_dim]

    # c. 实例化包装器
    wrapper = PromptAttributionWrapper(model, mol_rep_tensor)

    # d. 实例化积分梯度
    ig = IntegratedGradients(wrapper)

    # e. 准备基线（通常是零）
    baseline_prompts = torch.zeros_like(original_prompts)

    # f. 找到目标任务的索引
    try:
        target_task_idx = model.task_name.index(target_task)
    except ValueError:
        print(f"Error: Target task '{target_task}' not found.")
        return

    # 3. 执行归因计算
    print(f"Calculating prompt attributions for target task '{target_task}'...")
    attributions = ig.attribute(
        inputs = original_prompts,
        baselines = baseline_prompts,
        target = target_task_idx,  # 告诉Captum我们关心哪个输出
        n_steps = 200
    )

    # 4. 聚合与可视化
    # attributions 的形状是 [1, num_tasks, hidden_dim]
    # 我们将每个任务prompt的所有维度贡献度相加，得到每个任务的总贡献
    prompt_contributions = attributions.sum(dim = -1).squeeze().cpu().detach().numpy()

    df = pd.DataFrame({
        'Task': model.task_name,
        'Contribution Score': prompt_contributions
    }).sort_values(by = 'Contribution Score', ascending = False)

    plt.figure(figsize = (12, 18))
    sns.barplot(x = 'Contribution Score', y = 'Task', data = df, palette = 'coolwarm')

    plt.title(f"Contribution of Each Task Prompt to Prediction of '{target_task}'")
    plt.xlabel("Attribution Score (Contribution)")
    plt.ylabel("Source Task Prompt")
    plt.axvline(0, color = 'grey', linewidth = 0.8, linestyle = '--')  # 在0点画一条垂直线
    plt.tight_layout()

    output_path = os.path.join(save_dir, f'prompt_contribution_{target_task}.png')
    plt.savefig(output_path, dpi = 150)
    plt.close()

    print(f"--- Prompt contribution bar chart saved to {output_path} ---")

class Visualizer(object):
    def __init__(self, trainer):
        self.trainer = trainer
    def visualize_atom_contribution(self, args_for_viz, source_task_for_mol, mol_idx, tasks_to_analyze):
        task_name_viz = args_for_viz.task_for_viz
        mol_index_viz = args_for_viz.mol_index_for_viz
        if not task_name_viz or task_name_viz not in self.trainer.task_name:
            print(f"Error: Task '{task_name_viz}' not found or not specified. Use --task_for_viz.")
            return
        task_specific_data_dir = os.path.join(args_for_viz.preprocessed_data_dir, task_name_viz)
        full_dataset = PreprocessedDatasetWrapper(task_specific_data_dir)
        if mol_index_viz >= len(full_dataset):
            print(
                f"Error: --mol_index_for_viz={mol_index_viz} is out of bounds for task '{task_name_viz}' (size: {len(full_dataset)}).")
            return
        single_mol_data = full_dataset[mol_index_viz]
        smiles = single_mol_data.smiles

        # 准备Captum
        collator = DataCollator(spatial_pos_max_clip=self.trainer.args.spatial_pos_clip, device=self.trainer.device)
        collated_batch = collator([single_mol_data])


        wrapped_model = ModelWrapperForLIG(self.trainer.model, collated_batch)
        layer_to_attribute = self.trainer.model.encoder.atoms_emb.atom_encoder
        lig = LayerIntegratedGradients(wrapped_model, layer_to_attribute)
        input_features = collated_batch.x
        baseline = torch.zeros_like(input_features)

        # 计算归因分数
        print(f"Calculating attributions for molecule {mol_index_viz} from task {task_name_viz}...")

        attributions = lig.attribute(input_features,
                                     baselines = baseline,
                                     additional_forward_args = (task_name_viz, 'test'),

                                     n_steps = 200)
        # 归因结果的形状是嵌入层输出的形状: [batch, n_nodes, n_features, hidden_dim]
        # 我们需要将每个原子所有特征在所有维度上的贡献度加起来
        # squeeze(0) 去掉 batch 维度
        # 将每个原子的特征贡献度相加， 得到每个原子的总贡献度
        atom_contributions = attributions.sum(dim = (-1,-2)).squeeze().cpu().numpy()
        print(f"Shape of atom_contributions: {atom_contributions.shape}")
        # 可视化原子贡献归因
        output_filename = os.path.join(self.trainer.save_path or '.', f'atom_contribution_{task_name_viz}_{mol_index_viz}.png')
        visualize_atom_attributions(smiles, atom_contributions, output_filename)
        print(f"--- Atom contribution visualization saved to {output_filename} ---")

        #可视化官能团依赖
        print("\n--- Preparing synchronized data for attention visualization ---")
        self.trainer.model.eval()
        with torch.no_grad():
            # 使用我们为 Captum 准备好的 collated_batch，为目标分子重新运行一次模型
            self.trainer.model(collated_batch, task_name_viz, 'test')
            # 官能团注意力图
        target_attention_layer = self.trainer.model.encoder.atoms_emb.layers[-1].self_attention
        attention_map_tensor = target_attention_layer.attention_map.squeeze(0).cpu().numpy()
        num_atoms = single_mol_data.x.shape[0]
        print(
            f"Verifying synchronized shapes: Molecule Atoms={num_atoms}, Attention Map Shape={attention_map_tensor.shape}")
        if num_atoms != attention_map_tensor.shape[1]-1:
            print("Error: Shape mismatch between molecule and attention map! Aborting visualization.")
            return
        # 调用新的可视化函数
        visualize_functional_group_attention(smiles, attention_map_tensor)

        # 假设羧基中的某个氧原子索引是 10
        visualize_attention_overlay(smiles, attention_map_tensor, query_atom_idx = self.trainer.args.query_atom_idx)
        # 原子-任务交叉注意力可视化，关注同一个分子在不同任务下关注的不同部位
        # 2. 不同任务关注的分子归因
        visualize_cross_task_attribution(
            trainer = self.trainer,
            preprocessed_data_dir = args_for_viz.preprocessed_data_dir,
            source_task = source_task_for_mol,
            molecule_index = mol_idx,
            target_tasks = tasks_to_analyze,
            save_dir = self.trainer.save_path or '.'
        )


    def visualize_prompt(self, perplexity = 40):
        # 1. 提取 EMA prompts
        # shape [1, num_tasks, hidden_dim]
        prompts = self.trainer.model.encoder.ema_prompts.squeeze(0).cpu().numpy()

        # 2. 将任务映射到物种
        species_list = self.trainer.args.species_list
        if not species_list:
            print("Warning: --visualize_prompts is set, but no species list is defined for this dataset. Skipping.")
            return



        task_labels = self.trainer.task_name
        species_labels = [get_species_from_task(t, species_list) for t in task_labels]

        # 3. 执行 t-SNE 降维
        # Perplexity 建议小于样本数。任务数通常不多，设一个较小的值。
        perplexity_val = min(perplexity, len(task_labels) - 1)
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = perplexity_val, n_iter = 1000, random_state = 42)
        tsne_results = tsne.fit_transform(prompts)

        # 4. 使用 Seaborn 和 Matplotlib 绘图
        df = pd.DataFrame()
        df['tsne-1'] = tsne_results[:, 0]
        df['tsne-2'] = tsne_results[:, 1]
        df['species'] = species_labels
        df['task'] = task_labels

        def shorten_label(label):
            # 示例缩写规则，您可以根据您的任务名称格式自定义
            parts = label.replace('-', '_').split('_')
            if len(parts) >= 3:
                # 例如 'mouse_intraperitoneal_LD50' -> 'mouse_ip_LD50'
                route = parts[1]
                if route == 'intraperitoneal': route = 'ip'
                if route == 'intravenous': route = 'iv'
                if route == 'oral': route = 'o'
                if route == 'subcutaneous': route = 'sc'
                return f"{parts[0]}_{route}_{parts[-1]}"
            return label  # 如果不匹配规则，返回原标签

        df['short_task'] = df['task'].apply(shorten_label)
        plt.figure(figsize = (16, 8), dpi = 300)
        ax = sns.scatterplot(
            x = "tsne-1", y = "tsne-2",
            hue = "species",
            palette = sns.color_palette("hls", len(df['species'].unique())),
            data = df,
            legend = "full",
            alpha = 0.9,
            s = 200  # 点的大小
        )

        # 为每个点添加任务名称注释
        # for i in range(df.shape[0]):
        #     plt.text(x = df['tsne-1'][i] + 0.1, y = df['tsne-2'][i] + 0.1, s = df['task'][i],
        #              fontdict = dict(color = 'black', size = 8))
        texts = []
        for i in range(df.shape[0]):
            texts.append(plt.text(df['tsne-1'][i], df['tsne-2'][i], df['short_task'][i], size = 10))
        # a-SNE 智能调整文本位置
        # expand_points 增加点周围的填充
        # arrowprops 设置从文本指向点的箭头样式
        adjust_text(texts,
                    ax = ax,
                    expand_points = (1.2, 1.2),
                    arrowprops = dict(arrowstyle = "-", color = 'gray', lw = 0.5))

        plt.title('t-SNE Visualization of Task Prompts by Species')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')

        # 保存图像
        output_filename = 'task_prompts_tsne.svg'
        if self.trainer.save_path:
            output_filename = os.path.join(self.trainer.save_path, output_filename)

        plt.savefig(output_filename)
        print(f"--- Visualization saved to {output_filename} ---")
        plt.close()

        # 绘制相似度热图
        if hasattr(self.trainer.model.encoder, 'ema_prompts'):
            visualize_task_prompt_similarity(self.trainer.model, self.trainer.save_path or '.')
        else:
            print("Model does not have prompts, skipping similarity visualization.")

        # 另一种极坐标可视化图
        # if hasattr(self.trainer.model.encoder, 'ema_prompts'):
        #     visualize_prompt_polar_map(self.trainer.model, self.trainer.save_path or '.', self.trainer.args)
        # else:
        #     print("Model does not have prompts, skipping polar map visualization.")

    def visualize_predictions_histograms(self):
        all_preds = []
        all_gts = []
        all_species = []
        species_list = self.trainer.args.species_list
        if not species_list:
            print("Warning: --visualize_predictions is set, but no species list is defined. Skipping species coloring.")

        def get_species_from_task(task_name, species_list):
            if not species_list: return 'all_tasks'
            for species in species_list:
                if task_name.startswith(species.replace(" ", "_")):
                    return species
            return 'unknown'
            # 遍历所有任务，收集缓存的预测和真实值
        for task_name in self.trainer.task_name:
            if task_name in self.trainer.meter.cache_result and self.trainer.meter.cache_result[task_name]['pred']:
                preds_tensor = torch.cat(self.trainer.meter.cache_result[task_name]['pred'], dim = 0)
                gts_tensor = torch.cat(self.trainer.meter.cache_result[task_name]['gts'], dim = 0)

                species = get_species_from_task(task_name, species_list)

                all_preds.extend(preds_tensor.cpu().numpy().flatten())
                all_gts.extend(gts_tensor.cpu().numpy().flatten())
                all_species.extend([species] * len(preds_tensor))

        if not all_preds:
            print("Warning: No prediction data found to visualize.")
        else:
            # 创建 DataFrame
            df_preds = pd.DataFrame({
                'Prediction': all_preds,
                'Ground Truth': all_gts,
                'Species': all_species
            })

            # 绘图
            if self.trainer.args.visualize_predictions:
                print("--- Generating Prediction vs. Ground Truth Scatter Plot ---")
                plt.figure(figsize = (12, 12), dpi = 300)
                g = sns.scatterplot(data = df_preds, x = 'Prediction', y = 'Ground Truth', hue = 'Species',
                                    palette = 'hls', alpha = 0.6, s = 50)

                # 绘制 y=x 对角线作为参考
                min_val = min(df_preds['Prediction'].min(), df_preds['Ground Truth'].min())
                max_val = max(df_preds['Prediction'].max(), df_preds['Ground Truth'].max())
                g.plot([min_val, max_val], [min_val, max_val], 'r--', lw = 2, label = 'Perfect Prediction (y=x)')

                plt.title('Prediction vs. Ground Truth for All Test Tasks', fontsize = 22)
                plt.xlabel('Predicted Values', fontsize = 18)
                plt.ylabel('Ground Truth', fontsize = 18)
                plt.ylabel('Ground Truth Values')
                # 增大坐标轴刻度数字的字体
                plt.xticks(fontsize = 14)
                plt.yticks(fontsize = 14)
                plt.legend(fontsize=14, title_fontsize=16)
                plt.grid(True)

                # 保存图像
                output_filename = 'predictions_vs_truth_scatter.png'
                if self.trainer.save_path:
                    output_filename = os.path.join(self.trainer.save_path, output_filename)

                plt.savefig(output_filename)
                print(f"--- Prediction visualization saved to {output_filename} ---")
                plt.close()
            elif self.trainer.args.visualize_histograms:
                print("--- Generating Prediction vs. Ground Truth Distribution Histograms ---")
                # 为了用 seaborn 方便地绘图，我们先将数据"融合" (melt)
                df_melted = df_preds.melt(id_vars = ['Species'], value_vars = ['Prediction', 'Ground Truth'],
                                          var_name = 'Value Type', value_name = 'Value')

                # 图一：总体分布对比
                plt.figure(figsize = (12, 12), dpi = 300)
                sns.histplot(data = df_melted, x = 'Value', hue = 'Value Type', kde = False, bins = 50, palette = 'hls')
                plt.title('Overall Distribution of Predicted vs. Ground Truth Values', fontsize = 22)
                plt.xlabel('Value', fontsize = 18)
                plt.ylabel('Count', fontsize = 18)
                hist_output_filename = os.path.join(self.trainer.save_path or '.', 'predictions_vs_truth_overall_histogram.png')
                plt.savefig(hist_output_filename)
                print(f"--- Overall histogram saved to {hist_output_filename} ---")
                plt.close()

                # 图二：按物种分类的分布对比
                g_displot = sns.displot(data = df_melted, x = 'Value', hue = 'Value Type', col = 'Species',
                                        col_wrap = 4, kind = 'kde', fill = True, common_norm = False, palette = 'hls')
                g_displot.fig.suptitle('Distribution of Predicted vs. Ground Truth Values by Species', y = 1.02)
                # g_displot.fig.tight_layout(rect = [0, 0, 1, 0.95])
                g_displot.set(xlabel = None)
                displot_output_filename = os.path.join(self.trainer.save_path or '.',
                                                       'predictions_vs_truth_species_histogram.png')
                plt.savefig(displot_output_filename)
                print(f"--- Per-species histogram saved to {displot_output_filename} ---")
                plt.close()

                # 另一种按物种的直方图
                visualize_species_similarity_aggregated(self.trainer.model,self.trainer.save_path, self.trainer.args)
                visualize_task_prompt_similarity(self.trainer.model,self.trainer.save_path)