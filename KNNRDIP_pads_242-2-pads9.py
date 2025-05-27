import copy
import sys

import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, make_scorer, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit, GridSearchCV
import csv
import os
import pandas as pd
import json
import itertools
import datetime
import argparse  # 添加这个导入语句
import csv
import os

# 获取当前日期
now = datetime.datetime.now()
# 格式化月和日
month_day = now.strftime("%m%d")  # 月和日格式为"MMdd"，例如"0509"
# 设置日志的配置，美包含INFO标签
logging.basicConfig(
    filename=rf"./{month_day}_model_training.log",
    level=logging.INFO,
    format='%(message)s',
    filemode='a'
)


def save_confusion_matrix(all_labels, all_preds, activity_id):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    import datetime
    import os

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 设置类别名称
    class_names = ['HC', 'Mild PD', 'Moderate PD', 'Severe PD']

    # 创建一个画布
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制混淆矩阵热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)

    # 设置标题和标签
    ax.set_title(f'{activity_id} Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # 获取分类报告
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # 在每行最后一个格子的右上角显示精度、召回率和F1分数
    for i, label in enumerate(class_names):
        precision = report[label]['precision']
        recall = report[label]['recall']
        f1_score = report[label]['f1-score']

        # 将文本放在每行最后一个格子的右上角，右对齐
        ax.text(4.0, i + 0.0, f'Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1_score:.2f}',
                ha='right', va='top', fontsize=8, color='black')

    # 获取当前时间并格式化为字符串，命名文件
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'lgbm_confusion_matrix_{activity_id}_{current_time}.svg'

    # 保存混淆矩阵为SVG文件
    output_dir = "./confusion_matrices"
    os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在，创建它
    plt.savefig(os.path.join(output_dir, filename), format='svg')

    # 显示图像
    plt.show()


def tsne_visualization_minimal_with_frame(train_X, train_Y, new_test_X, new_test_Y):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    combined_X = np.vstack((train_X, new_test_X))
    # combined_Y = np.concatenate((train_Y, new_test_Y))
    # 使用 t-SNE 进行降维（保持原始数据尺度）
    tsne = TSNE(random_state=0)
    X_tsne = tsne.fit_transform(combined_X)
    # 绘制训练数据和测试数据
    plt.figure(figsize=(8, 6))
    # Adjust legend font size
    # plt.legend(loc='best', frameon=True, prop={'size': 20})
    plt.rcParams['legend.fontsize'] = 16

    plt.scatter(X_tsne[:len(train_X), 0], X_tsne[:len(train_X), 1], c='red', label='Train Data',
                s=10)
    # 绘制测试数据
    plt.scatter(X_tsne[len(train_X):, 0], X_tsne[len(train_X):, 1], c='blue', label='Test Data',
                s=10)
    # 保留框但去除刻度和标签
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    # plt.legend(loc='best', frameon=True)
    # 调整图例
    plt.legend(
        loc='best',
        frameon=True,
        markerscale=2,  # 调整图例标记的大小
        # handlelength=3,  # 调整图例中标记的长度
        # borderpad=1  # 调整图例边框到标记和文本之间的距离
    )
    from datetime import datetime
    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 将当前时间加到文件名中
    filename = f'LGBM-4_class-tsne_visualization_{current_time}.svg'
    # 保存文件
    plt.savefig(filename, format='svg')
    # 显示图像
    plt.show()


class PDClassifier:
    def __init__(self, data_path, activity_id, severity_mapping=None):
        if severity_mapping is None:
            # severity_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}  # 映射关系（如果改变数据集，注意修改这里）
            severity_mapping = {1: 0, 2: 1}  # 映射关系 pads修改处
        self.data_path = data_path  # 手工特征文件路径
        self.activity_id = activity_id  # 从手工特征文件中选取指定activity_id对应的数据
        self.severity_mapping = severity_mapping  # 映射关系
        self.PD_data = self.load_and_preprocess_data()  # 预处理数据文件 #====读取指定activity_id活动的数据

        # loaded_df = pd.read_csv('subject_grouping_4_class.csv')
        loaded_df = pd.read_csv('subject_grouping_pads_2_class.csv')  # pads修改处
        # 根据Activity_id匹配正确的分组结果
        print("当前的args.activity_id:" + str(self.activity_id))
        matched_row = loaded_df[loaded_df['dataset'] == self.activity_id]  # ====读取分组id信息
        if not matched_row.empty:
            # 从JSON格式还原为Python字典
            restored_groups = json.loads(matched_row['groups'].iloc[-1])
            # 将还原的分组结果展示为Python列表格式
            restored_lists = {int(k): v for k, v in restored_groups.items()}
            print('还原的分组结果:', restored_lists)
        else:
            raise ValueError(f'未找到Activity_id为 {self.activity_id} 的分组结果')
        # 假设字典的键是顺序数字的字符串，按键顺序将字典转换为列表
        restored_list = [restored_groups[str(i)] for i in range(len(restored_groups))]
        # 使用列表推导式将所有元素从 float 转换为 int
        restored_list = [[int(item) for item in sublist] for sublist in restored_list]
        self.fold_groups = restored_list
        self.bag_data_dict, self.patient_ids, self.fold_groups = self.group_data(self.fold_groups)  # ========预备组织数据

    # 返回指定对应acitivity_id的数据
    def load_and_preprocess_data(self):  # 如果改变数据集，注意修改这里
        data = pd.read_csv(self.data_path)  # 筛选出数据标签，选择出对应数据
        # 检查 self.activity_id 是否可以转换为数字，并且是否在 1 到 16 之间
        try:
            activity_id_num = int(self.activity_id)
            if 1 <= activity_id_num <= 14:  # pads 修改处
                data = data.loc[data['activity_label'] == activity_id_num]
            else:
                raise AssertionError(f"activity_id {self.activity_id} is not in the range 1 to 16.")
        except ValueError:
            pass
        print(f"data:{data}, activity_id: {self.activity_id}")
        data['Severity_Level'] = data['Severity_Level'].map(self.severity_mapping)
        data.dropna(inplace=True)
        # =====pads修改处=====
        # 将 Severity_Level 列转换为整数类型
        data['Severity_Level'] = data['Severity_Level'].astype(int)
        data['activity_label'] = data['activity_label'].astype(int)
        data['PatientID'] = data['PatientID'].astype(int)
        # =====pads修改处=====
        return data

    # 返回病人为单位的数据
    def group_data(self, fold_groups):
        grouped = self.PD_data.groupby(['PatientID', 'Severity_Level'])
        # 每个病人特征数据为list中一个元素，每个元素是一个二维ndarray，存储该病人手工特征数据，例如一个元素shape为(24, 220)
        # 表示该二维ndarry中，该病人有24个滑窗（活动片段），每个滑窗对应220维特征
        # 每个病人的标签为list中一个元素，每个元素是一维ndarray元素，存储该病人滑动窗口的标签，例如一个元素shape为(24,)
        # 表示该一维ndarray中，该病人24个滑动窗口（活动片段），每个滑窗对应1个标签，有24个标签
        bag_data_dict = dict()
        patient_ids = []
        for (patient_id, _), group in grouped:
            # 检查 self.activity_id 是否可以转换为数字，并且是否在 1 到 16 之间
            try:
                activity_id_num = int(self.activity_id)
                if 1 <= activity_id_num <= 14:  # 如果改变数据集，注意修改这里 pads修改处
                    bag_data = np.array(group.iloc[:, :-3])  # 获取特征数据，如果改变数据集，注意修改这里
                else:
                    raise AssertionError(f"activity_id {self.activity_id} is not in the range 1 to 14.")
            except ValueError:
                # bag_data = np.array(group.iloc[:, :-2])  # 获取特征数据 pads修改处
                raise ValueError("pads")
            bag_data_instance_label = (np.array(group.loc[:, 'Severity_Level']))  # 获取标签数据
            patient_ids.append(patient_id)
            if patient_id not in bag_data_dict.keys():
                bag_data_dict[patient_id] = {"pid": patient_id, "bag_data": bag_data,
                                             "bag_data_label": bag_data_instance_label[0],
                                             "bag_data_instance_label": bag_data_instance_label
                                             }
        for fold_num, fold_group in enumerate(fold_groups):
            print(f"第{fold_num}组， {len(fold_group)}人")
        return bag_data_dict, patient_ids, fold_groups

    def limit_instances(self, bag_data_dict, test_env, limited_instances=18):
        train_ids = []
        for num, fold_ids in enumerate(self.fold_groups):
            if num != test_env:
                train_ids.extend(fold_ids)
        test_ids = self.fold_groups[test_env]
        all_ids = train_ids + test_ids
        # 检查每个 pid 的数据是否满足条件
        for pid in all_ids:
            if len(self.bag_data_dict[pid]["bag_data"]) < limited_instances:
                raise ValueError(f"Data for participant {pid} is less than {limited_instances} rows.")
            else:
                bag_data_dict[pid]["bag_data"] = bag_data_dict[pid]["bag_data"][:limited_instances]
        return bag_data_dict

    def standardize_zscore(self, bag_data_dict, test_env):
        train_ids = []
        for num, fold_ids in enumerate(self.fold_groups):
            if num != test_env:
                train_ids.extend(fold_ids)
        test_ids = self.fold_groups[test_env]
        all_ids = train_ids + test_ids
        train_X = [bag_data_dict[pid]["bag_data"] for pid in train_ids]  # 这里写得有问题
        train_X = np.vstack(train_X)

        from sklearn.preprocessing import StandardScaler
        # 创建StandardScaler对象
        scaler = StandardScaler()
        # 使用train_X的均值和标准差来拟合并转换train_X
        scaler.fit(train_X)

        for pid in all_ids:
            bag_data_dict[pid]["bag_data"] = scaler.transform(bag_data_dict[pid]["bag_data"])
        return bag_data_dict

    def train_and_evaluate(self, test_env=None):
        k_num_list = [i for i in range(1, 17)]
        squared_sigma_list = [8 * (10 ** j) for j in range(0, 19)]
        cur_bag_data_dict = copy.deepcopy(self.bag_data_dict)
        # 对cur_data_dict限制实例个数
        cur_bag_data_dict = self.limit_instances(cur_bag_data_dict, test_env, 18)
        # 对cur_data_dict标准化
        # cur_bag_data_dict = self.standardize_zscore(cur_bag_data_dict, test_env)

        macro_f1_bst = 0
        accuracy_bst = 0

        for cur_squared_sigma in tqdm(squared_sigma_list, desc="Processing squared_sigma_list"):
            # 训练和测试数据集
            (train_ordered_full_map, train_ordered_full_map_label, test_ordered_full_map,
             test_ordered_full_map_label, min_cols) = (self.rdip_mapping(cur_bag_data_dict,
                                                                         test_env,
                                                                         cur_squared_sigma,
                                                                         instance_space='all'))
            for cur_k_num in tqdm(k_num_list, desc="Processing k_num_list"):
                # 准备好映射函数返回train和test，然后设置两层for循环，分别循环k_num和top_instance_num

                args = self.parse_args()
                knn = KNeighborsClassifier(n_neighbors=cur_k_num, weights='distance')
                # for cur_k_num in range(1, 17):
                # for m in tqdm(range(1, min_cols + 1, 1), desc="Processing m"):
                for m in range(1, min_cols + 1, 5):
                    total_pred_group_ls = []  # 记录每次验证的预测的病人病情标签
                    total_test_Y_group_ls = []  # 记录每次验证的真实的病人病情标签

                    mapped_train_X = train_ordered_full_map[:, :m]
                    mapped_train_Y = train_ordered_full_map_label
                    mapped_test_X = test_ordered_full_map[:, :m]
                    mapped_test_Y = test_ordered_full_map_label
                    # 使用训练好的模型，进行测试
                    knn.fit(mapped_train_X, mapped_train_Y)
                    y_pred = knn.predict(mapped_test_X)
                    # 将当前验证的病人级别的预测标签和真实标签记录
                    total_pred_group_ls.append(y_pred)
                    total_test_Y_group_ls.append(mapped_test_Y)

                    report_dict = classification_report(total_test_Y_group_ls[0], total_pred_group_ls[0], digits=4,
                                                        zero_division=0)
                    # print(report_dict)
                    report_dict_best = classification_report(total_test_Y_group_ls[0], total_pred_group_ls[0],
                                                             output_dict=True, digits=4, zero_division=0)
                    macro_f1 = report_dict_best['macro avg']['f1-score']
                    accuracy = report_dict_best['accuracy']

                    if macro_f1_bst <= macro_f1:
                        args.activity_id = self.activity_id
                        args.test_envs = test_env
                        args.k_num = cur_k_num
                        args.squared_sigma = cur_squared_sigma
                        args.m = m

                        macro_f1_bst = macro_f1
                        self.save_results_to_csv(args, report_dict_best)
                        print(report_dict)
                        if accuracy_bst == 0 or accuracy_bst <= accuracy:
                            accuracy_bst = accuracy
                    elif accuracy_bst <= accuracy:
                        args.activity_id = self.activity_id
                        args.test_envs = test_env
                        args.k_num = cur_k_num
                        args.squared_sigma = cur_squared_sigma
                        args.m = m

                        accuracy_bst = accuracy
                        self.save_results_to_csv(args, report_dict_best)
                        print(report_dict)
            # ======================可视化===================
            # tsne_visualization_minimal_with_frame(train_X, train_X, new_test_X, new_test_X)
            # save_confusion_matrix(test_Y_ls, y_pred_ls, self.activity_id)
            # ======================可视化===================

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Activity Classification with LSTM and CNN")
        # parser.add_argument('--activity_id', type=int, required=False, help="Activity ID for dataset selection")
        # parser.add_argument('--test_env', type=str, nargs='+',
        # default=['-1'], help="Test environment identifier list")
        args = parser.parse_args()
        return args

    def save_results_to_csv(self, args, report_best):
        # 获取要保存的CSV文件名
        # csv_filename = fr".\results_2_class\{args.activity_id}.csv"
        csv_filename = os.path.join(".", "RDIP_PADS_2_class", f"{args.activity_id}.csv")

        # 确定文件是否存在
        file_exists = os.path.isfile(csv_filename)

        # 准备要写入的数据，初始化为传入的args参数
        data_to_write = {
            'test_env': args.test_envs,
        }

        # 将classification_report的所有指标平展后加入要写入的数据字典中
        def flatten_and_format_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_and_format_dict(v, new_key, sep=sep).items())
                else:
                    # 如果是非support的指标，将每个指标乘以100并保留两位小数
                    if 'support' not in new_key and isinstance(v, (float, int)):
                        formatted_value = f"{v * 100:.2f}"
                    else:
                        # 对于support保持整数格式
                        formatted_value = f"{v:.0f}" if 'support' in new_key else v
                    items.append((new_key, formatted_value))
            return dict(items)

        # 将平展并格式化后的report_best内容加入到data_to_write
        formatted_report = flatten_and_format_dict(report_best)

        # 提取并将accuracy和macro avg相关指标放在test_env之后
        accuracy_metric = {'accuracy': formatted_report.pop('accuracy')}
        macro_metrics = {k: formatted_report.pop(k) for k in list(formatted_report.keys()) if 'macro avg' in k}

        # 更新数据顺序
        data_to_write.update(accuracy_metric)
        data_to_write.update(macro_metrics)

        # 将args的所有参数作为字典存储在结果数据中
        data_to_write['args'] = vars(args)

        # 更新剩余的formatted_report到data_to_write
        data_to_write.update(formatted_report)

        # 写入CSV文件
        with open(csv_filename, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_to_write.keys())
            # 如果文件不存在，则写入表头
            if not file_exists:
                writer.writeheader()
            # 写入数据
            writer.writerow(data_to_write)

    def rdip_mapping(self, bag_data_dict, test_env, squared_sigma, instance_space='all'):
        def similarity(B_i, x_k):
            squared_diff = np.square(B_i - x_k)
            sum_squared_diff = np.sum(squared_diff, axis=1)
            s_B_i_x_k_fai = np.max(np.exp(-1.0 * np.array(sum_squared_diff) / squared_sigma))
            return s_B_i_x_k_fai

        train_ids = []
        test_ids = self.fold_groups[test_env]
        for num, fold_ids in enumerate(self.fold_groups):
            if num != test_env:
                train_ids.extend(fold_ids)

        all_ids = train_ids + test_ids

        if instance_space == 'all':
           dis_instance_pool_ids = train_ids.copy()
        elif instance_space in ('pd', 'hc', 'mild', 'moderate', 'severe'):
           dis_instance_pool_ids = train_ids.copy()
            # pass
            # if instance_space == 'pd':
        else:
            raise ValueError(f"instance_space: {instance_space} 值传入错误！")

        # 要得到train_vidx, test_vidx
        train_vidx = []
        test_vidx = []
        all_X_fai_list = []
        all_X_fai_label_list = []
        for idx, all_pid in enumerate(all_ids):
            B_i = bag_data_dict[all_pid]["bag_data"]
            s_list = []
            for ins_pool_pid in dis_instance_pool_ids:
                dis_instance_pool = bag_data_dict[ins_pool_pid]["bag_data"]
                for dis_instance in dis_instance_pool:
                    s_i = similarity(B_i, dis_instance)
                    s_list.append(s_i)
            all_X_fai_list.append(np.array(s_list).reshape(len(s_list), 1))
            all_X_fai_label_list.append(bag_data_dict[all_pid]["bag_data_label"])

            if all_pid in train_ids:
                train_vidx.append(idx)
            elif all_pid in test_ids:
                test_vidx.append(idx)
            else:
                raise ValueError(f"all_pid: {all_pid} 既不在train_ids也不在test_ids中！")

        all_X_fai = np.hstack(all_X_fai_list)
        all_X_fai_label = np.array(all_X_fai_label_list)

        train_X_fai = all_X_fai[:, train_vidx]
        train_X_fai_label = all_X_fai_label[train_vidx]

        # 构建Q标签嵌入矩阵（这里直接影响最终的RDIP评分）
        Q_size = train_X_fai_label.size
        label_counts_dict = dict()
        for label, label_counts in enumerate(np.bincount(train_X_fai_label)):
            if label not in label_counts_dict.keys():
                label_counts_dict[label] = label_counts
        Q = np.zeros((Q_size, Q_size))
        # class_size = len(set(self.severity_mapping.values()))  # 类别数量
        for index_i, label_i in enumerate(train_X_fai_label):
            for index_j, label_j in enumerate(train_X_fai_label):
                row_idx_label_counts = label_counts_dict[label_i]
                column_idx_label_counts = label_counts_dict[label_j]
                if label_i == label_j:
                    if row_idx_label_counts < column_idx_label_counts:
                        # Q[index_i, index_j] = -1 / (
                        #         row_idx_label_counts / Q_size)
                        Q[index_i, index_j] = -1 / np.sqrt(
                            0.5 * ((row_idx_label_counts + column_idx_label_counts) / Q_size))
                    else:
                        # Q[index_i, index_j] = -1 / (
                        #         column_idx_label_counts / Q_size)
                        Q[index_i, index_j] = -1 * np.sqrt(
                            0.5 * ((row_idx_label_counts + column_idx_label_counts) / Q_size))
                else:
                    if row_idx_label_counts < column_idx_label_counts:
                        # Q[index_i, index_j] = 1 / (
                        #         row_idx_label_counts / Q_size)
                        Q[index_i, index_j] = 1 * np.sqrt(
                            0.5 * ((row_idx_label_counts + column_idx_label_counts) / Q_size))
                    else:
                        # Q[index_i, index_j] = 1 / (
                        #         column_idx_label_counts / Q_size)
                        Q[index_i, index_j] = 1 * np.sqrt(
                            0.5 * ((row_idx_label_counts + column_idx_label_counts) / Q_size))
                # else:
                #     raise ValueError("Q矩阵分配权重错误")

        A = np.count_nonzero(Q < 0)  # 统计标签相同Q中序号对的个数
        B = np.count_nonzero(Q > 0)  # 统计标签不相同Q中序号对的个数

        Q[Q < 0] /= A
        Q[Q > 0] /= B

        # 计算 D 的对角线上的元素
        D_diagonal = np.sum(Q, axis=1)
        # 创建对角矩阵 D
        D = np.diag(D_diagonal)
        # 创建拉普拉斯矩阵L
        L = D - Q

        scores_list = []
        scores = np.diagonal(train_X_fai @ L @ train_X_fai.T)
        scores_list.append(scores)

        # # ===============使用J(P)计算所有X中实例得分===============
        sorted_indices = np.argsort(-scores)
        train_ordered_full_map = all_X_fai[:, train_vidx].T[:, sorted_indices]
        train_ordered_full_map_label = all_X_fai_label[train_vidx]
        test_ordered_full_map = all_X_fai[:, test_vidx].T[:, sorted_indices]
        test_ordered_full_map_label = all_X_fai_label[test_vidx]
        # # ===============使用J(P)计算所有X中实例得分===============
        min_cols = min(test_ordered_full_map.shape[1], 1500)
        return (train_ordered_full_map, train_ordered_full_map_label,
                test_ordered_full_map, test_ordered_full_map_label, min_cols)

    def create_train_test_split(self, fold_num, test_ids):
        train_ids = []
        for num, fold_ids in enumerate(self.fold_groups):
            if num != fold_num:
                train_ids.extend(fold_ids)
        # train_X = [self.bag_data_dict[pid]["bag_data"] for pid in train_ids]
        # =======================================
        # 检查每个 pid 的数据是否满足条件
        for pid in train_ids:
            if len(self.bag_data_dict[pid]["bag_data"]) < 36:
                raise ValueError(f"Data for participant {pid} is less than 36 rows.")
        # =======================================
        train_X = [self.bag_data_dict[pid]["bag_data"][:36] for pid in train_ids]

        train_X = np.vstack(train_X)
        # train_Y = [self.bag_data_dict[pid]["bag_data_instance_label"] for pid in train_ids]
        train_Y = [self.bag_data_dict[pid]["bag_data_instance_label"][:36] for pid in train_ids]
        train_Y = np.hstack(train_Y)

        # test_X_ls = [self.bag_data_dict[pid]["bag_data"] for pid in test_ids]
        # =======================================
        # 先检查每个 pid 的数据是否足够长
        for pid in test_ids:
            if len(self.bag_data_dict[pid]["bag_data"]) < 36:
                raise ValueError(f"Data for participant {pid} is less than 36 rows.")
        # =======================================
        test_X_ls = [self.bag_data_dict[pid]["bag_data"][:36] for pid in test_ids]
        # test_Y_ls = [self.bag_data_dict[pid]["bag_data_label"] for pid in test_ids]
        test_Y_ls = [self.bag_data_dict[pid]["bag_data_label"] for pid in test_ids]

        # # ====================标准化=======================
        # from sklearn.preprocessing import StandardScaler
        # # 创建StandardScaler对象
        # scaler = StandardScaler()
        # # 使用train_X的均值和标准差来拟合并转换train_X
        # train_X = scaler.fit_transform(train_X)
        # for idx, test_X in enumerate(test_X_ls):
        #     test_X_ls[idx] = scaler.transform(test_X)
        # # ====================标准化=======================
        return train_X, train_Y, test_X_ls, test_Y_ls, train_ids, test_ids

    # 创建模型和返回参数(lgb和xgb只返回参数，模型返回为None，需要自己手动创建模型在训练测试中)
    def create_model(self, classifier, k_num=10):
        if classifier == 'logistic_l1':

            logistic_l1_params = {
                'penalty': 'l1',  # 正则方式为l1
                'solver': 'saga',  # 'saga'  solver supports L1 regularization and is suitable for large datasets
                'C': 0.1,  # 正则化力度，C越小正则越强，越大正则越弱(重要调整)
                'random_state': 0,  # 随机种子
                'multi_class': 'multinomial',
                'max_iter': 50,  # 迭代次数(重要调整)
                'n_jobs': -1,  # 使用所有可用的CPU核心
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = logistic_l1_params
            model = make_pipeline(StandardScaler(), LogisticRegression(**logistic_l1_params))
        elif classifier == 'logistic_l2':
            # Logistic Regression with L2 regularization
            logistic_l2_params = {
                'penalty': 'l2',  # 正则方式为l2
                'solver': 'saga',  # 'saga' solver supports L1 regularization and is suitable for large datasets
                'C': 0.01,  # c是正则化参数的倒数，越小正则越强(重要调整)
                'random_state': 0,  # 随机种子
                'multi_class': 'multinomial',
                'max_iter': 50,  # 迭代次数(重要调整)
                'n_jobs': -1,  # 使用所有可用的CPU核心
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = logistic_l2_params
            model = make_pipeline(StandardScaler(), LogisticRegression(**logistic_l2_params))
        elif classifier == 'svm_l1':
            # Linear SVM with L1 regularization
            linear_svm_l1_params = {
                'penalty': 'l1',  # 正则方式为l1
                'loss': 'squared_hinge',  # 损失计算，尽量使得远离边界
                'dual': False,
                'C': 0.01,  # c是正则化参数的倒数，越小正则越强(重要调整)
                'random_state': 0,
                'max_iter': 1000,  # 迭代次数(重要调整)
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = linear_svm_l1_params
            model = make_pipeline(StandardScaler(), LinearSVC(**linear_svm_l1_params))
        elif classifier == 'svm_l2':
            # Linear SVM with L2 regularization
            linear_svm_l2_params = {
                'penalty': 'l2',  # 正则方式为l1
                'loss': 'squared_hinge',  # 损失计算，尽量使得远离边界
                'dual': True,
                'C': 0.01,  # c是正则化参数的倒数，越小正则越强(重要调整)
                'random_state': 0,
                'max_iter': 1000,  # 迭代次数(重要调整)
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = linear_svm_l2_params
            model = make_pipeline(StandardScaler(), LinearSVC(**linear_svm_l2_params))
        elif classifier == 'knn':
            knn_params = {
                'n_neighbors': k_num,  # 邻居数量(重要调整)
                'weights': 'distance',  # 距离衡量
                'algorithm': 'auto',  # 距离计算算法，是否使用KTTree
                'n_jobs': -1,  # 使用所有可用的CPU核心
            }
            params = knn_params
            model = make_pipeline(StandardScaler(), KNeighborsClassifier(**knn_params))
            # model = make_pipeline(KNeighborsClassifier(**knn_params))
        elif classifier == 'bayes':
            bayes_params = {
                'priors': None,  # 使用训练数据自动计算先验概率
                'var_smoothing': 1e-9  # 避免除以零错误的平滑参数
            }
            params = bayes_params
            model = make_pipeline(StandardScaler(), GaussianNB(**bayes_params))
        elif classifier == 'rf':
            rf_params = {
                'n_estimators': 500,  # 树的数量
                'max_depth': 5,
                'min_samples_split': 2,  # 只有当该节点包含至少两个样本时，才会进行分裂
                'min_samples_leaf': 1,  # 树中的叶子节点必须拥有的最小样本数量
                'max_features': 0.75,  # 分裂节点时考虑的随机特征的数量是总特征数量的 75%
                'bootstrap': True,  # 为 True 时，每棵树训练数据是通过从原始训练数据中进行有放回的抽样得到的，同一个数据点可能会被多次选中。
                'random_state': 0,
                'n_jobs': -1,  # 使用所有可用的CPU核心
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = rf_params
            model = make_pipeline(StandardScaler(), RandomForestClassifier(**rf_params))
        elif classifier == 'lgbm':
            lgb_params = {
                'learning_rate': 0.02,
                'min_child_samples': 1,  # 子节点最小样本个数
                'max_depth': 1,  # 树的最大深度
                # 'lambda_l1': 0.001,  # 控制过拟合的超参数
                # 'lambda_l2': 0.0001,  # 控制过拟合的超参数
                # 'min_split_gain': 0.015,  # 最小分裂增益
                'boosting': 'gbdt',
                'objective': 'multiclass',
                # 'n_estimators': 300,  # 决策树的最大数量
                'metric': 'multi_error',
                'num_class': len(set(self.severity_mapping.values())),
                'feature_fraction': .90,  # 每次选取百分之75的特征进行训练，控制过拟合
                'bagging_fraction': .75,  # 每次选取百分之85的数据进行训练，控制过拟合
                'seed': 0,
                'num_threads': -1,
                'verbose': -1,
                # 'early_stopping_rounds': 50,  # 当验证集在训练一百次过后准确率还没提升的时候停止
                'num_leaves': 128,
            }
            params = lgb_params
            model = None
        elif classifier == 'xgb':
            xgb_params = {
                'max_depth': 7,  # 树的最大深度
                'learning_rate': 0.02,  # 学习率
                # 'n_estimators': 200,  # 树的数量
                # 'gamma': 0.1,  # 最小分裂增益
                'min_child_weight': 1,  # 最小子权重
                'subsample': 0.75,  # 子采样比例
                'colsample_bytree': 0.70,  # 每棵树的特征采样比例
                'reg_alpha': 0.10,  # L1正则化
                # 'reg_lambda': 0.50,  # L2正则化
                'objective': 'multi:softprob',  # 多分类目标
                'eval_metric': 'mlogloss',  # 评估指标
                'num_class': len(set(self.severity_mapping.values())),  # 类别数量
                'seed': 0,  # 随机种子
                'nthread': 20,  # 线程数
            }
            params = xgb_params
            model = None
        elif classifier == 'mlp_2':
            mlp_2_params = {
                'hidden_layer_sizes': (128, 64),  # 隐藏层神经元数量
                'activation': 'relu',  # 激活函数
                'solver': 'adam',  # 优化算法
                'alpha': 0.0001,  # l2正则(重要调整)
                'batch_size': 'auto',  # 批量大小
                'learning_rate': 'constant',  # 学习率调度，表示是否学习率不变
                'learning_rate_init': 5e-4,  # 学习率大小(重要调整)
                'max_iter': 200,  # epoch(重要调整)
                'random_state': 2024,  # 随机种子
                'early_stopping': True,  # 早停
                'validation_fraction': 0.1,  # 使用10%的数据作为验证集
                'n_iter_no_change': 10,  # 若10个迭代不再提升，则提前停止
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = mlp_2_params
            model = make_pipeline(StandardScaler(), MLPClassifier(**mlp_2_params))
        elif classifier == 'mlp_4':
            mlp_4_params = {
                'hidden_layer_sizes': (64, 32, 16, 8),
                'activation': 'relu',  # 激活函数
                'solver': 'adam',  # 优化算法
                'alpha': 0.001,  # l2正则
                'batch_size': 'auto',  # 批量大小
                'learning_rate': 'constant',  # 学习率调度，表示是否学习率不变
                'learning_rate_init': 5e-4,  # 学习率大小
                'max_iter': 200,  # epoch
                'random_state': 2024,  # 随机种子
                'early_stopping': True,  # 早停
                'validation_fraction': 0.1,  # 使用10%的数据作为验证集
                'n_iter_no_change': 10,  # 若10个迭代不再提升，则提前停止
                # 'n_jobs': -1,  # 使用所有可用的CPU核心
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = mlp_4_params
            model = make_pipeline(StandardScaler(), MLPClassifier(**mlp_4_params))
        elif classifier == 'mlp_8':
            mlp_8_params = {
                'hidden_layer_sizes': (256, 128, 64, 64, 32, 32, 16, 8),
                'activation': 'relu',  # 激活函数
                'solver': 'adam',  # 优化算法
                'alpha': 0.001,  # l2正则
                'batch_size': 'auto',  # 批量大小
                'learning_rate': 'constant',  # 学习率调度，表示是否学习率不变
                'learning_rate_init': 5e-4,  # 学习率大小
                'max_iter': 200,  # epoch(重要调整)
                'random_state': 2024,  # 随机种子
                'early_stopping': True,  # 早停
                'validation_fraction': 0.1,  # 使用10%的数据作为验证集
                'n_iter_no_change': 10,  # 若10个迭代不再提升，则提前停止
                # 'n_jobs': -1,  # 使用所有可用的CPU核心
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = mlp_8_params
            model = make_pipeline(StandardScaler(), MLPClassifier(**mlp_8_params))
        else:
            raise ValueError("Unsupported classifier type. Supported types are 'knn', 'mlp', and 'svm'.")

        return model, params

    def predict_most_likely_class(self, model, test_X_ls, classifier):
        y_pred_ls = []
        for test_X in test_X_ls:
            if classifier in ['logistic_l1', 'logistic_l2', 'knn', 'bayes', 'rf', 'mlp_2', 'mlp_4', 'mlp_8']:
                y_pred_prob = model.predict_proba(test_X)
                y_pred = np.argmax(y_pred_prob, axis=1)
            elif classifier in ['svm_l1', 'svm_l2']:
                y_pred = model.predict(test_X)
            elif classifier == 'lgbm':
                y_pred_prob = model.predict(test_X, num_iteration=model.best_iteration)
                y_pred = np.argmax(y_pred_prob, axis=1)
            elif classifier == 'xgb':
                y_pred_prob = model.predict(test_X, iteration_range=(0, model.best_iteration))
                # y_pred_prob = model.predict(test_X, iteration_range=(0, model.best_iteration + 1))
                y_pred = np.argmax(y_pred_prob, axis=1)
            else:
                raise ValueError("Unsupported classifier type for prediction.")
            counts = np.bincount(y_pred)
            y_pred_ls.append(np.argmax(counts))
        return y_pred_ls


if __name__ == '__main__':
    activity_range = [int(i) for i in range(9, 10)]  # 16个活动
    for activity_id in tqdm(activity_range, desc="Processing activity_range"):
        classifier = PDClassifier(r"0219_242D_ws100_ol0.5_wristr_acc_gro_0_3-20hz_100hz.csv", activity_id)
        for cur_test_env in tqdm([int(j) for j in range(0, 4)], desc="Processing test_env"):
            classifier.train_and_evaluate(test_env=cur_test_env)  # 选择对应的model进行留一验证