import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import numpy as np
from tqdm import tqdm

TARGET_LABEL = 'responder_6'

# 自定义时间序列数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, data, device, feature_columns, target_column, weight_column="weight", num_symbols=39, time_steps=968):
        self.device = device  # 设备
        self.num_symbols = num_symbols  # 符号数量
        self.time_steps = time_steps  # 时间步长
        self.features, self.targets, self.weights, self.masks = self._prepare_data(data, feature_columns, target_column, weight_column)  # 准备数据

    def _prepare_data(self, data, feature_columns, target_column, weight_column):
        features, targets, weights, masks = [], [], [], []
        grouped_by_date = data.groupby("date_id")  # 对全量的数据按照日期分组

        # 按照日期循环迭代数据, 每个 date_group 是 (time_steps=968)×(num_symbols=39) 行的 dataframe
        for date_id, date_group in tqdm(grouped_by_date, desc="Processing Dates", total=len(grouped_by_date)):

            # 初始化四个常值张量存储单日数据, 包括特征数据、目标值数据、权重数据 (用于计算 r2) 和掩码数据 (用来标记哪些时间步是真实数据, 哪些是填充数据)
            date_features = torch.zeros((self.num_symbols, self.time_steps, len(feature_columns)))  # 初始化特征张量 (39, 968, num_features)
            date_targets = torch.zeros((self.num_symbols, self.time_steps))  # 初始化目标张量 (39, 968)
            date_weights = torch.zeros((self.num_symbols, self.time_steps))  # 初始化权重张量(39, 968)
            date_masks = torch.ones((self.num_symbols, self.time_steps))  # 初始化掩码张量(39, 968)

            grouped_by_symbol = date_group.groupby("symbol_id")  # 将单日的数据按 symbol_id 分组

            # 按照 symbol_id 迭代循环每天的数据, 每个 group 是 (time_steps=968) 行的 dataframe
            for symbol_id, group in grouped_by_symbol:

                group = group.sort_values("time_id")  # 按时间排序
                x = group[feature_columns].values  # 特征值, (time_steps=968) 行 num_features 列的 dataframe
                y = group[target_column].values  # 目标值, (time_steps=968) 行 1 列的 dataframe
                w = group[weight_column].values  # 权重值, (time_steps=968) 行 1 列的 dataframe

                # 如果某个 symbol_id 对应数据的 time_steps 数量 > 968, 那么就只保留最后 968 个 time_steps 的数据
                if len(group) > self.time_steps:
                    x, y, w = x[-self.time_steps:], y[-self.time_steps:], w[-self.time_steps:]  # 截取后time_steps个数据
                
                # 如果某个 symbol_id 对应数据的 time_steps 数量 < 968, 那么就将其 padding 到 968 个 time_steps
                elif len(group) < self.time_steps:
                    pad_size = self.time_steps - len(group)  # 计算需要填充的 time_steps 数 (行数)
                    x = np.pad(x, ((0, pad_size), (0, 0)), mode="constant")  # 用 0 填充特征
                    y = np.pad(y, (0, pad_size), mode="constant")  # 用 0 填充目标
                    w = np.pad(w, (0, pad_size), mode="constant")  # 用 0 填充权重
                    date_masks[symbol_id, -pad_size:] = 0  # 更新掩码, 将 padding 数据对应部分的掩码设置为 0

                # x.shape: (968, num_features), y.shape: (968,), w.shape: (968,)
                date_features[symbol_id] = torch.FloatTensor(x)  # 将修剪工整的单个 symbol_id 的特征数据转换为 torch 张量后存入提前创建好的单日特征数据张量
                date_targets[symbol_id] = torch.FloatTensor(y)  # 将修剪工整的单个 symbol_id 的目标值数据 (responsder_6) 转换为 torch 张量后存入提前创建好的单日目标值数据张量
                date_weights[symbol_id] = torch.FloatTensor(w)  # 将修剪工整的单个 symbol_id 的权重数据转换为 torch 张量后存入提前创建好的单日权重数据张量

            # 将收集好的单日数据按次序 append 到空列表中
            features.append(date_features)  # 添加特征, 循环结束时的结构: num_dates=560 个形状为 (num_symbols=39, time_steps=968, num_features) 的三维 torch 张量组成的列表
            targets.append(date_targets)  # 添加目标, 循环结束时的结构: num_dates=560 个形状为 (num_symbols=39, time_steps=968) 的二维 torch 张量组成的列表
            weights.append(date_weights)  # 添加权重, 循环结束时的结构: num_dates=560 个形状为 (num_symbols=39, time_steps=968) 的二维 torch 张量组成的列表
            masks.append(date_masks)  # 添加掩码, 循环结束时的结构: num_dates=560 个形状为 (num_symbols=39, time_steps=968) 的二维 torch 张量组成的列表

        # 将由多个 torch 张量组成的列表堆叠成一个新的高维 torch 张量
        features = torch.stack(features).to(self.device)  # 堆叠特征并移动到设备, shape: (num_dates=560, num_symbols=39, time_steps=968, num_features)
        targets = torch.stack(targets).to(self.device)  # 堆叠目标并移动到设备, shape: (num_dates=560, num_symbols=39, time_steps=968)
        weights = torch.stack(weights).to(self.device)  # 堆叠权重并移动到设备, shape: (num_dates=560, num_symbols=39, time_steps=968)
        masks = torch.stack(masks).to(self.device)  # 堆叠掩码并移动到设备, shape: (num_dates=560, num_symbols=39, time_steps=968)

        return features, targets, weights, masks

    def __len__(self):
        return len(self.features)  # 返回数据集大小

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.weights[idx], self.masks[idx]  # 返回指定索引的数据

# 数据模块类，用于加载数据
class TimeSeriesDataModule(LightningDataModule):
    def __init__(self, train_data, feature_columns, batch_size, valid_data=None, device="cpu", num_symbols=39, time_steps=968):
        super().__init__()
        self.train_data = train_data  # 训练数据
        self.valid_data = valid_data  # 验证数据
        self.feature_columns = feature_columns  # 特征列
        self.batch_size = batch_size  # 批量大小, 我将 batch_size 设置为 1, 即每次迭代得到的都是单日数据
        self.device = device  # 设备
        self.num_symbols = num_symbols  # 符号数量
        self.time_steps = time_steps  # 时间步长
        self.train_dataset = None  # 训练数据集
        self.val_dataset = None  # 验证数据集

    def setup(self, stage=None):
        self.train_dataset = TimeSeriesDataset(self.train_data, self.device, self.feature_columns, TARGET_LABEL, num_symbols=self.num_symbols)  # 初始化训练数据集
        if self.valid_data is not None:
            self.val_dataset = TimeSeriesDataset(self.valid_data, self.device, self.feature_columns, TARGET_LABEL, num_symbols=self.num_symbols)  # 初始化验证数据集

    def train_dataloader(self, num_workers=0):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)  # 返回训练数据加载器

    def val_dataloader(self, num_workers=0):
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not provided.")  # 如果没有验证数据集，抛出错误
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)  # 返回验证数据加载器
