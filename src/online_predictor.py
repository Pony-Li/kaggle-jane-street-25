import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
import polars as pl
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import kaggle_evaluation.jane_street_inference_server as js_infer
from model_gru import Hyperparameters

params = Hyperparameters()

class JsGruOnlinePredictor:
     
    def __init__(self, 
                 test_parquet, 
                 lags_parquet,
                 time_steps_total,
                 feature_names,
                 model,
                 historical_data,
                 prev_hidden_states):
        self.test_parquet = test_parquet
        self.lags_parquet = lags_parquet
        self.lags_ = None # 最近一次到的滞后数据副本, lags 到来时触发换天逻辑
        self.model = model # 已经离线训练好的 GRU 模型: 在线阶段会不断调用 eval() 推理, 必要时切到 train() 微调
        self.feature_names = feature_names
        self.test_input = np.zeros((39, 968, len(self.feature_names)), dtype=np.float32)
        self.pbar = tqdm(total=time_steps_total)
        self.prev_hidden_states= prev_hidden_states
        self.passed_days = 0 # 天数计数器
        self.historical_cache = [] # 当天陆续到达的 test 分块按列对齐追加，用于在换天时合并成上一天的完整数据
        self.historical_data = historical_data # 离线阶段准备好的历史若干天列表, 在线时会不断 append 最近天数并截断 [-50:], 作为在线学习的原材料池
        self.begin = False # 是否开始
        self.batches = None
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4) # 在线微调时用的 AdamW (与离线训练的优化器互不干扰, lr 是离线时的 1/10)
        self.num_symbols = 39
        self.time_steps = 968
        self.device = torch.device(f'cuda:{params.gpu_id}' if torch.cuda.is_available() and params.use_gpu else 'cpu')
        self.online_learning_count = 0
        self.if_online_learning = True
        self.cache_columns = ['date_id','time_id','symbol_id','weight','time']+ [f"feature_{i:02d}" for i in range(79)] + [f'responder_{i}' for i in range(0, 9)]

    def run_inference_server(self):

        inference_server = js_infer.JSInferenceServer(self.predict)
        
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            inference_server.serve()
        else:
            inference_server.run_local_gateway((self.test_parquet, self.lags_parquet))

    def output(self,features,mask):
        with torch.no_grad():
            # model.prev_hidden_state=None
            self.model.eval()
            self.model.to('cuda:0')
            output = self.model(features)
            output = output[mask==1]
        return output[:,-1].cpu().numpy()
    
    # def prepare_data(self, data, feature_columns, target_column, weight_column):
    #     features, targets, weights, masks = [], [], [], []
    #     data = data.to_pandas()
    #     grouped_by_date = data.groupby("date_id")  # 对全量的数据按照日期分组

    #     for date_id, date_group in grouped_by_date:
    #         date_features = torch.zeros((self.num_symbols, self.time_steps, len(feature_columns)))  # 初始化特征张量（39，968，79)
    #         date_targets = torch.zeros((self.num_symbols, self.time_steps))  # 初始化目标张量 (39,968)
    #         date_weights = torch.zeros((self.num_symbols, self.time_steps))  # 初始化权重张量(39,968)
    #         date_masks = torch.ones((self.num_symbols, self.time_steps))  # 初始化掩码张量(39,968)

    #         grouped_by_symbol = date_group.groupby("symbol_id")  # 按符号分组
    #         for symbol_id, group in grouped_by_symbol:
    #             group = group.sort_values("time_id")  # 按时间排序
    #             x = group[feature_columns].values  # 特征值
    #             y = group[target_column].values  # 目标值
    #             w = group[weight_column].values  # 权重值

    #             if len(group) > self.time_steps:
    #                 x, y, w = x[-self.time_steps:], y[-self.time_steps:], w[-self.time_steps:]  # 截取后 time_steps 个数据
    #             elif len(group) < self.time_steps:
    #                 print(pad_size)
    #                 pad_size = self.time_steps - len(group)  # 计算需要填充的大小
    #                 x = np.pad(x, ((0, pad_size), (0, 0)), mode="constant")  # 填充特征
    #                 y = np.pad(y, (0, pad_size), mode="constant")  # 填充目标
    #                 w = np.pad(w, (0, pad_size), mode="constant")  # 填充权重
    #                 date_masks[symbol_id, -pad_size:] = 0  # 更新掩码
                    
    #             date_features[symbol_id] = torch.FloatTensor(x)  # 转换为张量
    #             date_targets[symbol_id] = torch.FloatTensor(y)  # 转换为张量
    #             date_weights[symbol_id] = torch.FloatTensor(w)  # 转换为张量

    #         features.append(date_features)  # 添加特征
    #         targets.append(date_targets)  # 添加目标
    #         weights.append(date_weights)  # 添加权重
    #         masks.append(date_masks)  # 添加掩码

    #     features = torch.stack(features).to(self.device)  # 堆叠特征并移动到设备
    #     targets = torch.stack(targets).to(self.device)  # 堆叠目标并移动到设备
    #     weights = torch.stack(weights).to(self.device)  # 堆叠权重并移动到设备
    #     masks = torch.stack(masks).to(self.device)  # 堆叠掩码并移动到设备
        
    #     return features, targets, weights, masks # (date_ids, 39, 968, 79)

    # 与上面的 prepare_data 函数相比, 下面的版本没有使用 to_pandas 将 polars dataframe 转化为 pandas dataframe, 大大节省了时间, 避免了 10 天一次的在线学习过程耗时过长
    def prepare_data(self, data: pl.DataFrame, feature_columns, target_column: str, weight_column: str):
        """
        输入:
        - data: Polars DataFrame, 包含至少 ['date_id','time_id','symbol_id', feature_columns..., target_column, weight_column]
        输出:
        - features: (D, 39, 968, F)
        - targets : (D, 39, 968)
        - weights : (D, 39, 968)
        - masks   : (D, 39, 968)
        说明:
        - 全程不转 Pandas, 避免昂贵的往返
        - 每天对每个 symbol 先按 time_id 排序, 再放入 (39, 968, F) 的槽位
        - 若记录不足 968, 尾部做 0 填充，并在 mask 上标 0
        - 若记录超过 968, 只取最后 968 条
        """
        F = len(feature_columns)
        features_list, targets_list, weights_list, masks_list = [], [], [], []

        # 按天拆分（保持原有顺序）
        # 如果 data 很大，partition_by 的拷贝成本仍然存在，但比转成 Pandas 便宜得多
        for date_df in data.partition_by('date_id', maintain_order=True):
            # 在目标设备上直接分配，避免之后再次搬运
            date_features = torch.zeros((self.num_symbols, self.time_steps, F), device=self.device, dtype=torch.float32)
            date_targets  = torch.zeros((self.num_symbols, self.time_steps),   device=self.device, dtype=torch.float32)
            date_weights  = torch.zeros((self.num_symbols, self.time_steps),   device=self.device, dtype=torch.float32)
            date_masks    = torch.ones( (self.num_symbols, self.time_steps),   device=self.device, dtype=torch.float32)

            # 按 symbol 拆分
            for sym_df in date_df.partition_by('symbol_id', maintain_order=True):
                # 取得当前 symbol_id（假设为 0..38，可直接作为行下标使用；若不是，请建立映射字典）
                sid = int(sym_df['symbol_id'][0])

                # 按 time_id 排序
                sym_df = sym_df.sort('time_id')

                # 取出 numpy 数组（Polars → NumPy 是零拷贝或低拷贝，远比转 Pandas 快）
                x_np = sym_df.select(feature_columns).to_numpy()               # (n, F)
                y_np = sym_df.select(target_column).to_numpy().reshape(-1)     # (n,)
                w_np = sym_df.select(weight_column).to_numpy().reshape(-1)     # (n,)
                n = x_np.shape[0]

                # 根据长度写入到 (time_steps) 的槽位
                if n >= self.time_steps:
                    # 超过 968: 取最后 968 条，右对齐也行，这里沿用“取末尾”的策略
                    x_np = x_np[-self.time_steps:]
                    y_np = y_np[-self.time_steps:]
                    w_np = w_np[-self.time_steps:]

                    # 直接整段拷到 GPU 张量（避免 numpy.pad）
                    date_features[sid, :, :].copy_(torch.from_numpy(x_np).to(self.device))
                    date_targets[sid, :].copy_(torch.from_numpy(y_np).to(self.device))
                    date_weights[sid, :].copy_(torch.from_numpy(w_np).to(self.device))
                    # mask 保持 1（全部有效）
                else:
                    # 不足 968: 前 n 步写数据，后面自动保留为 0，并把 mask 在尾部置 0
                    # 先把需要的切片拿出来，避免重复索引
                    tgt_feat = date_features[sid, :n, :]
                    tgt_y    = date_targets[sid, :n]
                    tgt_w    = date_weights[sid, :n]

                    tgt_feat.copy_(torch.from_numpy(x_np).to(self.device))
                    tgt_y.copy_(torch.from_numpy(y_np).to(self.device))
                    tgt_w.copy_(torch.from_numpy(w_np).to(self.device))

                    # 尾部 padding 的位置 mask 置 0（与原先逻辑一致：尾部是填充）
                    date_masks[sid, n:] = 0.0

            # 收集每天的张量
            features_list.append(date_features)
            targets_list.append(date_targets)
            weights_list.append(date_weights)
            masks_list.append(date_masks)

        # 堆叠成 (D, 39, 968, F) / (D, 39, 968)
        features = torch.stack(features_list, dim=0)  # 已在 device 上
        targets  = torch.stack(targets_list,  dim=0)
        weights  = torch.stack(weights_list,  dim=0)
        masks    = torch.stack(masks_list,    dim=0)

        return features, targets, weights, masks
    
    def online_learning(self):
        self.online_learning_count += 1
        self.model.train()
        features, targets, weights, masks = self.batches
        for i in tqdm(range(len(features)),total= len(features),colour='green',desc=f'The {self.online_learning_count} Epoch Online Learning'):
            x,y,w,m = features[i],targets[i],weights[i],masks[i]
            y_pred = self.model(x)
            if y_pred.shape != y.shape:
                y_pred = y_pred.view_as(y)  # 调整预测形状
                m = m.view_as(y)  # 调整掩码形状
                y_pred, y = y_pred[m == 1], y[m == 1]  # 应用掩码
            loss = F.mse_loss(y_pred, y, reduction='none').mean()
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(),max_norm=1)
            self.optimizer.step()
            self.prev_hidden_states.pop(0)
            self.prev_hidden_states.append(self.model.prev_hidden_state)

    def predict(self, test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:

        """
        Make a prediction.
        test 是一个 time_id 内各个 symbol_id 的全部特征数据, shape of test: (num_symbols, num_features)
        lags 是上一天所有 time_id 和 symbol_id 的全部 responder 数据, shape of lags: (num_times, num_symbols, num_responders)
        """

        if lags is not None: # 新一天的第一个时间段, time_id = 0
            self.lags_ = lags
            self.passed_days += 1
            print(f'date form begin {self.passed_days}')

            if self.begin: # 是否已经开始过预测 (即是否已经有"昨天"）

                # 把昨天积累的所有 test 分块拼接成上一天的完整表 last_day_cahe (self.historical_cache 是个 list, 每个元素是昨天某时间步的 test 子表)
                last_day_cahe = pl.concat(self.historical_cache)

                # API 提供的 lags 数据是"昨天"的真实 responder 数据, 因此把 date_id 统一减 1, 使其与 last_day_cahe 能在相同 date_id 上对齐
                lags = lags.with_columns((pl.col('date_id') - 1).alias('date_id'))

                # 按 (date_id, time_id, symbol_id) 把 lags 左连接到昨天的完整数据上
                last_day_cahe  = last_day_cahe.join(lags,on=['date_id','time_id','symbol_id'],how='left')

                # 把 responder_i_lag_1 改名为标准的 responder_i，这样后续训练 or 评估代码能把它当作标签列使用
                last_day_cahe = last_day_cahe.rename({f'responder_{i}_lag_1':f'responder_{i}' for i in range(0,9)})
                
                # 缺失值填 0, 只保留 self.cache_columns 指定的列: ['date_id','time_id','symbol_id','weight','time'] + feature_00..78 + responder_0..8
                last_day_cahe = last_day_cahe.fill_null(0)[self.cache_columns]

                # 把昨天的完整数据 (feature + responder) 加进历史池; 只保留最近 50 天 (控制内存与训练数据量)
                self.historical_data.append(last_day_cahe)
                self.historical_data = self.historical_data[-50:]

                # 把最近 50 天的数据拼接成一个大的 polars dataframe
                online_learning_data = pl.concat(self.historical_data,how='vertical_relaxed')

                if self.if_online_learning and self.passed_days==10:
                    self.batches = self.prepare_data(online_learning_data, feature_columns=self.feature_names, target_column='responder_6',weight_column='weight')
                    self.online_learning()
                    self.historical_cache = []
                    self.passed_days = 0
                    
        test = test.fill_null(0) # 在当前时间步的缺失值处补 0
        test = test.with_columns((pl.col('time_id') / 967).alias('time')) # 增加归一化时间列 time ∈ [0,1]
        preds = np.zeros(test.shape[0]) # preds 预分配 (长度 = 本步 test 的行数)

        symbol_mask = np.zeros(39, dtype=bool) # 创建一个长度为 39 的布尔掩码, 标记本步实际出现的 symbol
        unique_symbols = np.array(sorted(test.to_pandas()['symbol_id'].unique())) 
        symbol_mask[:len(unique_symbols)] = True

        self.test_input = np.roll(self.test_input, shift=-1, axis=1) # 沿时间轴左移一格，把最老的一列丢掉，为本步新特征腾出最后一列
        feat = test[self.feature_names].to_numpy() # 把当前 time_id 的 test 中的特征矩阵取出, shape: (num_symbols, num_features)
        if len(test) < 39: # 若不足 39 行, 就在底部补 0 使其形状为 (39, num_features)
            feat = np.pad(feat, ((0, len(symbol_mask) - len(unique_symbols)), (0, 0)), 'constant', constant_values = (0,0))

        self.test_input[:, -1, :] = feat # 把形状为 (39, num_features) 的当前 time_id 特征数据写到滚动窗口的最后一列
        features = self.test_input
        features = torch.tensor(features).to('cuda')
        preds += self.output(features, symbol_mask)

        predictions = test.select(['row_id']).with_columns(
                        pl.Series(
                            name='responder_6',  # 预测结果列的名称
                            values=np.clip(preds, a_min=-5, a_max=5),  # 将预测结果限制在 -5 到 5 的范围内
                            dtype=pl.Float64,  # 预测列的数据类型为 Float64
                        )
                    ) # 构造提交用的 DataFrame: 行号对齐 row_id, 预测值裁剪到 [-5, 5]
        
        # 维护一个固定长度的"隐藏态队列" (先进先出)
        self.prev_hidden_states.append(self.model.prev_hidden_state)
        self.prev_hidden_states.pop(0)

        # 更新 tqdm 进度条 (时间步 +1)
        self.pbar.update(1)
        self.pbar.refresh()

        # 把本步 test 放入当天缓存，等"跨天"时再合并成 last_day_cahe；
        self.historical_cache.append(test)

        # 标记系统已开始工作 (供下一次遇到 lags 时判断有无"昨天"的缓存可用)
        self.begin = True
        
        # The predict function must return a DataFrame
        assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
        # with columns 'row_id', 'responer_6'
        assert list(predictions.columns) == ['row_id','responder_6']
        # and as many rows as the test data.
        assert len(predictions) == len(test)
       
        return predictions
