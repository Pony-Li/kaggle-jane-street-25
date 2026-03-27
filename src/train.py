import polars as pl
import torch
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataset import TimeSeriesDataModule
from model_gru import GRUNetworkWithConv, Hyperparameters
from utils import calculate_r2, encode_column
import warnings
warnings.filterwarnings('ignore')

def main():
    params = Hyperparameters()
    category_mappings = {'feature_09': {2: 0, 4: 1, 9: 2, 11: 3, 12: 4, 14: 5, 15: 6, 25: 7, 26: 8, 30: 9, 34: 10, 42: 11, 44: 12, 46: 13, 49: 14, 50: 15, 57: 16, 64: 17, 68: 18, 70: 19, 81: 20, 82: 21},
                    'feature_10': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 10: 7, 12: 8},
                    'feature_11': {9: 0, 11: 1, 13: 2, 16: 3, 24: 4, 25: 5, 34: 6, 40: 7, 48: 8, 50: 9, 59: 10, 62: 11, 63: 12, 66: 13,76: 14, 150: 15, 158: 16, 159: 17, 171: 18, 195: 19, 214: 20, 230: 21, 261: 22, 297: 23, 336: 24, 376: 25, 388: 26, 410: 27, 522: 28, 534: 29, 539: 30},
                    'time_id' : {i : i/967 for i in range(968)}}
    feature_columns = sorted(list(set([f"feature_{i:02d}" for i in range(79)])-set(["feature_09","feature_10","feature_11","feature_61"])))
    feature_columns.extend(['time'])

    # 读取训练数据并过滤日期范围
    raw_data = pl.read_parquet('data/train.parquet').filter(pl.col('date_id').is_between(1100, 1700))
    
    # 特征选择和预处理
    raw_data = raw_data.with_columns((pl.col('time_id')/967).alias('time'))
    for col in ["feature_09","feature_10","feature_11"]:
        raw_data = encode_column(raw_data, col,category_mappings[col]) 
    raw_data = raw_data.to_pandas()  # 将数据转换为 Pandas DataFrame
    
    # 划分训练集和验证集
    valid_data = raw_data[raw_data['date_id'].between(1660, 1700)]  # 验证集
    train_data = raw_data[raw_data['date_id'] < 1660]  # 训练集

    # 填充缺失值
    train_data[feature_columns] = train_data[feature_columns].fillna(0)  # 训练集填充0
    valid_data[feature_columns] = valid_data[feature_columns].fillna(0)  # 验证集填充0

    print("the shape of training data: ", train_data.shape)  # (20687128, 94)
    print("Column info: ", train_data.columns.tolist())
    for col in train_data.columns[:3]:
        # date_id — unique values: 560, time_id — unique values: 968, symbol_id — unique values: 39
        print(f"{col} — unique values: {train_data[col].nunique()}")

    data_module = TimeSeriesDataModule(train_data, feature_columns, params.batch_size, valid_data=valid_data)
    data_module.setup()

    model = GRUNetworkWithConv(
        input_size=len(feature_columns),
        hidden_size=params.hidden_layer_size,
        output_size=1,
        num_layers=2,
        learning_rate=params.learning_rate,
        weight_decay=params.weight_decay
    )

    trainer = Trainer(
        max_epochs=params.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            EarlyStopping('val_loss', patience=params.early_stopping_patience, mode='min'),
            ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename="GRU")
        ]
    )
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())


if __name__ == "__main__":
    main()
