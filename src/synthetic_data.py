import os
import polars as pl

data_path = './data'
train_path = os.path.join(data_path, 'train.parquet/partition_id=*/part-0.parquet')

train_data = pl.read_parquet(train_path)
print('train data shape:', train_data.shape)

# date_id 大于1640的作为验证集
valid_data = train_data.filter((pl.col('date_id') >= 1640))
valid_data.write_parquet('./data/valid_data.parquet')

# 构造 syn_test 和 syn_lag
date_offset = 1650
pl_all = pl.read_parquet('./data/valid_data.parquet')

# make syn_test 构建模拟 test 数据集
syn_test = pl_all.with_columns(
    pl.lit(True).alias("is_scored"), # 新增一列 is_scored, 值全部为 True, 标识所有行都是有效的评分数据 (模拟 Kaggle API)
    pl.col('date_id') - date_offset # 将 date_id 减去偏移值 1650, 使数据重新编号, 从 0 开始
    ).with_row_index(name="row_id", offset=0) # 给每一行加上一个从 0 开始的行索引 row_id (模拟 Kaggle API)

syn_test = syn_test.select(
    ['row_id', 'date_id', 'time_id', 'symbol_id', 'weight', 'is_scored'] + [f'feature_{x:02}' for x in range(79)]
) # 只保留 syn_test 中 Kaggle 测试集所需的字段

# 将 syn_test 按 date_id 进行分组, 返回一个字典 {date_id: 对应DataFrame}, 模拟 Kaggle 官方 test 数据的按日期分片结构
syn_test_partition = syn_test.partition_by('date_id', maintain_order=True, as_dict=True)

output_dir = "./data/synthetic_test.parquet"
os.makedirs(output_dir, exist_ok=True)

# 如果有 date_id < 0 的行, 找到它们的最大 row_id, 作为偏移量 (一般用于确保 row_id 连续编号，从0开始)
row_id_offset = syn_test.filter(pl.col('date_id')<0).select('row_id').max().item()
print("row_id_offset:", row_id_offset)

# 对已经经过 group_by 的 syn_test_partition 进行下面的二元组遍历, key 是 date_id, _df 是一个 date_id 对应的全部数据
for key, _df in syn_test_partition.items():
    if key[0] >= 0: # 对于 date_id >= 0 的每个分组
        os.makedirs(f"{output_dir}/date_id={key[0]}", exist_ok=True) # 建立子目录 synthetic_test.parquet/date_id=xxx
        _df = _df.with_columns(pl.col('row_id')-row_id_offset) # 调整 row_id, 保证编号连续
        _df.write_parquet(f"{output_dir}/date_id={key[0]}/part-0.parquet") # 写入 part-0.parquet 文件, 模仿 Kaggle 官方按日期存储 test 数据的方式

# make syn_lag 构建模拟 lags 数据集
syn_lag = pl_all.select(
    ['date_id', 'time_id', 'symbol_id'] + [f'responder_{x}' for x in range(9)]
).with_columns(pl.col('date_id')-date_offset)

# 将 responder_x 重命名为 responder_x_lag_1, 模拟滞后值
syn_lag = syn_lag.rename({f'responder_{x}': f'responder_{x}_lag_1' for x in range(9)})

# 将 syn_lags 按 date_id 进行分组, 返回一个字典 {date_id: 对应DataFrame}, 模拟 Kaggle 官方 lags 数据的按日期分片结构
syn_lag_partition = syn_lag.partition_by('date_id', maintain_order=True, as_dict=True)

output_dir = "./data/synthetic_lags.parquet"
os.makedirs(output_dir, exist_ok=True)

for key, _df in syn_lag_partition.items():
    os.makedirs(f"{output_dir}/date_id={key[0]+1}", exist_ok=True)
    _df = _df.with_columns(pl.col('date_id')+1) # lags 的 date_id 被整体 +1, 这是因为 lags 提供的是前一天的响应值, 要与下一天的 test 数据对齐
    _df.write_parquet(f"{output_dir}/date_id={key[0]+1}/part-0.parquet")