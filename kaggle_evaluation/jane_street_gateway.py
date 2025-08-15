"""Gateway notebook for https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting"""

import os

import kaggle_evaluation.core.base_gateway
import kaggle_evaluation.core.templates
import polars as pl


class JSGateway(kaggle_evaluation.core.templates.Gateway):
    def __init__(self, data_paths: tuple[str, str] | None = None):
        super().__init__(data_paths, file_share_dir=None)
        self.data_paths = data_paths
        self.set_response_timeout_seconds(60)

    def unpack_data_paths(self):
        if not self.data_paths:
            self.test_path = (
                "/kaggle/input/jane-street-realtime-marketdata-forecasting/test.parquet"
            )
            self.lags_path = (
                "/kaggle/input/jane-street-realtime-marketdata-forecasting/lags.parquet"
            )
        else:
            self.test_path, self.lags_path = self.data_paths

    def generate_data_batches(self):
        date_ids = sorted(
            pl.scan_parquet(self.test_path) # 使用 Polars 的惰性 API 扫描 self.test_path 下的所有 parquet 文件, 不立即加载数据
            .select(pl.col("date_id").unique()) # 选取 date_id 列的所有不同取值 (即不同交易日的 ID)
            .collect() # 执行计算 (从惰性转为实际读取数据)
            .get_column("date_id") # 取出 date_id 列的数据
        )
        assert date_ids[0] == 0

        for date_id in date_ids:
            test_batches = pl.read_parquet(
                os.path.join(self.test_path, f"date_id={date_id}"), # 读取 self.test_path 下 date_id={date_id} 对应的文件夹 (分区存储结构)
            ).group_by("time_id", maintain_order=True) # 按 time_id (一天内的时间片段) 分组, maintain_order=True 表示保持时间顺序
            lags = pl.read_parquet(
                os.path.join(self.lags_path, f"date_id={date_id}"),
            ) # 从 self.lags_path 下读取当前日期 date_id 对应的滞后特征
            for time_id, test in test_batches: # test_batches 是经过 group_by 的 dataframe, 支持这样的二元组迭代遍历
                # 当 time_id == 0 (一天的第一个时间点) 把完整的 lags 传入; 否则传入 None (表示滞后特征已在第一个时间步加载)
                test_data = (test, lags if time_id[0] == 0 else None) # test 是 39 行的数据 (每个 symbol_id 为一行)
                validation_data = test.select('row_id') # 仅从 test 中选取 row_id 列, 作为验证标识 (提交格式要求)
                yield test_data, validation_data


if __name__ == "__main__":
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        gateway = JSGateway()
        # Relies on valid default data paths
        gateway.run()
    else:
        print("Skipping run for now")
