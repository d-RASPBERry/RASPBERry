结果概览（`result_00464.json`）
================================

采集方式
--------
- 文件实际体积：`stat` 显示 3 372 056 B（≈3.2 MB，主要来自缩进与换行）。
- 使用临时 Python 脚本递归遍历 JSON，衡量每个路径经 `json.dumps`（无缩进）序列化后的字符数。该方式仅反映“有效数据量”，不含格式化空白，得到的总量 ≈0.70 MB。

主要占用来源
------------
- `result.config`：≈ 669 KB（有效数据量）  
  - `result.config.replay_buffer_config`：≈ 663 KB  
    - `obs_space`：≈ 662 KB  
      - `bounded_below`：≈ 179 KB  
      - `bounded_above`：≈ 179 KB  
      - `low`：≈ 152 KB  
      - `high`：≈ 152 KB
- 其它字段（`info`、`sampler_results`、`td_error` 等）均 < 10 KB。

结论
----
- 绝大多数体积来自 `obs_space` 中的高维数组；删除或压缩这些字段即可让单次迭代结果回落到几十 KB。
- 已在写 JSON 前剥离 `obs_space` / `action_space`，后续生成的新文件将不再包含这些大块数据；必要时另存摘要或单独备份空间定义。

