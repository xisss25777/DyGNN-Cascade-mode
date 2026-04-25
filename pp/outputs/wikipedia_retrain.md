# wikipedia 数据集实验摘要

## 核心结果
- 样本数：600
- 特征数：29
- 融合权重 alpha：N/A
- MAE：12.22
- RMSE：21.52
- MAPE：0.1803

## Top 特征
- slice_13_attention: 0.196296
- slice_12_attention: 0.164115
- slice_10_attention: 0.158141
- slice_9_attention: 0.143013
- slice_8_attention: 0.129746
- slice_11_attention: 0.098692
- slice_7_attention: 0.047374
- slice_6_attention: 0.023813

## 代表性测试样本
- 级联 22_aug3：预测规模 71.1707，主导模式 早期快速扩散模式，删除实验 delta=15.3408
- 级联 68_aug1：预测规模 45.3697，主导模式 局部聚集后外溢模式，删除实验 delta=3.0232
- 级联 22_aug2：预测规模 49.5954，主导模式 早期快速扩散模式，删除实验 delta=5.9018
- 级联 58_aug1：预测规模 42.058，主导模式 桥接式跨层传播模式，删除实验 delta=0.7099
- 级联 491_aug1：预测规模 44.2235，主导模式 桥接式跨层传播模式，删除实验 delta=3.7944

## 文件说明
- JSON 结果文件：wikipedia_retrain.json
- 当前摘要文件：wikipedia_retrain.md