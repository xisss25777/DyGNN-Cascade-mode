# wikipedia 数据集实验摘要

## 核心结果
- 样本数：600
- 特征数：29
- 融合权重 alpha：N/A
- MAE：13.05
- RMSE：23.93
- MAPE：0.181

## Top 特征
- slice_12_attention: 0.999862
- slice_13_attention: 6.6e-05
- slice_2_attention: 2.7e-05
- slice_8_attention: 2e-05
- slice_11_attention: 1.3e-05
- slice_10_attention: 6e-06
- slice_9_attention: 4e-06
- slice_4_attention: 1e-06

## 代表性测试样本
- 级联 22_aug3：预测规模 65.7087，主导模式 早期快速扩散模式，删除实验 delta=13.7655
- 级联 68_aug1：预测规模 41.2174，主导模式 局部聚集后外溢模式，删除实验 delta=-0.7367
- 级联 22_aug2：预测规模 41.2844，主导模式 早期快速扩散模式，删除实验 delta=-0.3884
- 级联 58_aug1：预测规模 42.5439，主导模式 桥接式跨层传播模式，删除实验 delta=-0.875
- 级联 491_aug1：预测规模 41.4608，主导模式 桥接式跨层传播模式，删除实验 delta=-1.1993

## 文件说明
- JSON 结果文件：wikipedia_gpu.json
- 当前摘要文件：wikipedia_gpu.md