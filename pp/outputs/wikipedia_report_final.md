# wikipedia 数据集实验摘要

## 核心结果
- 样本数：1000
- 特征数：29
- 融合权重 alpha：N/A
- MAE：8.9415
- RMSE：23.0808
- MAPE：0.4998

## Top 特征
- slice_12_attention: 0.292496
- slice_11_attention: 0.211724
- slice_10_attention: 0.153201
- slice_9_attention: 0.105022
- slice_8_attention: 0.076474
- slice_7_attention: 0.053353
- slice_6_attention: 0.038521
- slice_5_attention: 0.027642

## 代表性测试样本
- 级联 424：预测规模 9.17，主导模式 早期快速扩散模式，删除实验 delta=-10.7556
- 级联 496：预测规模 22.7804，主导模式 早期快速扩散模式，删除实验 delta=7.079
- 级联 705：预测规模 8.483，主导模式 早期快速扩散模式，删除实验 delta=6.8756
- 级联 950：预测规模 6.5385，主导模式 早期快速扩散模式，删除实验 delta=-0.6804
- 级联 185：预测规模 26.7204，主导模式 早期快速扩散模式，删除实验 delta=3.9987

## 文件说明
- JSON 结果文件：wikipedia_report_final.json
- 当前摘要文件：wikipedia_report_final.md