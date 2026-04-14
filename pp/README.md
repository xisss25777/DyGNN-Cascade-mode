# 信息级联规模预测与关键传播模式识别

本项目围绕“基于动态图神经网络的信息级联规模预测与关键传播模式解释”实现，当前支持两个真实数据集：

- `sample_data/wikipedia.csv`
- `sample_data/reddit.csv`

每次只运行一个数据集，统一通过 [main.py](/Users/zhou/Desktop/霍尼韦尔系统开发/pp/main.py) 选择。

## 当前模型框架

代码主线如下：

1. [cascade_model/dynamic_graph.py](/Users/zhou/Desktop/霍尼韦尔系统开发/pp/cascade_model/dynamic_graph.py)
   将早期传播过程按时间片切分，构造动态图快照序列。
2. [cascade_model/dgnn.py](/Users/zhou/Desktop/霍尼韦尔系统开发/pp/cascade_model/dgnn.py)
   对每个时间片做图卷积编码，再通过 `GRU + attention` 聚合时间演化信息，预测最终传播规模。
3. [cascade_model/patterns.py](/Users/zhou/Desktop/霍尼韦尔系统开发/pp/cascade_model/patterns.py)
   输出关键传播模式识别结果。
4. [cascade_model/evaluation.py](/Users/zhou/Desktop/霍尼韦尔系统开发/pp/cascade_model/evaluation.py)
   计算误差指标并执行删除实验。

说明：

- `wikipedia` 和 `reddit` 当前默认走 DGNN 流程。
- `cascade` 和 `synthetic` 仍可走旧的特征工程基线流程，便于做对照实验。

## 数据说明

`wikipedia.csv` 与 `reddit.csv` 是同构交互流数据，程序会：

- 按 `item_id` 聚合成单个级联样本
- 根据时间顺序构造传播序列
- 在早期观测窗口内切成多个动态图快照
- 将节点属性、边结构、图级统计共同输入 DGNN

## 运行方式

`--dataset` 现在是必填参数，不会再默认跑虚拟数据。

运行 Wikipedia：

```bash
python3 main.py --dataset wikipedia --output outputs/wikipedia_result.json
```

运行 Reddit：

```bash
python3 main.py --dataset reddit --output outputs/reddit_result.json
```

如果你想显式指定输入路径，也可以写：

```bash
python3 main.py --input sample_data/wikipedia.csv --dataset wikipedia --output outputs/wikipedia_result.json
python3 main.py --input sample_data/reddit.csv --dataset reddit --output outputs/reddit_result.json
```

## 输出内容

每次运行会生成两类结果：

- `*.json`：结构化实验结果
- `*.md`：便于直接阅读和整理到论文中的实验摘要

例如运行 Wikipedia 后会生成：

- `outputs/wikipedia_result.json`
- `outputs/wikipedia_result.md`

控制台还会打印：

- 数据集名称
- 样本数与特征维数
- MAE / RMSE / MAPE
- Top 特征或注意力切片
- 代表性测试样本结果

## 模型评价与对比

如果需要在同一数据集上比较：

- 动态图特征基线模型
- DGNN 主模型

可以运行：

```bash
python3 compare_models.py --dataset wikipedia
python3 compare_models.py --dataset reddit
```

该命令会生成：

- `outputs/<dataset>_model_compare.json`
- `outputs/<dataset>_model_compare.md`

适合直接整理到论文“模型评价与比较分析”部分。

## 模型准确性与性能评价

如果需要汇总两个真实数据集上的准确性与性能评价结果，可以先分别运行：

```bash
python3 compare_models.py --dataset wikipedia
python3 compare_models.py --dataset reddit
python3 summarize_evaluation.py
```

最终会生成：

- `outputs/model_evaluation_summary.json`
- `outputs/model_evaluation_summary.md`

当前评价维度包括：

- 准确性：`MAE`、`RMSE`、`MAPE`
- 性能：运行时间 `Runtime`

当前实验结论摘要如下：

- 在 `Wikipedia` 数据集上，基线模型在 `MAE`、`RMSE`、`MAPE` 三项指标上均优于 DGNN。
- 在 `Reddit` 数据集上，DGNN 在 `MAE` 与 `RMSE` 上优于基线模型，但基线模型在 `MAPE` 上更优。
- 从时间开销看，DGNN 的训练时间明显高于基线模型，但其结构表达方式更符合“动态图神经网络”建模思路。

对应结果文件可直接查看：

- [outputs/model_evaluation_summary.md](/Users/zhou/Desktop/霍尼韦尔系统开发/pp/outputs/model_evaluation_summary.md)
- [outputs/model_evaluation_summary.json](/Users/zhou/Desktop/霍尼韦尔系统开发/pp/outputs/model_evaluation_summary.json)

## 图表生成

生成论文风实验图：

```bash
python3 generate_publication_figure.py
```

输出文件：

- [outputs/figures/publication_results_figure.svg](/Users/zhou/Desktop/霍尼韦尔系统开发/pp/outputs/figures/publication_results_figure.svg)

## 依赖说明

当前 DGNN 版本依赖：

- `torch`
- `numpy`

这些包目前安装在用户级 Python 目录，不是项目内单独虚拟环境。

## 可选模式

如果需要运行旧格式级联数据：

```bash
python3 main.py --input /path/to/cascades.csv --dataset cascade --output outputs/cascade_result.json
```

如果只想跑模拟数据：

```bash
python3 main.py --dataset synthetic --output outputs/synthetic_result.json
```

## 后续优化方向

- 将当前 `GraphConv + GRU + attention` 升级为更强的时间图网络结构
- 为 Reddit 重尾分布加入更稳的损失函数或分层训练策略
- 在解释层加入更严格的注意力可视化与反事实扰动分析
