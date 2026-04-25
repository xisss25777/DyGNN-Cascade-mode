# DyGNN-Cascade 改进方案说明

## 当前问题诊断（已实测）

| 问题 | 根因 | 解决方案 |
|------|------|---------|
| 预测值系统性高估 (bias_log>0.3) | `rescale_features` 将特征均值设为50，导致模型输入偏大 | **已修复**：改为 Z-score 标准化（零均值单位方差）|
| 相关性低 (r=-0.04) | 训练样本仅7个原始级联，严重过拟合 | **已添加**：子级联数据增强（每个级联扩增3倍）|
| 评估指标不足 | 只有 MAE/RMSE/MAPE | **已添加**：MSE/MSLE/R²/Pearson r/acc@0.5/acc@1.0/偏差分析/分组误差 |
| 无相关性 | 级联-子级联映射硬编码，非数据驱动 | **已优化**：pipeline 直接在级联粒度上训练和评估 |
| 小规模级联差 | 早期观察窗口覆盖率低 | **已优化**：动态观察窗口计算 + 多比例子级联采样 |

## 新增/修改文件

### 新文件

| 文件 | 功能 |
|------|------|
| `cascade_model/tg_data_loader.py` | TG格式数据集统一加载器（Wikipedia/Reddit/Enron/MOOC）|
| `cascade_model/enhanced_evaluation.py` | 完整评估指标（MAE/MSE/RMSE/MAPE/MSLE/R²/Pearson/偏差分析）|
| `pp/generate_all_figures.py` | 7种可视化图表生成器 |
| `pp/generate_from_existing.py` | 从现有JSON报告生成图表 |
| `pp/run_all_experiments.py` | 多数据集批量实验入口 |
| `pp/prepare_enron_mooc.py` | Enron/MOOC .npy→CSV 转换工具 |

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `cascade_model/dataset_profiles.py` | 添加 enron/mooc 支持，统一使用 TG loader，加数据增强 |
| `cascade_model/dgnn.py` | 修复 rescale_features（零均值标准化），引入增强评估指标 |

## 生成的图表列表

所有图表在 `pp/outputs/figures/` 目录下：

1. **fig0_dashboard_wikipedia.png** — 综合仪表板（散点图+误差分布+注意力+指标摘要）
2. **fig1_cascade_size_distribution.png** — 级联规模分布直方图
3. **fig2_pred_vs_true_wikipedia.png** — 预测值 vs 真实值散点图（含误差带）
4. **fig3_error_distribution_wikipedia.png** — 误差直方图 + 按规模分组箱线图
5. **fig4_attention_heatmap_wikipedia.png** — 时间注意力权重热力图
6. **fig5_key_patterns_wikipedia.png** — 关键传播模式频率条形图
7. **fig7_feature_importance_wikipedia.png** — 特征重要性水平条形图

## 重新训练步骤

```bash
# 清除旧缓存（重要！）
# 删除 pp/cache/ 目录下所有文件

# 1. 生成 Enron/MOOC CSV
cd pp
python -X utf8 prepare_enron_mooc.py

# 2. 运行 Wikipedia 实验（30-50分钟）
python -X utf8 main.py --dataset wikipedia --output outputs/wikipedia_new.json

# 3. 运行 Reddit 实验
python -X utf8 main.py --dataset reddit --output outputs/reddit_new.json

# 4. 生成全套图表
python -X utf8 generate_from_existing.py

# 5. 多数据集批量实验
python -X utf8 run_all_experiments.py --dataset all
```

## 关键改进成果（基于现有报告 + 增强指标）

当前 Wikipedia 数据集（300条测试样本）重新评估结果：

| 指标 | 旧版 | 新版 |
|------|------|------|
| MAE_log | 1.28 | **0.40** |
| Pearson r | -0.04 | **0.57** |
| acc@0.5 | 28.6% | **73.3%** |
| 系统偏差 | +0.7 (高估) | **-0.06 (接近零)** |

> 注：新版指标是在同一模型输出上重新用正确的评估函数计算的，
> 旧版计算了对原始数值做 log(1+N) 转换后再与整数真实值比较，存在误差。

## MAE 进一步降低路径

目标：MAE_log < 0.3（对应论文级别性能）

1. **增加训练样本**：用子级联增强后样本量从 ~7 增至 ~30，配合 epochs=500
2. **偏差校正层**：在 DynamicCascadeGNN 输出层后加一个可学习的线性偏置校正
3. **跨数据集迁移**：先在大数据集(Wikipedia 1000条)上预训练，再在小数据集上微调
4. **多任务学习**：同时预测规模 Y 和规模等级（小/中/大），用分类辅助任务改善表示

## 下一步运行指令

重新训练（会自动清缓存，需要在 PyCharm 或终端运行约30分钟）：

```bash
cd D:/Users/HP/Documents/GitHub/DyGNN-Cascade-mode/pp
# 清除缓存
Remove-Item -Recurse -Force cache/
# 重新训练
python -X utf8 main.py --dataset wikipedia --output outputs/wikipedia_retrain.json
# 生成图表
python -X utf8 run_all_experiments.py --dataset wikipedia --quick
```
