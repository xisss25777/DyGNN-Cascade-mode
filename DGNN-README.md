# 信息级联预测框架

## 项目简介
本项目实现了基于动态图神经网络（DGNN）的信息级联预测模型，用于预测信息在社交网络中的传播规模。

## 核心功能
- **动态图神经网络**：使用GCN和GRU结合的架构，捕捉信息传播的时空特征
- **超参数优化**：自动搜索最佳模型参数，提高预测性能
- **缓存加速**：使用缓存机制加速数据加载和模型训练
- **早停机制**：当模型性能不再改善时自动停止训练，避免过拟合
- **特征重要性分析**：识别影响信息传播的关键特征
- **模式识别**：识别信息传播的关键模式

## 目录结构
```
├── cascade_model/          # 核心模型代码
│   ├── __init__.py         # 包初始化文件
│   ├── config.py           # 配置文件
│   ├── data.py             # 数据处理
│   ├── dgnn.py             # 动态图神经网络模型
│   ├── dynamic_graph.py    # 动态图构建
│   ├── cache_utils.py      # 缓存工具
│   ├── hyperparameter_search.py # 超参数优化
│   ├── evaluation.py       # 评估指标
│   └── dataset_profiles.py # 数据集配置
├── pp/                     # 运行脚本
│   ├── main.py             # 主脚本
│   ├── compare_models.py   # 模型比较
│   └── generate_publication_figure.py # 可视化
├── README.md               # 项目说明
└── .gitignore              # Git 忽略文件
```

## 环境搭建
1. 克隆仓库：
   ```bash
   git clone <仓库地址>
   cd <项目目录>
   ```

2. 创建虚拟环境：
   ```bash
   python -m venv .venv
   ```

3. 激活虚拟环境：
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`

4. 安装依赖：
   ```bash
   pip install torch torchvision torchaudio
   ```

## 使用方法
### 运行模型训练
```bash
.venv\Scripts\python.exe pp\main.py --dataset wikipedia
```

### 运行超参数搜索
```bash
.venv\Scripts\python.exe cascade_model\hyperparameter_search.py
```

### 运行模型比较
```bash
.venv\Scripts\python.exe pp\compare_models.py --dataset wikipedia
```

### 生成可视化结果
```bash
.venv\Scripts\python.exe pp\generate_publication_figure.py
```

## 数据集
支持的数据集：
- Wikipedia
- Reddit

## 评估指标
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

## 核心技术
- **动态图神经网络**：捕捉信息传播的时空特征
- **注意力机制**：关注重要的时间切片
- **时空掩码**：提高模型的可解释性
- **联合损失函数**：同时优化预测性能和解释能力
- **超参数优化**：自动搜索最佳模型参数
- **缓存加速**：提高训练速度和效率

## 贡献
欢迎提交 Issue 和 Pull Request！
