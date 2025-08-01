# Local M2 Mac Setup Guide

恭喜！你的 M2 Mac 已经成功设置为运行机械可解释性评估。

## ✅ 设置状态

- **PyTorch**: ✅ 已安装 (版本 2.7.1)
- **MPS 加速**: ✅ 可用 (Apple Silicon 加速)
- **Transformers**: ✅ 已安装
- **可视化库**: ✅ 已安装 (Matplotlib, Seaborn, Plotly)
- **Jupyter**: ✅ 已安装

## 🚀 快速开始

### 1. 启动环境

```bash
# 激活虚拟环境
source venv/bin/activate

# 启动 Jupyter notebook
jupyter notebook
```

### 2. 访问 Jupyter

打开浏览器访问: `http://localhost:8888`

### 3. 运行第一个笔记本

在 Jupyter 中打开: `notebooks/01_activations_analysis_local.ipynb`

## 📚 评估内容

### 核心问题解答

1. **什么是激活 (Activations)?**
   - 神经网络处理输入后的输出值
   - 捕获模型的内部表示
   - 对可解释性至关重要

2. **如何找到特定 token 的激活?**
   - 使用 PyTorch hooks 提取激活
   - 在特定层注册钩子函数
   - 分析每个 token 的激活模式

3. **稀疏自编码器 (SAE) 的目的?**
   - 从神经激活中提取特征
   - 实现单义性 (monosemanticity)
   - 提高可解释性

### 实验内容

1. **激活分析** (`01_activations_analysis_local.ipynb`)
   - 提取 GPT-2 的激活
   - 可视化激活热图
   - 比较不同 token 的激活模式

2. **稀疏自编码器** (即将添加)
   - 训练 SAE 网络
   - 特征提取和分析
   - 单义性研究

3. **Bonus: 角色扮演模型** (即将添加)
   - 扩展到 Stheno-8B 等模型
   - 多层 MLP 分析

## 🔧 技术细节

### M2 Mac 优化

- **MPS 加速**: 使用 Metal Performance Shaders
- **内存优化**: 适合中等规模模型
- **性能**: 比 CPU 快 3-5 倍

### 模型选择

- **GPT-2**: 适合入门和实验
- **GPT-2 Medium**: 更复杂的分析
- **Stheno-8B**: Bonus 任务 (需要更多内存)

## 📊 预期结果

### 激活分析
- 不同 token 的独特激活模式
- 语义相似词的相似激活
- 位置效应分析

### SAE 分析
- 特征提取和可视化
- 单义性度量
- 稀疏性分析

## 🎯 评估提交

### 必需内容
1. **书面报告** (PDF/Markdown)
   - 实验设置和结果
   - 分析和见解
   - 研究提案

2. **代码和笔记本**
   - 所有 Jupyter 笔记本
   - 源代码文件
   - 实验结果

### 加分项
- 扩展到其他模型
- 创新的可视化
- 深入的理论分析

## 🆘 故障排除

### 常见问题

1. **模型下载失败**
   ```bash
   # 使用镜像或代理
   export HF_ENDPOINT=https://hf-mirror.com
   ```

2. **内存不足**
   - 使用更小的模型
   - 减少批次大小
   - 使用梯度检查点

3. **MPS 错误**
   - 回退到 CPU: `device = "cpu"`
   - 检查 PyTorch 版本

## 📈 性能基准

在 M2 Mac 上的预期性能:

| 任务 | 模型 | 时间 | 内存 |
|------|------|------|------|
| 激活提取 | GPT-2 | ~2s | ~2GB |
| SAE 训练 | GPT-2 | ~10min | ~4GB |
| 可视化 | - | ~1s | ~1GB |

## 🎉 开始你的分析！

现在你可以开始进行机械可解释性分析了。记住:

1. **理解概念**: 先理解激活和 SAE 的基本概念
2. **实验为主**: 多运行实验，观察结果
3. **记录发现**: 详细记录你的观察和见解
4. **创新思考**: 提出自己的研究想法

祝你评估顺利！🚀 