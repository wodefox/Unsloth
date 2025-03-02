# Unsloth GUI Finetuner 🚀

**一个基于 Unsloth 的图形化大语言模型（LLM）微调工具，无需编写代码，一键启动高效训练！**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

![Demo](assets/demo.gif) <!-- 可替换为实际项目演示动图或截图 -->

## ✨ 项目亮点
- **零代码图形化界面**：通过 GUI 轻松配置模型、数据集和训练参数。
- **一键启动训练**：支持 Llama、Mistral 等模型的 4-bit 量化和 LoRA 微调。
- **实时监控**：训练过程可视化，实时显示损失曲线和 GPU 资源占用。
- **模型快速评测**：内置交互式对话测试，验证微调效果。
- **跨平台支持**：Windows / Linux / macOS 均可运行。

## 🛠️ 功能概览
1. **模型加载**：支持本地模型或 Hugging Face 仓库模型。
2. **数据管理**：上传 JSON/CSV 数据集，自动格式化提示模板。
3. **训练配置**：图形化调整学习率、批量大小、LoRA 参数等。
4. **训练监控**：实时显示损失、显存占用和训练进度。
5. **模型导出**：保存完整模型或 LoRA 适配器，支持 ONNX 格式。

## 📥 安装与运行

### 环境要求
- **操作系统**: Windows 10+, Linux (Ubuntu 20.04+), macOS 12+
- **GPU**: 推荐 NVIDIA GPU（至少 8GB 显存）
- **Python**: 3.8 或更高版本

### 快速开始
1. **克隆仓库**：
   ```bash
   git clone https://github.com/wodefox/Unsloth.git
   cd unsloth
2.**RUN**
```bash
   python w.py
