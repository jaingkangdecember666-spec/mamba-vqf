# 模型和一键训练

`weights/` 内包含 C038 的 4 个实际推理权重和旧 TCN。C038 的增益、门控、更新间隔、组件路径及 SHA-256 见 [recipe.json](recipe.json)。C038 使用内禀 Mamba、C013 读出、C020 时序 Mamba 和 C038 路由。旧 TCN 作为对照。

## 重训最后一层

在安装 Python 3.12 和 PowerShell 的 Windows 上，从仓库根目录运行。

```powershell
.\model\train.ps1
```

脚本按登记的 160 步 AdamW 重训 C023，使用 SLSQP 重训 C035，再将两套读出的权重逐元素取中点得到 C038。使用 141 条外部来源的 15,680 个归档时刻，沿用冻结的 Mamba 主干和归一化。训练基于已归档的外部训练张量，适用于复现最后两次拟合。完整主干重训还需要原始训练数据、原实验环境和脚本压缩包中的前序训练入口。

输出是 `model/trained/router.pt` 和 `model/trained/verification.json`。训练后核对最大参数差和共同训练时刻的预测角差，容差分别为 0.00001 和 0.0001°。使用已安装依赖的解释器，或指定新输出目录时可执行以下命令。

```powershell
.\model\train.ps1 -Python C:\path\to\python.exe -Output .\retrained-c038
```

在 FZR-MiniPC 原实验环境实测重训，C035 求解 67 次迭代。15,680 个时刻的预测最大角差约为 6.59 × 10⁻¹⁴°。

上游第三方预训练权重按各自许可和固定版本从发布方获取。[TimesFM 3.0 PyTorch](https://huggingface.co/google/timesfm-3.0-pytorch/tree/43046b85ec22d584a13f8098c2ed39c889e129c2) · [Timestral 209M](https://huggingface.co/mistralai/Timestral-209M/tree/9874d252ab0aae16f7f7c999f98d6ad7d8365)
