# ai4sensors

## 六轴 IMU 倾角实验

冻结 Mamba 模型 C038 在八套主评数据上均优于默认 FullVQF。加入四套外部迁移检验后为 9/12 套。自采 R5/R6 共 20 条，平均倾角 RMSE 从 2.05962° 降至 1.92030°。固定 1.5 秒的 VQF 为 1.77569°。

- [离线报告](results/report.html)下载后用浏览器打开。[数据集划分](results/dataset-splits.md) · [主评 365 条成绩](results/main.csv) · [外部迁移逐条成绩](results/transfer.csv)
- [模型和一键训练](model/README.md)包含 C038 四个推理所需的权重文件、旧 TCN 对照、外部训练张量和 SHA-256。训练只重做最终的读出层，Mamba 主干直接使用归档权重。
- [实验脚本压缩包](experiment-source.zip)保留原目录结构。完整原生重放还需原数据及对应的评分与训练中间产物。

仓库保留精简的实验交付。原始采集、全部中间缓存、安装环境和第三方原版大模型由其本地档案或发布方保存。
