# D-FINE 火力发电厂检测改进实验

本目录集中保存论文网络改进的实现、统一开关和消融 YAML。训练、验证和测试
均通过 `settings.py` 选择同一结构，避免权重与模型配置不一致。

## 当前正式方案

- `qlcs.py`：查询引导的隐式部件采样（QLCS）。在仅有整厂框标注的条件下，
  从动态参考框内部学习潜在部件证据，主要改善复合目标表征与严格定位质量。
  已完成的三随机种子实验保留该模块，当前实现和超参数不作改动。
- `dsqc.py`：解码稳定性感知查询校准（DSQC）。比较倒数第二层与最后一层中
  同一查询的语义表示、预测框 IoU、中心和尺度变化，再对最终分类 logits 做
  有界残差校准。三随机种子结果显示，该模块主要改善测试集 P、R 和 F1。
- `mgca.py`：多粒度上下文聚合（MGCA）。在解码器展平多尺度特征之前，仅对
  P4/P5 并行提取局部、区域和长程上下文，再进行逐位置动态聚合。该模块面向
  大型火电厂目标及相似工业背景，目标是改善跨数据划分泛化、AP50 和 F1。

DSQC 只接入最终解码层，几何稳定性信号全部停止梯度，不修改 D-FINE 的 FDR
回归路径和输出框。校准头末层使用零初始化，首次加载官方预训练权重时等价于
未启用 DSQC 的分类输出，降低新模块破坏既有 QLCS 能力的风险。

MGCA 使用共享轻量参数处理 P4/P5，保持 P3 原样以保护细节。三路分别采用
3×3 局部卷积、1×7/7×1 区域卷积和平均池化后的 1×11/11×1 长程卷积；
输出投影零初始化，并以有界残差接入预训练特征。它沿用 D-FINE 原始损失，
不修改匹配、分类头和回归头。

## 已归档的 QFBCG 与 QACG

`qfbcg.py` 及其 YAML 仅为复现已完成实验而保留，不再是当前正式方案。三随机种子
结果显示，QLCS+QFBCG 相比 QLCS 的验证集 P、F1、AP75 和 mAP50:95 均下降，
测试集 P 与 mAP50:95 也下降。其“扩大框外环带作为背景候选”的假设不适合边界
复杂、周围设施与主体相关的大型火电厂目标，因此不继续微调 QFBCG。

`qacg.py` 及其 YAML 同样只为复现保留。QACG 的三次验证均值有所提高，但
独立测试均值 AP50/F1 仅为 0.8984/0.8467，低于 QLCS+DSQC 的
0.9228/0.8782；两个共同随机种子的配对结果也下降。该结果说明继续在最终
分类 logits 上叠加查询竞争校准容易适配验证分布，不能作为正式第三模块。

## 统一开关

在 `settings.py` 修改 `IMPROVEMENT_MODE`：

| 模式 | 网络结构 | 用途 |
|---|---|---|
| `baseline` | 原始 D-FINE-M | 基线 |
| `qlcs` | D-FINE-M + QLCS | 已完成的第一模块消融 |
| `dsqc` | D-FINE-M + DSQC | 新第二模块独立消融 |
| `qlcs_dsqc` | D-FINE-M + QLCS + DSQC | 已完成的第二步累计消融 |
| `qacg` | D-FINE-M + QACG | 旧失败实验复现 |
| `qlcs_dsqc_qacg` | D-FINE-M + QLCS + DSQC + QACG | 旧失败实验复现 |
| `mgca` | D-FINE-M + MGCA | 新第三模块独立消融 |
| `qlcs_dsqc_mgca` | D-FINE-M + QLCS + DSQC + MGCA | 新第三步累计消融/当前模式 |
| `qfbcg` | D-FINE-M + QFBCG | 旧失败实验复现 |
| `qlcs_qfbcg` | D-FINE-M + QLCS + QFBCG | 旧失败实验复现 |

当前选择为 `qlcs_dsqc_mgca`，对应配置
`dfine_hgnetv2_m_qlcs_dsqc_mgca.yml`。该配置直接继承已确定的 QLCS+DSQC
配置：QLCS 仍在第 0～3 层启用，DSQC 只在第 3 层启用；新增 MGCA 仅处理
送入解码器的 P4/P5，QACG 关闭。

MGCA 的结构动机来自遥感检测中的选择性多范围上下文研究：[PKINet（CVPR
2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html)
使用多核与上下文锚处理尺度变化和多样上下文，[LSKNet（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.html)
使用动态感受野选择适配不同目标。当前实现只吸收其“多粒度、动态选择”的
思想，并针对 D-FINE 的多尺度 memory 设计轻量适配器，并非复刻其骨干网络。

## 节省时间的实验顺序

1. 保持 `MODEL_SIZE = "m"`、VRAC 关闭、学习率 `2e-4`，其余训练设置与既有
   基线和 QLCS 实验完全一致。
2. 所有模型从同一官方 D-FINE-M 预训练权重重新训练，不从 QLCS 权重继续训练。
3. 先用较弱且更有代表性的随机种子 3407 训练 `qlcs_dsqc_mgca` 作快速筛选，
   与同种子的 QLCS+DSQC 结果直接比较。
4. 单次结果有潜力后，再补种子 18、2026，并报告三次均值和标准差。模型选择
   只使用验证集；测试集留到结构和超参数固定后进行一次最终评价。
5. 正式顺序消融表建议为
   `baseline → +QLCS → +QLCS+DSQC → +QLCS+DSQC+MGCA`；篇幅允许时再补
   `mgca` 单模块行。

## 判定 MGCA 是否保留

采用重新验证后的 QLCS+DSQC 三次验证均值：AP50 0.9224、P 0.8181、
R 0.8869、F1 0.8509、AP75 0.5871、mAP50:95 0.5837。MGCA 的核心判据如下：

- 首轮 SD3407 的 AP50 不低于 0.9274、F1 不低于 0.8409，且至少一项有
  明确提高；任何核心指标下降超过 0.010 即停止该方案；
- 三次验证均值 AP50 不低于 0.9224、F1 不低于 0.8509，最好分别提高
  0.003 和 0.005 以上；
- 至少两个随机种子的 AP50/F1 同向改善，不能依赖单个最好结果；
- P 或 R 的提高不能以另一项明显下降为代价；
- 参数量和 FLOPs 的小幅增加与精度收益相称。

验证集通过后才能固定结构并评估独立测试集。若测试集 AP50/F1 再次明显下降，
即使验证均值提高也必须归档；不能继续根据测试集反向调整模块。
