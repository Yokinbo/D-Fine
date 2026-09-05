# D-FINE 火力发电厂检测改进实验

本目录集中保存论文网络改进的实现、统一开关和消融 YAML。训练、验证和测试
均通过 `settings.py` 选择同一结构，避免权重与模型配置不一致。

## 保留的参照方案：QLCS + DSQC

- `qlcs.py`：查询引导的隐式部件采样（QLCS）。在仅有整厂框标注的条件下，
  从动态参考框内部学习潜在部件证据，主要改善复合目标表征与严格定位质量。
  已完成的三随机种子实验保留该模块，当前实现和超参数不作改动。
- `dsqc.py`：解码稳定性感知查询校准（DSQC）。比较倒数第二层与最后一层中
  同一查询的语义表示、预测框 IoU、中心和尺度变化，再对最终分类 logits 做
  有界残差校准。三随机种子结果显示，该模块主要改善测试集 P、R 和 F1。
- 新候选 `qcr.py`：质量约束排序正则（Quality-Constrained Ranking，QCR）。
  只在训练期约束可靠目标与低重叠困难负样本的置信排序；推理网络仍为
  QLCS+DSQC。尚无精度提升结论，不列为已验证的正式改进。

这里的“冻结保留”指 QLCS/DSQC 的实现、结构和超参数不改；不是冻结它们的
可训练权重。每个消融实验仍从相同官方预训练权重开始，正常训练全部网络。

DSQC 只接入最终解码层，几何稳定性信号全部停止梯度，不修改 D-FINE 的 FDR
回归路径和输出框。校准头末层使用零初始化，首次加载官方预训练权重时等价于
未启用 DSQC 的分类输出，降低新模块破坏既有 QLCS 能力的风险。

QCR 保留 VFL、FDR、GO-LSD 和匹配器，只额外计算一次最终正常查询的训练损失；
不修改辅助层、编码器、DN 损失，也没有新的推理参数或算子。实现公式、文献依据、
风险与实验判据见 [QCR_EXPERIMENT.md](QCR_EXPERIMENT.md)。

## 已归档的 QFBCG、QACG、MGCA 与 SHEA

`qfbcg.py` 及其 YAML 仅为复现已完成实验而保留，不再是当前正式方案。三随机种子
结果显示，QLCS+QFBCG 相比 QLCS 的验证集 P、F1、AP75 和 mAP50:95 均下降，
测试集 P 与 mAP50:95 也下降。这不足以支持保留该实现；下降机制仍需逐框误差
分析，不能只凭汇总指标断言是框外背景导致。

`qacg.py` 及其 YAML 同样只为复现保留。QACG 的三次验证均值有所提高，但
独立测试均值 AP50/F1 仅为 0.8984/0.8467，低于 QLCS+DSQC 的
0.9228/0.8782；两个共同随机种子的配对结果也下降。没有呈现一致收益，
不作为正式第三模块。旧 QACG 第三次文件名为 SD16，不能冒充 SD18 配对。

`mgca.py` 及其 YAML 只用于复现。MGCA 三次验证均值 AP50/F1 为
0.9208/0.8426，低于 QLCS+DSQC 的 0.9224/0.8509；独立测试均值
AP50/P/F1 为 0.9134/0.8173/0.8448，也低于前一阶段的
0.9228/0.8573/0.8782。虽然 AP75 提高，但不符合本文以 AP50 和 F1 为核心
的目标；汇总指标不能证明具体下降机制。

`shea.py` 及其 YAML 保留复现。SHEA 三次验证均值 AP50/P/R/F1 为
0.9200/0.8047/0.8970/0.8478；测试均值为
0.9072/0.8333/0.8865/0.8590。相较 QLCS+DSQC 未满足核心指标目标，当前关闭。

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
| `mgca` | D-FINE-M + MGCA | 旧失败实验复现 |
| `qlcs_dsqc_mgca` | D-FINE-M + QLCS + DSQC + MGCA | 旧失败实验复现 |
| `qlcs_dsqc_shea` | D-FINE-M + QLCS + DSQC + SHEA | 旧实验复现 |
| `qlcs_dsqc_qcr` | QLCS + DSQC 网络，训练时附加 QCR | 当前待验证候选 |
| `qfbcg` | D-FINE-M + QFBCG | 旧失败实验复现 |
| `qlcs_qfbcg` | D-FINE-M + QLCS + QFBCG | 旧失败实验复现 |

当前选择为 `qlcs_dsqc_qcr`，对应配置
`dfine_hgnetv2_m_qlcs_dsqc_qcr.yml`。直接继承 QLCS+DSQC 的配置；
QFBCG、QACG、MGCA、SHEA 均关闭。QCR 系数、候选数和渐入轮数均放在该 YAML。

旧 SHEA 的设计来源（仅用于复现记录）：

SHEA 受 [Decoupled DETR（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Decoupled_DETR_Spatially_Disentangling_Localization_and_Classification_for_Improved_End-to-End_ICCV_2023_paper.html)
中分类与定位需要不同区域证据的结论，以及 [Salience DETR（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Salience_DETR_Enhancing_Detection_Transformer_with_Hierarchical_Salience_Filtering_Refinement_CVPR_2024_paper.html)
的判别性证据与查询细化思想启发。当前实现是针对 QLCS 潜在部件和 D-FINE
后期分类查询设计的轻量适配器，并非复刻上述论文模块。

## 节省时间的实验顺序

1. 保持 `MODEL_SIZE = "m"`、VRAC 关闭、学习率 `2e-4`，其余训练设置与既有
   基线和 QLCS 实验完全一致。
2. 所有模型从同一官方 D-FINE-M 预训练权重重新训练，不从 QLCS 权重继续训练。
3. 先用种子 3407 训练 `qlcs_dsqc_qcr`，检查头 5～10 轮 QCR 样本数和损失量级；
   这是实现/激活检查，不据此断言精度成功或失败。不在途中反复调参。
4. 首轮按原定 150 轮完成，与同种子 QLCS+DSQC 配对比较。有潜力后补种子
   18、2026，报告均值、样本标准差和每种子变化。若根据首轮调整配置，它属于
   新候选实验，不能与旧配置混合平均。
5. 正式顺序消融表建议为
   `baseline → +QLCS → +QLCS+DSQC → +QLCS+DSQC（训练加QCR）`。
   未通过时最后一行可作为无效尝试记录，不能强行写为有效创新。

## 判定 QCR 是否保留

采用重新验证后的 QLCS+DSQC 三次验证均值：AP50 0.9224、P 0.8181、
R 0.8869、F1 0.8509、AP75 0.5871、mAP50:95 0.5837。以下为预先约定的
工程筛选目标，不是统计显著性检验，也不是对收益的承诺：

- 首轮 SD3407 与前一阶段同种子结果 AP50/P/R/F1
  0.9274/0.7914/0.8970/0.8409 配对比较；P 与 F1 应同时提高，AP50 下降
  不得超过 0.005，R 下降不得超过 0.010；
- 三次验证均值 AP50 不低于 0.9224、F1 高于 0.8509，P 高于 0.8181；
  理想增益为 P 至少提高 0.010、F1 至少提高 0.005；
- 至少两个随机种子的 P/F1 同向改善，不能依赖单个最好结果；
- P 的提高不能以 AP50 或 R 明显下降为代价；
- QCR 推理参数/FLOPs 与 QLCS+DSQC 相同；训练耗时会增加，实际 FPS 仍须
  在相同硬件、输入、精度模式和测速口径下测量，不能凭结构相同伪造时间。

继续设计与筛选只用训练/验证集。只查看测试数据不会改动权重，但若测试结果影响
模块去留或下一版设计，它就参与了选择；旧测试集不能再声称是完全未参与选择的
最终留出集。若条件允许，增加未查看过的独立最终留出集，否则如实披露此限制。
