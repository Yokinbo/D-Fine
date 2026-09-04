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
- `shea.py`：显著—整体证据对齐（SHEA）。复用 QLCS 的框内潜在部件，分别
  构造查询相关显著证据和完整厂区整体证据，只细化后两层送入分类头的查询，
  目标是抑制仅由单个相似工业结构触发的误检，提高 P 和 F1。

DSQC 只接入最终解码层，几何稳定性信号全部停止梯度，不修改 D-FINE 的 FDR
回归路径和输出框。校准头末层使用零初始化，首次加载官方预训练权重时等价于
未启用 DSQC 的分类输出，降低新模块破坏既有 QLCS 能力的风险。

SHEA 的查询和部件条件输入均停止梯度，避免新增分类分支反向扰动已经确定的
QLCS 和回归路径；正常查询经过分类专用的有界残差，训练期去噪查询保持原样。
输出投影零初始化，首次接入时严格等价于 QLCS+DSQC。该模块沿用 D-FINE
原始损失，不修改匹配、最终 logits 公式或框回归结果。

## 已归档的 QFBCG、QACG 与 MGCA

`qfbcg.py` 及其 YAML 仅为复现已完成实验而保留，不再是当前正式方案。三随机种子
结果显示，QLCS+QFBCG 相比 QLCS 的验证集 P、F1、AP75 和 mAP50:95 均下降，
测试集 P 与 mAP50:95 也下降。其“扩大框外环带作为背景候选”的假设不适合边界
复杂、周围设施与主体相关的大型火电厂目标，因此不继续微调 QFBCG。

`qacg.py` 及其 YAML 同样只为复现保留。QACG 的三次验证均值有所提高，但
独立测试均值 AP50/F1 仅为 0.8984/0.8467，低于 QLCS+DSQC 的
0.9228/0.8782；两个共同随机种子的配对结果也下降。该结果说明继续在最终
分类 logits 上叠加查询竞争校准容易适配验证分布，不能作为正式第三模块。

`mgca.py` 及其 YAML 只用于复现。MGCA 三次验证均值 AP50/F1 为
0.9208/0.8426，低于 QLCS+DSQC 的 0.9224/0.8509；独立测试均值
AP50/P/F1 为 0.9134/0.8173/0.8448，也低于前一阶段的
0.9228/0.8573/0.8782。它虽然提高 AP75，但扩大共享上下文同时削弱了
分类判别能力，不符合本文以 AP50 和 F1 为核心的目标。

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
| `qlcs_dsqc_shea` | D-FINE-M + QLCS + DSQC + SHEA | 新第三步累计消融/当前模式 |
| `qfbcg` | D-FINE-M + QFBCG | 旧失败实验复现 |
| `qlcs_qfbcg` | D-FINE-M + QLCS + QFBCG | 旧失败实验复现 |

当前选择为 `qlcs_dsqc_shea`，对应配置
`dfine_hgnetv2_m_qlcs_dsqc_shea.yml`。该配置直接继承已确定的 QLCS+DSQC
配置：QLCS 仍在第 0～3 层启用，DSQC 只在第 3 层启用；SHEA 共享参数地
处理第 2、3 层正常查询的分类特征，QFBCG、QACG 和 MGCA 均关闭。

SHEA 受 [Decoupled DETR（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Decoupled_DETR_Spatially_Disentangling_Localization_and_Classification_for_Improved_End-to-End_ICCV_2023_paper.html)
中分类与定位需要不同区域证据的结论，以及 [Salience DETR（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Salience_DETR_Enhancing_Detection_Transformer_with_Hierarchical_Salience_Filtering_Refinement_CVPR_2024_paper.html)
的判别性证据与查询细化思想启发。当前实现是针对 QLCS 潜在部件和 D-FINE
后期分类查询设计的轻量适配器，并非复刻上述论文模块。

## 节省时间的实验顺序

1. 保持 `MODEL_SIZE = "m"`、VRAC 关闭、学习率 `2e-4`，其余训练设置与既有
   基线和 QLCS 实验完全一致。
2. 所有模型从同一官方 D-FINE-M 预训练权重重新训练，不从 QLCS 权重继续训练。
3. 先用较弱且更有代表性的随机种子 3407 训练 `qlcs_dsqc_shea` 作快速筛选，
   与同种子的 QLCS+DSQC 结果直接比较。
4. 单次结果有潜力后，再补种子 18、2026，并报告三次均值和标准差。模型选择
   只使用验证集；测试集留到结构和超参数固定后进行一次最终评价。
5. 正式顺序消融表建议为
   `baseline → +QLCS → +QLCS+DSQC → +QLCS+DSQC+SHEA`。

## 判定 SHEA 是否保留

采用重新验证后的 QLCS+DSQC 三次验证均值：AP50 0.9224、P 0.8181、
R 0.8869、F1 0.8509、AP75 0.5871、mAP50:95 0.5837。SHEA 的核心判据如下：

- 首轮 SD3407 与前一阶段同种子结果 AP50/P/R/F1
  0.9274/0.7914/0.8970/0.8409 配对比较；P 与 F1 应同时提高，AP50 下降
  不得超过 0.005，R 下降不得超过 0.010；
- 三次验证均值 AP50 不低于 0.9224、F1 高于 0.8509，P 高于 0.8181；
  理想增益为 P 至少提高 0.010、F1 至少提高 0.005；
- 至少两个随机种子的 P/F1 同向改善，不能依赖单个最好结果；
- P 的提高不能以 AP50 或 R 明显下降为代价；
- 参数量和 FLOPs 的小幅增加与精度收益相称。

验证集通过后才能固定结构并评估独立测试集。若测试集 AP50/F1 再次明显下降，
即使验证均值提高也必须归档；不能继续根据测试集反向调整模块。
