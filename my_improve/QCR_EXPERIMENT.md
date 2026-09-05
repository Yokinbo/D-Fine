# QCR：质量约束排序正则（实验候选）

日期：2026-09-05。保留 QLCS+DSQC；不改变它们的实现或超参数，不冻结训练权重。
本方案尚未完成真实数据训练，不能宣称有效、显著提升或原创性已获证明。

## 为什么选择这个方向

论文核心是 AP50 和 F1@0.5，不是单独让 P 变大。此前多种第三模块没有稳定提升
两项核心指标，继续叠加分类特征分支缺少证据。QCR 改为对训练时的相对置信排序
施加约束，同时保留原 VFL 的绝对分数监督，推理网络不增加模块。

本地 QLCS+DSQC SD3407 验证报告在 conf=0.5 下 TP=148、FP=39、FN=17；
提高阈值会改变 P/R。它只说明存在固定阈值误检与分数区分问题，不能证明这些 FP
主要是背景，也不能排除定位偏差、重复框或标注问题。QCR 的可检验假设是：
可靠真目标与低重叠高分负候选的排序约束，可能改善当前 P/F1 而不牺牲 AP50。
如果主要错误不属于这个类型，该方法可能无效。

## 文献依据与边界

- [RC-DETR，Neural Networks 2025](https://www.sciencedirect.com/science/article/pii/S0893608024008402)：
  在行人检测中用排序式对比约束区分相似前景/背景，无额外推理开销。这是
  本方案的方向性启发；未获得可核验的作者实现，本代码不是它的复现。
- [Bucketed Ranking-based Losses，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7634_ECCV_2024_paper.php)：
  排序损失可用于包括 Co-DETR 在内的检测训练。其分桶和特定梯度算法与这里的
  softplus 正则不同，不应把本实现称为 BRS。
- [DEIM，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html)：
  MAL 有 D-FINE 相关实验依据，但[官方配置](https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/base/deim.yml)
  使用 gamma=1.5，[实现](https://github.com/Intellindust-AI-Lab/DEIM/blob/main/engine/deim/deim_criterion.py)
  将正例目标设为 IoU^gamma。比如 IoU=0.6 时目标约为0.465，对固定0.5阈值
  召回存在风险；这是数学上的可能性，不是实测结论。本轮不替换 VFL 为 MAL。
- [PaQ-DETR，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Kang_PaQ-DETR_Learning_Pattern_and_Quality-Aware_Dynamic_Queries_for_Object_Detection_CVPR_2026_paper.pdf)：
  动态模式查询和质量感知监督也是候选思路，但改动范围更大。当前单类别任务与
  高召回、相对低 P 的证据，不足以优先支持这类大幅查询重构。

原 D-FINE VFL 已有 IoU 软标签及负例 p² 加权。QCR 不能被描述为“首次引入质量
监督/困难负样本训练”；新增点是有可靠性筛选与跨层正例保护的**联合正负排序**。
这些基础操作本身不是自动成立的论文创新。若实验有效，仍需说明与已有排序方法
的具体差异，补充机制消融和误检可视化。

## 算法与固定初值

只处理最终正常查询的最终分类 logits（包括已有 DSQC 校准），不处理 DN。

1. 可靠正例：最终匹配到 GT，且预测框与匹配 GT 的 IoU >= 0.5。
2. 负候选：未落入最终/辅助/预输出/编码器匹配的 GO 并集，且与所有 GT 的
   最大 IoU < 0.3。IoU 处在模糊区间的候选不受新增负约束。
3. 每图取负候选中得分最高的至多8个，与可靠正例配对。
4. 每对损失为 `IoU_pos * stopgrad(sigmoid(z_neg)^2) * softplus(0.5 + z_neg - z_pos)`。
   IoU、配对选择与权重均停止梯度。先按配对数量平均，再按批次图像数平均；
   **不除以权重之和**，否则会抵消易负例衰减。
5. 空 GT 图用被选负候选的 `p² * softplus(z_neg)` 均值；有 GT 但无可靠正例、
   或无合格负候选时返回连接计算图的0，不强迫不可靠预测参与排序。
6. 总损失 `L = L_original + 0.2 * warmup * L_QCR`。
   `warmup = clip((epoch + step/steps_per_epoch)/5, 0, 1)`。
   前5轮从0线性渐入，第5轮起全量；恢复训练由 epoch/step 重建同一系数。

现实现只支持单类别；多类别会明确报错，不能不加修改用于其他类别数。
框没有来自此正则的直接梯度，但分类梯度仍能通过共享特征间接影响后续定位。
GO 保护也会排除部分重复框，因此不能声称 QCR 能解决所有类型的误检。
排序只约束相对大小，不能保证固定 conf=0.5 的 P/F1 上升，必须同时报告 R/AP。

## 开关与训练

- 开关：`settings.py` 的 `IMPROVEMENT_MODE = "qlcs_dsqc_qcr"`。
- 超参数：`dfine_hgnetv2_m_qlcs_dsqc_qcr.yml`。仅继承 QLCS+DSQC，旧第三模块关闭。
- `train.py` 当前 seed=3407，lr=2e-4，epochs=150，输出
  `output/qlcs_dsqc_qcr/2e-4_SD3407`。后续换种子必须同时换输出目录。
- 所有候选从相同官方预训练权重训练；不要从旧双模块 best 权重续训后冒充公平消融。
- 不改变数据划分、图像大小、增强、阈值、优化器、选权重规则。按既有规则统一
  使用验证 AP50 选出的 best 权重，并在**同一权重**上计算 AP50/P/R/F1/AP75/mAP。
- `valid.py`、`test.py` 本次未修改，保留了用户本地改动和旧权重路径。评估新训练
  权重时必须手动更新 CHECKPOINT_PATH 与输出目录。QCR不增加推理结构，故旧
  双模块权重可能也能加载，**能加载不代表选对实验权重**。

## 如何尽早发现无效训练

终端每批仅增加 `qcr_loss`、`qcr_pairs`；每轮末打印完整 QCR diagnostics。
`log.txt` 保存 `train_qcr_*` 均值，TensorBoard 保存 `QCR/*` 与 `Loss/loss_qcr`。

- `positive/negative/pairs`：每图可靠正例、已选负例、配对数量的批均值。
- `protected/ambiguous`：GO 保护排除、IoU保护排除数量；两者不重计。
- `no_positive/empty_gt`：无可靠正例图、有空GT图的比例。
- `raw/weighted/scale`：原始正则、乘权重后正则、当前渐入系数。
- `vfl_ratio`：加权 QCR / 最终层 VFL 的量级参考；不是梯度贡献百分比。
- `violations`：尚未满足0.5 logit间隔的正负配对数。

头5～10轮若始终无配对、或正则几乎为0，先检查选样/真实误检类型；不要直接
反复盲增系数。单图空对正常；早期损失下降不能替代完整验证精度。

## 验证参照与保留规则

以下引用用户确认的 SD18 **重训版本**，不是旧 SD18；不要按结果好坏挑选混搭。
原始运行、重训原因、数据/权重路径和随机性设置仍应留档。

| QLCS+DSQC验证 | AP50 | P@0.5 | R@0.5 | F1@0.5 | AP75 | mAP50:95 |
|---|---:|---:|---:|---:|---:|---:|
| SD18重训 | 0.8973 | 0.8114 | 0.8606 | 0.8353 | 0.5658 | 0.5794 |
| SD2026 | 0.9426 | 0.8514 | 0.9030 | 0.8765 | 0.6242 | 0.6053 |
| SD3407 | 0.9274 | 0.7914 | 0.8970 | 0.8409 | 0.5712 | 0.5665 |
| 三次均值 | 0.9224 | 0.8181 | 0.8869 | 0.8509 | 0.5871 | 0.5837 |

先做 SD3407；如有潜力，固定同一配置补齐18、2026。三次均值的 AP50 不下降、
F1 提升是核心，P 提升不能用明显降低 R 换取。期望 F1 增加至少0.005、P增加
至少0.010，且至少两种子同向改善；这些是工程目标，不等于统计显著性。
只有3个随机种子时应报告均值±样本标准差和逐种子差值，不声称稳定性已充分证明。

继续筛选只用训练/验证集，不再用历史测试分数反向迭代。此前如果测试结果影响
了模块选择，需在论文中如实披露，尽可能准备新的未查看最终留出集。

## 本地校验命令

```powershell
python -m unittest discover -s my_improve/tests -p "test_qcr*.py" -v
python -m my_improve.tests.smoke_qcr
```

前者测试筛选、梯度、空标注、AMP、渐入及原损失兼容性；后者只用随机输入跑
512尺寸推理等价检查和一次完整训练步骤，不读取数据/权重，不保存训练结果。
