# DSQC + PAD：候选框对齐去噪

2026-09-09。实现已完成；尚无真实精度收益结论，不保证改进成功。
RBA 三种子实验仍待完整验证结果，不据训练中波动判断其失败；本候选与 RBA 分开实验。

## 1. 方法与边界

PAD（Proposal-Aligned Denoising）是本任务的工作名称，不是已发表模块名。
它从模型自己的编码器候选中取少量定位有偏差的框，作为正去噪查询的参考起点，
学习恢复到原训练真值。它改的是辅助训练样本来源，不是标注、后处理或新损失。

保留 DSQC 算法、超参数和正常训练；QLCS/QCR/RBA/CSGA 等其他候选关闭但代码保留。
与 RBA 的区别：RBA 附加末层边界惩罚，PAD 保留原损失，仅改变部分 DN 初始参考框。
背景误检、重复检测不一定能被此方法纠正，AP50/F1 收益仍是待验证假设。

文献方向依据：

- [DEIM，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html)
  关注匹配与训练监督质量；本实验没有复现其 Dense O2O 或 MAL。
- [Mr. DETR，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Mr._DETR_Instructive_Multi-Route_Training_for_Detection_Transformers_CVPR_2025_paper.html)
  展示训练期辅助路线、推理时移除的研究方向；本实验没有新增其多路线解码器。

PAD 是针对现有 D-FINE 去噪机制的本地适配，不能把换名或工程组合本身当成创新性证明。
此前训练标注的只读模拟表明，原随机正 DN 起点本来已有较丰富的 IoU 分布，
不能宣称原噪声“错误”。PAD 的假设是模型自身的偏差来源可能比单纯改噪声强度更有价值。

## 2. 实现定义

1. 仍先生成原 DN，保持随机数调用顺序、正负槽位、类别嵌入和 attention mask。
2. 复用 `_get_decoder_input()` 生成的普通查询编码器候选：
   `sigmoid(enc_bbox_head(enc_topk_memory) + enc_topk_anchors)`，以及已有分类 logits。
   编码器候选只计算一次，不增加预测头、额外前向或最终输出框缓存。
3. 采样器局部 detach 候选框、分数及真值，在 FP32 中计算 IoU。
   原编码器输出仍保留梯度，继续接受原编码器监督。
4. 每个候选的最大 IoU 必须对应当前 DN 槽位原有真值，IoU 在 `[0.3, 0.7]`；
   多目标情况下最高和次高 IoU 差小于 `0.05` 则跳过。无效几何/分数不参加采样。
5. 每个真值每批最多替换 `floor(DN组数 * 当前比例)` 个正 DN 参考框；
   同一真值合格候选按该类别分数降序选择不同候选索引，随机选取它的正 DN 组槽位。
   不额外设置分类置信度门槛；候选不足保留原随机起点，不复制少数候选填满预算。
6. 替换比例前 5 轮线性从 0 增至 0.25，随后保持上限。预算取整会使实际启动稍晚于第一个 step。
   空目标、无合格候选、零预算全部回退；不把原随机起点替换成精确真值来凑数。

这里的唯一性指候选索引，两个不同索引仍可能预测近似相同的框；没有新增 NMS。
参考框使用原模型的归一化 `cxcywh`，不额外裁剪 `xyxy` 到图内；坐标逆 sigmoid 与原实现一致。
IoU 区间不代表匹配必然正确，也不能保证 FDR 回归范围一定覆盖真值。
第一层 pre_bbox_head 会先修正参考框；若效果差，还应分析其后的回归目标范围，而非只看初始 IoU。

原负 DN、padding、正常查询、分组、`dn_positive_idx`、归一化和损失定义均不改。
VFL、L1、GIoU、FGL、DDF 与原 DSQC 版本相同，没有 `pad_loss`。
训练后的参数和原损失数值会改变，这是改变训练样本的预期结果。
DSQC 目前也作用于 DN 查询，因此 PAD 与 DSQC 有训练交互，不能声称完全独立。

## 3. 开关及运行

| 模式 | DSQC | PAD | 用途 |
|---|---|---|---|
| `baseline` | 关 | 关 | 原始 D-FINE |
| `dsqc` | 开 | 关 | 当前直接对照 |
| `pad` | 关 | 开 | PAD 独立消融 |
| `dsqc_pad` | 开 | 开 | 本次默认候选 |

当前 `my_improve/settings.py` 已设置 `IMPROVEMENT_MODE = "dsqc_pad"`。
新 YAML 只提供 M 规模；原 RBA 模式及其 YAML 未删除或改写。
`train.py` 默认：seed=3407，input=512，batch=6，lr=2e-4，epochs=150，增强停止=135，AMP 开启。
从相同官方 `weight/dfine_m_obj2coco.pth` 开始，不从已训练 DSQC/RBA best 权重继续微调。
输出为 `E:\YOLO\D-FINE\output\dsqc_pad\2e-4_SD3407`，与旧实验隔离。

```powershell
python train.py
```

启动应显示 DSQC/PAD 开启，RBA/QCR/QLCS/CSGA 等关闭。换种子须同时改 SEED 和独立输出目录。
PAD 不自动启动训练；不要与已有 RBA 进程争抢同一张显卡。
若需要新启动或续跑 RBA，先切回 `dsqc_rba`，并恢复对应输出及 resume/tuning 设置。
续训只能恢复同一实验的 last.pth；PAD 进度由 engine 按 epoch/step 重建，没有隐藏累计计数。
完整随机数复现仍依赖原训练框架的随机状态管理，并非仅凭相同 seed 就承诺跨环境逐位一致。

模型推理结构不变，PAD 无训练参数或 buffer；19,515,558 参数与 DSQC 相同。
验证/测试不需要调用 PAD 进度设置。若自行调用 `model.train()` 前向，须先执行
`model.decoder.pad.set_progress(epoch, step, epoch_step)`；标准 train.py 已自动处理。
用户当前 valid.py、test.py 和分析脚本的权重路径未修改，评估时请指向本次新权重，
并按统一规则选择同一个 checkpoint 的整套指标。

## 4. 看哪些日志

终端每批显示 `pad_replaced`（实际替换数）、`pad_actual_ratio`（占正 DN 比例）、
`pad_coverage`（有合格候选的真值比例）。其余诊断在 epoch 末、log.txt、TensorBoard 的 `PAD/*`。

| 字段（log.txt 前缀 `train_pad_`） | 含义 |
|---|---|
| `ratio` | 渐进替换上限，最终 0.25，不等于实际替换率 |
| `positive_slots` / `requested` / `replaced` | 正 DN 数 / 整数预算 / 实際使用数 |
| `candidates` / `covered_gt` / `total_gt` | 合格候选数 / 有候选真值数 / 真值数 |
| `ambiguous` / `invalid_proposals` | 被跳过的歧义候选 / 无效候选 |
| `original_iou` / `replacement_iou` | 被替换槽位原起点 / 新起点相对目标的 IoU |
| `original_area_ratio` / `replacement_area_ratio` | 相对真值面积的原/新起点比值 |

计数显示批次/进程平均值，可能不是整数；epoch 替换率和覆盖率按汇总计数计算。
IoU/面积比用额外 `*_sum` 按实际替换数加权汇总。无替换时相关诊断为 0，不表示模型预测为 0。
若暖启动完成后一整轮都没有替换，程序打印警告（不自动中止），应先查覆盖率再追加完整种子。
低替换比例不一定是故障：唯一候选不足会回退；训练后期候选越来越准确、超出难度区间也会减少。

## 5. 最小验证和实验止损

已执行：72 项单元/集成测试；另有 CPU 完整 512 输入模型、原完整损失和优化器合成步测试。
包括禁用/零比例/空目标/无候选的输出及 RNG 一致性、正常 encoder 梯度保留、负槽位与掩码不变。
仅采样器的小张量混合精度测试使用 CUDA；完整模型测试默认 CPU，未启动真实训练。

```powershell
python -m unittest discover -s my_improve/tests -p 'test_*.py' -q
python -m my_improve.tests.smoke_pad
```

合成目标特意围绕编码器候选构造，保证可以测试替换逻辑；不代表真实训练集覆盖率或准确率。
所有真实训练/验证数据及现有权重未被修改，未创建正式训练 checkpoint。

下一步先跑 DSQC+PAD 的 SD3407：

1. 前 5～10 轮检查真实训练候选覆盖、实际替换率和数值稳定性；若长期零替换，先停止检查，
   不投入三次完整实验，也不把仅打开开关当成有效消融。
2. 保持原 150 轮与选权重规则。完整同种子参照 DSQC 的 AP50≈0.9416、F1≈0.8636。
   若两项均下降超过 0.005，本候选暂不追加两个种子；这是工程筛选规则，不是显著性检验。
3. 有继续验证价值后固定配置补 18/2026，报告全部配对结果、均值和样本标准差。
   保留要求 AP50/F1 均值均提高，至少两个种子同向改善；P 提升不能掩盖明显召回损失。
4. 全程使用同一 checkpoint 的整套指标，confidence=0.5、match IoU=0.5，不能混用 best_map50 和 best_f1。
   旧截图与最新分析的小差异先按 checkpoint/标注哈希及评估设置核对，不挑较高的一套。
5. PAD 和 RBA 分开比较；两者各自有证据后才考虑组合。若最终保留 PAD，补 pad 单独三种子完成四组消融。

后续选择只使用训练/验证数据，不反复用测试集决定候选。以前测试结果影响过开发，论文应如实披露。
