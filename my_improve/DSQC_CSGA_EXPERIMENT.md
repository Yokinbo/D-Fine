# 下一步：DSQC + CSGA（验证集消融）

当前优先参照为仅DSQC，不再预设QLCS+DSQC是最佳模型。四组结果补齐后，仅DSQC
在验证集核心AP50/F1的三次均值更高；这仍不是统计显著性或最终测试泛化的证明。

## 当前配置

- `settings.py`: `IMPROVEMENT_MODE = "dsqc_csga"`。
- YAML: `dfine_hgnetv2_m_dsqc_csga.yml`，直接继承仅DSQC配置。
- DSQC开启，CSGA开启；QLCS/QFBCG/QACG/MGCA/SHEA/QCR关闭。
- DSQC、CSGA实现及超参数均不改；原始损失和匹配方式不改。
- 保持512输入、batch6、lr=2e-4、150轮、135轮停止增强、AMP、种子3407。
- 当前输出：`output/dsqc_csga/2e-4_SD3407`。补18、2026时同步更换输出目录。
- 从相同官方`dfine_m_obj2coco.pth`开始训练，RESUME为空，不从DSQC best权重续训。
- `valid.py`/`test.py`用户原权重路径未修改；新实验完成后必须选对应新权重和输出目录。
  固定验证best_map50选权重规则，在同一权重上计算全部指标，不混用best_f1结果。

## 配对参照（仅DSQC，来自用户三次验证截图）

| 种子 | AP50 | P | R | F1 | AP75 | mAP50:95 |
|---|---:|---:|---:|---:|---:|---:|
| 18 | 0.9287 | 0.8305 | 0.8909 | 0.8596 | 0.5521 | 0.5908 |
| 2026 | 0.9169 | 0.8177 | 0.8970 | 0.8555 | 0.5662 | 0.5947 |
| 3407 | 0.9416 | 0.8128 | 0.9212 | 0.8636 | 0.6298 | 0.6046 |
| 均值 | 0.9291 | 0.8203 | 0.9030 | 0.8596 | 0.5827 | 0.5967 |

先跑3407检查CSGA激活和训练稳定性；完整训练后与同种子比较，有潜力则固定配置
补18、2026，不挑换种子或混合不同超参数结果。核心看三次平均AP50/F1，并同时
检查P、R。增益不得仅来自牺牲召回；报告逐种子差值、均值及样本标准差。
不能只超过较弱的旧双模块组合就宣布成功，必须以仅DSQC为直接参照。

本轮只回答“CSGA在DSQC上是否有增益”。若有效且计划作为论文第二模块，再补
仅CSGA，构成 baseline / DSQC / CSGA / DSQC+CSGA 的完整四组消融。
原QLCS四组和三模块CSGA配置均保留复现，不删除或覆盖原权重。
设计与筛选使用训练/验证集，不用测试集反复选择候选。

## 校验

```powershell
python -m unittest discover -s my_improve/tests -p "test_*.py" -v
python -m my_improve.tests.smoke_csga
# 保留旧三模块路径的检查：
python -m my_improve.tests.smoke_csga --reference qlcs_dsqc
```

CSGA技术依据、边界与许可见[原CSGA设计记录](CSGA_EXPERIMENT.md)；其中三模块
实验计划属于上一阶段历史记录，当前模式和参照以本文为准。
