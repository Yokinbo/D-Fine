# DSQC + QCR 验证集实验

当前模式 `dsqc_qcr`：DSQC开启，QCR作为训练期附加排序正则开启；QLCS、CSGA及其他候选模块关闭。
不改变DSQC实现，也不改变旧QCR实现和超参数。原始损失与匹配算法保留。
设计依据与机制见 [QCR历史记录](QCR_EXPERIMENT.md)，其中旧三模块结果不是本组合的结果。

## 训练与比较

- 从相同官方预训练权重开始，不从DSQC最佳权重续训。
- 学习率2e-4、150轮、增强停止135轮、batch6、输入512保持不变。
- 当前种子3407，输出 `output/dsqc_qcr/2e-4_SD3407`。
- 补种子18、2026时同步修改种子和输出目录；固定配置，记录全部结果。
- 对照仅DSQC，使用相同验证集、权重选择规则及conf=0.5评价口径。

| 仅DSQC参照 | AP50 | F1 |
|---|---:|---:|
| SD18 | 0.9287 | 0.8596 |
| SD2026 | 0.9169 | 0.8555 |
| SD3407 | 0.9416 | 0.8636 |
| 三次均值 | 0.9291 | 0.8596 |

主要判断AP50与F1的联合收益，P提升不能以明显损失R为代价。报告逐种子差值和均值/标准差，
不以单次最好结果替代三次结果。不根据测试集反复调整；本方案尚未证明有效。
QCR只支持当前单类别实验。其推理结构与仅DSQC一致，但训练后权重和精度可以不同。

`valid.py`、`test.py`的本地权重/输出路径未自动修改，新训练完成后手动指向新结果。

## 自检

```powershell
python -m unittest discover -s my_improve/tests -p 'test_*.py' -q
python -m my_improve.tests.smoke_qcr
# 历史三模块自检仍可运行：
python -m my_improve.tests.smoke_qcr --reference qlcs_dsqc
```
