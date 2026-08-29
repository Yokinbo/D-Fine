# D-FINE 模型改进实验

本目录集中保存论文模型改进的实现、开关和消融 YAML。当前只完成第一项：

- `qlcs.py`：查询引导的隐式部件采样（QLCS）；
- `dfine_hgnetv2_m_qlcs.yml`：D-FINE-M + QLCS 单模块配置；
- `settings.py`：训练、验证和测试共同使用的改进模式开关。

## 第一项消融怎么运行

1. 在 `settings.py` 中设置 `IMPROVEMENT_MODE = "qlcs"`。
2. 保持 `experiment_config.py` 中 `MODEL_SIZE = "m"`、关闭 VRAC。
3. 在 `train.py` 中为本次 QLCS 实验填写新的 `OUTPUT_DIR`，不要覆盖基线结果。
4. 运行 `python train.py`。训练完成后，验证和测试仍保持同一个 `qlcs` 模式，
   再分别填写改进权重与结果目录。

若要复现原始基线，只需把 `IMPROVEMENT_MODE` 改回 `"baseline"`。当前阶段不要
同时开启 QLCS 和 VRAC，以免单模块增益无法归因。

后续将按消融顺序继续加入整体—部件—背景关系门控与结构一致性监督；在前一项
经验证有效前，不提前叠加下一项。
