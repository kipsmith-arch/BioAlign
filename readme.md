# BioAlign
- **项目名称**：基于 QLoRA 与 DPO 的完整大模型后训练（Post-training）流水线——生物医学领域适配与对齐
- **项目描述**：
    - 针对生物医学领域复杂的指令理解与生成需求，设计并实现了**完整的大模型后训练（Post-training）流水线**，覆盖「领域继续预训练 → 参数高效微调（PEFT）→ 直接偏好优化（DPO）对齐」三个环节，全程在单卡 T4（16GB）上以资源高效方式完成。
    - **阶段一（领域继续预训练）**：基于多组学未标注生物序列（人类基因组 GRCh、RNAcentral 非编码 RNA、Swiss-Prot 蛋白序列）进行 next-token 预训练，使模型获得生物序列的底层表征能力。通过「有无该阶段」的消融实验，验证了论文（Biology-Instructions, arXiv:2412.19191）"序列预训练是微调必不可少的前置"这一结论。
    - **阶段二（PEFT）**：利用 `Biology-Instructions` 多组学指令数据集（DNA/RNA/蛋白/多分子共 21 类任务），基于 **transformers + peft + bitsandbytes** 对 `Qwen2.5-3B` 模型进行 **4-bit QLoRA** 微调（`lora_r=16`、NF4 量化），以极低显存成本（<12GB）完成领域知识注入与指令遵循能力唤醒，模型在分类/回归等下游任务上从接近随机提升到可用水平。
    - **阶段三（RL）**：自主构造**生物领域偏好数据**（以 held-out 样本的标准答案为 chosen、基座模型生成回答为 rejected），使用 `trl` 库的 `DPOTrainer` 进行直接偏好优化，通过优化 chosen/rejected 响应对的隐含奖励提升回答质量，避免了传统 RLHF 复杂的奖励建模与 PPO 训练流程。
- **成果**：构建了一套完整的、资源高效的 LLM 领域适配与对齐方案：三阶段消融证明每个环节均带来有效增益；DPO 对齐后模型回答质量显著提升、且**不损失领域任务能力**（SFT 后与 DPO 后在相同任务集上的指标对比）；验证了在有限算力（单卡 T4）下复现工业级后训练技术栈（continue pretraining → QLoRA SFT → DPO）的可行性。
