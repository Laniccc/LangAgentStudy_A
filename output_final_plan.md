# 基于WaveSP-Net小波域稀疏提示与频域对齐的ASVspoof 2021 LA/DF EER联合优化方案

## 1. 背景与目标
WaveSP-Net在Deepfake-Eval-2024上EER=10.58%，SpoofCeleb上EER=0.13%，未在ASVspoof 2021 LA/DF上直接评测。已有实验在2019LA训练、2021DF测试EER=9%。目标：针对2021DF的未知编码器频谱泄漏和LA的编解码伪影进行改进，降低EER。

## 2. 核心改进方向
### 2.1 P0：对抗性频带扰动模块（LWD阶段）
- **操作**：在WaveSP-Net的可学习小波分解（LWD）阶段，对高-低频系数施加可学习的频带掩码扰动，模拟未知编码器频谱泄漏。
- **证据**：WaveSP-Net消融实验（Xuan等，ICASSP 2026）证实去除WDS导致EER相对上升35.54%（10.58%→14.34%），固定滤波器导致相对上升56.44%。
- **预期收益**：提升对2021DF未知编码器的泛化性，同时防止LA过拟合特定频带。

### 2.2 P0：Partial-WSPT-XLSR前端 + 双向Mamba后端 + 时频能量对齐
- **操作**：采用WaveSP-Net的Partial-WSPT-XLSR前端（冻结XLSR-300M，仅训练提示令牌10个/层，其中4个小波稀疏令牌）与双向Mamba后端（参考XLSR-Mamba的DuaBiMamba结构）。增加时域-小波域双流特征对齐模块，通过对比损失强制真实与伪造语音的时频能量分布对齐。
- **证据**：
  - XLSR-Mamba（Xiao & Das, IEEE SPL 2025）在ASVspoof 2021 LA上EER=0.93%，DF上EER=1.88%（官方GitHub README）。
  - Fake-Mamba（Xuan等，2025）在DF上EER=1.74%，LA上EER=0.97%。
  - WaveSP-Net在SpoofCeleb（含LA-like场景）上EER=0.13%。
- **预期收益**：LA EER接近或低于0.93%；DF EER目标降至1.5%以下（基于XLSR-Mamba的1.88%和Fake-Mamba的1.74%）。

### 2.3 P1：可学习高通/低通滤波器组（冻结XLSR旁路）
- **操作**：在冻结XLSR旁路中插入轻量级可学习高通/低通滤波器组（借鉴WaveSP-Net的LWD模块，1D可学习小波滤波器F0低通+F1高通），滤波器系数与提示令牌联合优化。输出经小波域稀疏化（WDS，sparsity ratio=0.1）后与原始XLSR特征拼接。
- **证据**：
  - SONAR（Nitzan等，ICML 2026）Table 1：SONAR-Finetune在DF上EER=1.45%，LA上EER=1.20%；SONAR-Full在DF上EER=1.57%，LA上EER=1.55%。注意：3.69%为XLSR+AASIST基线，非SONAR自身结果。
  - WaveSP-Net可学习滤波器优于固定滤波器（固定滤波器EER 16.55% vs 可学习10.58%）。
- **预期收益**：DF场景增强高通支路抑制低频伪造痕迹；LA场景平衡双支路捕获编解码伪影。

### 2.4 P1：频域一致性正则化损失
- **操作**：增加频域一致性正则化损失，约束原始音频与频带扰动版本的小波域提示嵌入保持一致，防止模型过度依赖特定频带（谱偏置）。
- **证据**：FreqDebias频域去偏置策略（图像域迁移，待补充完整引用：需提供作者、标题、会议）。
- **预期收益**：提升对未知编码器的泛化性。

## 3. 实验设计
- **数据集**：ASVspoof 2019 LA训练集训练；ASVspoof 2021 LA eval和DF eval分别评测。注意：2019 LA不含DF数据，2021 DF编码器类型完全不同，属严格跨域评估。
- **模型配置**：冻结XLSR-300M，仅训练提示令牌（10个/层，4个小波稀疏令牌）和Mamba分类器。可训练参数约4.146M（1.298%）。
- **评估指标**：EER、min t-DCF。
- **消融实验**：验证各模块贡献（对抗性扰动、双流对齐、滤波器组、正则化损失）。

## 4. 风险与缓解
- **对抗性频带扰动**：可能引入训练不稳定。缓解：采用渐进式扰动幅度调度。
- **双流对齐模块**：增加计算开销。缓解：仅在前向传播中激活，参数高效。
- **可学习滤波器组**：需调参避免过拟合。缓解：使用WDS稀疏正则化（ρ=0.1）。
- **依赖XLSR-300M**：确保预训练模型可用。

## 5. 引用说明
- XLSR-Mamba DF EER=1.88%（来源：官方GitHub README github.com/swagshaw/XLSR-Mamba；Hugging Face模型卡）。
- SONAR DF EER=1.45%（SONAR-Finetune）或1.57%（SONAR-Full）（来源：SONAR arXiv:2511.21325 Table 1）。
- ASVspoof 2021 DF挑战赛最佳系统EER=15.64%（Team T23，来源：arXiv:2109.00537）。
- FreqDebias、WaveDIF、FreqNet等图像域方法需补充完整引用（作者、标题、年份、会议）后方可正式引用。