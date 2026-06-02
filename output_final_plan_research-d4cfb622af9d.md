# 基于WavLM+XLSR双前端融合与Tri-Attention时频双向Mamba的ASVspoof 2021 LA/DF EER联合优化方案

## 1. 背景与目标
WaveSP-Net在Deepfake-Eval-2024上EER=10.58%，SpoofCeleb上EER=0.13%，未在ASVspoof 2021 LA/DF上直接评测。已有实验在2019LA训练、2021DF测试EER=9%。ASVspoof 2021 DF SOTA已达0.5%-1%区间（Guo 2024: WavLM+MFA DF EER=0.42%），LA SOTA约0.82%-1.20%（XLSR-Mamba LA EER=0.93%）。需针对DF编解码伪影和LA域偏移进行改进。首轮方案提出双前端融合、CAM、域自适应提示微调等方向。本轮追问要求研究Yang等Tri-Attention Fusion文献（需从PDF核实EER数值）并整合其创新点。

## 2. 核心改进方向
### 2.1 P0：WavLM Large + XLSR-300M双前端并行冻结 + 可学习门控融合 + Partial-WSPT提示调优
- **操作**：将WaveSP-Net的单一XLSR-300M前端替换为WavLM Large与XLSR-300M双流并行前端（均冻结），分别提取互补特征后通过可学习门控融合模块（Gated Fusion）自适应加权融合，再输入Partial-WSPT提示调优框架（10个提示令牌/层，其中4个小波稀疏令牌）。
- **证据**：
  - Guo et al., ICASSP 2024: WavLM首次用于DF取得SOTA（EER=0.42%）。
  - Xiao & Das, IEEE SPL 2025: XLSR-Mamba LA EER=0.93%。
  - Zhang et al., XWSB, IEEE SLT 2024: XLS-R+WavLM混合系统验证特征互补有效性。
- **预期收益**：LA EER接近或低于0.93%（XLSR分支主导），DF EER降至0.5%以下（WavLM分支主导），门控融合自适应平衡双流贡献。

### 2.2 P0：将WaveSP-Net的单一双向Mamba后端替换为BiMamba-ST双分支架构 + Tri-Attention Fusion
- **操作**：将WaveSP-Net的单一双向Mamba后端替换为BiMamba-ST双分支架构——一个分支处理频谱特征（沿频域轴的BiMamba），另一分支处理时序特征（沿时域轴的BiMamba），分别捕获编解码伪影的频谱耦合与时间连续性。在双分支输出后插入Tri-Attention Fusion模块，通过三重注意力（时域自注意力+频域自注意力+交叉注意力）渐进融合双分支特征。
- **证据**：
  - Yang et al., ICASSP 2026: Tri-Attention Fusion在ASVspoof 2021上EER需从PDF核实。
  - BiCrossMamba-ST (Interspeech 2025)在DF21上相对AASIST降低26.3% EER。
  - Fake-Mamba (IEEE ASRU 2025) DF EER=1.74%。
- **预期收益**：DF EER显著降低，时频联合注意力精准定位编解码伪影的时频耦合模式；LA EER降低，BiMamba-ST频谱分支增强对TTS/VC合成伪影的频域建模能力。

### 2.3 P0：可学习跨频带注意力模块（CAM）替代随机WDS
- **操作**：在小波子带间引入可学习跨频带注意力模块（CAM），利用语音编解码伪影在低/高频子带间的耦合特性，在LWD输出的低/高频系数间执行交叉注意力（低频查询高频+高频查询低频），替代原有随机稀疏化策略。
- **证据**：
  - WaveSP-Net消融实验：去除WDS导致EER相对上升35.54%（Xuan et al., ICASSP 2026）。
  - FreqFAN频带注意力思想可迁移至语音小波子带系数。
  - Guo et al., ICASSP 2024: WavLM+MFA DF EER=0.42%。
- **预期收益**：主要利好DF场景，通过可学习的跨频带交互精准定位编解码伪影，预期DF EER显著降低。

### 2.4 P1：域自适应提示微调（对比学习 + 仅更新提示参数）
- **操作**：在ASVspoof 2019上训练后，仅更新提示参数进行2021域自适应微调，引入频域对比损失（借鉴SONAR的Jensen-Shannon散度），约束低/高频子带表征的判别性分离，缓解LA→DF域偏移。
- **证据**：
  - SONAR (ICML 2026): 高频残差对比学习在DF上EER=1.45%，LA上EER=1.20%。
  - Oiso et al. (Interspeech 2024) 验证了测试时提示微调在音频deepfake检测域自适应中的有效性。
- **预期收益**：直接针对LA/DF的域偏移问题，预期可同时降低两者EER，特别是跨域（LA→DF）场景下的EER。

### 2.5 P1：多尺度小波包分解（WPT）
- **操作**：将WaveSP-Net中的离散小波变换（DWT）替换为多尺度小波包分解（WPT），对高低频均做递归分解，精细捕获窄带编解码伪影。
- **证据**：
  - WaveSP-Net可学习滤波器优于固定滤波器（固定滤波器EER 16.55% vs 可学习10.58%）。
  - 图像域Haar小波多分辨率分析已被用于检测deepfake频域伪造痕迹。
- **预期收益**：主要利好DF场景，通过更精细的频带划分增强对窄带编解码伪影的捕获能力。

## 3. 实验设计
- **数据集**：使用ASVspoof 2019 LA训练集训练，ASVspoof 2021 LA eval和DF eval分别评测。注意：2019 LA不含DF数据，2021 DF编码器类型完全不同，属严格跨域评估。
- **模型配置**：冻结WavLM Large和XLSR-300M，仅训练门控融合模块、提示令牌（10个/层，4个小波稀疏令牌）、CAM模块、BiMamba-ST双分支和Tri-Attention Fusion模块。可训练参数约4.146M（1.298%）。
- **评估指标**：EER、min t-DCF。
- **消融实验**：验证各模块贡献（双前端融合、BiMamba-ST+Tri-Attention、CAM、对比损失、WPT）。

## 4. 风险与缓解
- **双前端融合计算开销**：增加计算和内存占用。缓解：采用混合精度训练，门控融合轻量化设计。
- **BiMamba-ST+Tri-Attention Fusion模型复杂度**：增加模型复杂度。缓解：采用渐进式注意力权重调度。
- **门控融合训练不稳定性**：可能引入训练不稳定性。缓解：采用渐进式门控权重调度。
- **CAM模块训练不稳定性**：可能增加训练不稳定性。缓解：采用渐进式注意力权重调度。
- **WPT过拟合风险**：增加频带数量可能导致过拟合。缓解：使用WDS稀疏正则化（ρ=0.1）。
- **域自适应微调协议兼容性**：需确保2019→2021协议兼容。缓解：严格遵循ASVspoof官方协议。
- **依赖WavLM和XLSR预训练模型**：确保预训练模型可用。

## 5. 引用说明
- Guo et al., ICASSP 2024: WavLM首次用于DF取得SOTA（EER=0.42%）。
- Xiao & Das, IEEE SPL 2025: XLSR-Mamba LA EER=0.93%，DF EER=1.88%。
- Zhang et al., XWSB, IEEE SLT 2024: XLS-R+WavLM混合系统验证特征互补有效性。
- Xuan et al., ICASSP 2026: WaveSP-Net消融实验（去除WDS导致EER相对上升35.54%）。
- Yang et al., ICASSP 2026: Tri-Attention Fusion在ASVspoof 2021上EER需从PDF核实。
- BiCrossMamba-ST (Interspeech 2025): 在DF21上相对AASIST降低26.3% EER。
- Fake-Mamba (IEEE ASRU 2025): DF EER=1.74%。
- SONAR (ICML 2026): 高频残差对比学习在DF上EER=1.45%，LA上EER=1.20%。
- FreqFAN频带注意力思想可迁移至语音小波子带系数。
- 图像域Haar小波多分辨率分析已被用于检测deepfake频域伪造痕迹。
- 频域去偏置（FreqDebias）策略可迁移为小波域一致性正则化损失。