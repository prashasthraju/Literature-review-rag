<!-- page-start:1 -->

## Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation

Taekyung Ki1,⋆ Sangwon Jang1,⋆ Jaehyeong Jo1 Jaehong Yoon2 Sung Ju Hwang1,3 1KAIST 2NTU Singapore 3DeepAuto.ai

{taekyung.ki, sangwon.jang, sungju.hwang}kaist.ac.kr

[https://taekyungki.github.io/AvatarForcing](https://taekyungki.github.io/AvatarForcing)

User

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000000_2acb5b82bcb8ab299d341e4157378fdb1c052ebc77b6db2acba696d679c961d0.png)

Interactive Head Avatar

(Generated) Context

Figure 1. Overview of Avatar Forcing. It can generate a real-time interactive avatar video conditioned on user motion and audio, as well as avatar audio. The avatar naturally mirrors the user's expression, such as smiling when the user smiles, for more engaging interactions.

## Abstract

Talking head generation creates lifelike avatars from static portraits for virtual communication and content creation. However, current models do not yet convey the feeling of truly interactive communication, often generating one-way responses that lack emotional engagement. We identify two key challenges toward truly interactive avatars: generating motion in real-time under causal constraints and learning expressive, vibrant reactions without additional labeled data. To address these challenges, we propose Avatar Forcing, a new framework for interactive head avatar generation that models real-time user-avatar interactions through diffusion forcing. This design allows the avatar to process real-time multimodal inputs, including the user's audio and motion, with low latency for instant reactions to both verbal and non-verbal cues such as speech, nods, and laughter. Furthermore, we introduce a direct preference optimization method that leverages synthetic losing samples constructed by dropping user conditions, enabling label-free learning of expressive interaction. Experimental results demonstrate that our framework enables real-time interaction with low latency (≈ 500ms), achieving 6.8× speedup compared to the baseline, and produces reactive and expressive avatar motion, which is preferred over 80% against the baseline.

⋆ Equal contribution.

## 1. Introduction

Talking head generation animates static portrait images into lifelike avatars that can speak like humans. These systems are increasingly used to create virtual presenters, hosts, and educators that can substitute for real human presence in many scenarios. They also support customized avatars that users can interact with, for instance, chatting with their favorite characters [21, 43, 53], offering a practical tool for content creation and visual communication.

However, existing avatar generation models fail to fully replicate the feeling of real face-to-face interaction. They primarily focus on generating audio-synchronized lip or natural head motions [28, 33, 39, 54, 61-64] to deliver information accurately and naturally, rather than engaging in interactive conversations. Such one-way interaction overlooks the bidirectional nature of real-life conversations, where the continuous exchange of verbal and non-verbal signals plays a crucial role in communication. For instance, active listening behaviors such as nodding or empathic responses encourage the speaker to continue, while expressive speaking behaviors such as smiling or making eye contact contribute to a more realistic and immersive conversation.

<!-- page-break:2 -->

We identify two major challenges in generating truly interactive avatars. The first challenge is the real-time processing of users' multimodal inputs. An interactive avatar system must continuously receive and respond to the user's multimodal cues, such as speech, head motion, and facial expression, which requires both low inference time and minimal latency. Existing methods [70] achieve fast inference within a motion latent space [14] but have high latency because they need the full conversation context (e.g., over 3 seconds), including future frames. The model must wait for a sufficiently long audio segment before generating motion, causing a notable delay in user interaction, as illustrated in Fig. 4. This highlights the need for a causal motion generation framework that reacts immediately to live inputs.

The second challenge is learning expressive and vibrant interactive motion. Expressiveness in human interactions is inherently difficult to define or annotate, and the lack of well-curated data makes the natural interactive behaviors hard to model. For listening behaviors in particular, we observe that most of the training data are less expressive and low-variant, often exhibiting stiff posture (See Fig. 5). Moreover, unlike lip synchronization, which is strongly tied with the avatar audio and therefore relatively easy to learn, reacting appropriately to user cues can correspond to a wide range of plausible motions. This ambiguity greatly increases learning difficulty, often resulting in less diverse, stiff motions, particularly in response to non-verbal cues.

To address these challenges, we present a new interactive head avatar generation framework, Avatar Forcing, which models the causal interaction between the user and the avatar in a learned motion latent space [28, 59]. Inspired by recent diffusion-forcing-based interactive video generative models [2, 25, 65], we employ causal diffusion forcing [6, 30, 65] to generate interactive head motion latents while continuously processing multimodal user inputs in real-time. Unlike the previous approach [70] that requires the future conversation context (See Fig. 4), Avatar Forcing causally generates interactive avatar videos by efficiently reusing past information through key-value (KV) caching. This design enables real-time reactive avatar generation with minimal latency.

Moreover, following the learning-from-losing paradigm of preference optimization methods [41, 46, 58], we propose a preference optimization method to enhance the interactiveness of the avatar motion. By synthesizing underexpressive motion latents via dropping the user signals and using them as less-preferred samples, we achieve significantly improved expressiveness and reactiveness, without the need for additional human annotations for natural interaction. As a result, Avatar Forcing can produce more natural and engaging interactive videos as illustrated in Fig. 1.

Extensive experiments demonstrate that Avatar Forcing supports real-time interaction with a latency of roughly 500ms. Moreover, the proposed preference optimization significantly improves motion reactiveness and richness, producing more natural and engaging videos. In human evaluations, our method is preferred over 80% against the strongest baseline, demonstrating clear advantages in naturalness, responsiveness, and overall interaction quality.

## 2. Related Works

## 2.1. Talking Avatar Generation

Talking avatar generation aims to generate a lifelike talking avatar video from a given reference image and an audio. Earlier methods in this field had focused on synthesizing accurate lip movement from the driving audio [3, 18, 27, 44, 69]. These approaches are extended to generate holistic head movements, including a rhythmical head motion [28, 54, 62, 69] and vivid facial expressions, such as eye blink and jaw movement [14, 15, 19, 62, 66]. SadTalker [66], for instance, leverages 3Dmorphable models (3DMM) [13] as an intermediate representation for the non-verbal facial expression. EMO [54] and its subsequent models [8, 10, 26] leverage the foundation image diffusion model (e.g., StableDiffusion [47]) for photorealistic portrait animation. Recent works [28, 36, 62] introduce generative modeling techniques, such as diffusion models or flow matching into a learned motion space [14, 59], achieving real-time head motion generation.

## 2.2. Listening Avatar Generation

Another line of works [24, 35, 38, 40, 49, 68] focus on generating realistic listening head motions, such as nodding or focusing. Generating such responsive motions is challenging because the relationship between the speaker and listener is inherently one-to-many [40, 60], and the cues are context-dependent and weakly supervised. Hence, most of the works generate the personalized listening motion [40, 68] or leverage explicit control signals, such as text instruction [38] and pose prior [60].

<!-- page-break:3 -->

## 2.3. Dyadic Conversational Avatar Generation

Recently, several studies investigated dyadic motion generation [20, 55, 70], which involves modeling the interactive behavior of two participants engaged in a conversational setting. DIM [55] quantizes dyadic motions into two discrete latent spaces [57], one for the verbal and one for the non-verbal. However, it requires a manual roleswitching signal between the two spaces, resulting in discontinuous transitions between speaker and listener states. INFP [70] addresses this limitation by introducing audioconditioned verbal and non-verbal memory banks within a unified motion generator. However, this is ill-suited for real-time interactive generation as its bidirectional transformer [42] requires access to the entire context of the conversation. ARIG [20] generates implicit 3D keypoints [19] as its motion representation, primarily focusing on facial expressions. Yet it struggles with temporal consistency and fails to produce holistic head motions.

In this work, we focus on modeling the interactive behavior of both participants continuously influencing each other through verbal and non-verbal signals. We generate holistic interactive motions, for instance, talking, head movement, listening, and focusing, using a diffusion forcing framework that promptly responds to the multimodal user signals.

## 3. Background

Diffusion Forcing Diffusion forcing [6] stands out as an efficient sequential generative model that predicts next token conditioned on the noisy past tokens. Let x1 = (x11 , x21 , · · · , xN1 ) ∈ RN×3×H×W denote a sequence of N tokens sampled from data distribution p1. Each token is corrupted with per-token independent noise levels t := (t1, t2, · · · , tN) ∈ [0, 1]N , forming an independently noised sequence xt := (x1t1 , x2t2 , · · · , xNtN ), where xntn = tnxn1 + (1 - tn)xn0 and x0 = (xn0 )Nn=1 ∼ p0. Here, we follow the noise scheduler of flow matching [34]. The training objective of diffusion forcing is to regress the vector field vθ toward the target vector field vn tn = xn1 - xn0 :

<!-- formula-not-decoded -->

Diffusion forcing reformulates conventional teacher forcing in terms of diffusion models, allowing causal sequence generation with diffusion guidance [22] for controllable generation and flexible sampling procedure.

Direct Preference Optimization Direct Preference Optimization (DPO) [46] aligns a model with human preferences without explicitly training a reward model. The training objective LDPO is formulated as follows:

<!-- formula-not-decoded -->

where c is the condition, xw and xl denote preferred and less preferred samples, respectively, and πref is a frozen reference model during the optimization, typically initialized by the pre-trained weight of πθ. Here, σ(·) is the sigmoid function, and β is the deviation parameter. DiffusionDPO [58] extends DPO to diffusion models by reformulating the objective with diffusion likelihoods, enabling preference optimization using the evidence lower bound.

## 4. Avatar Forcing

In this work, we present Avatar Forcing, which generates a video of a real-time interactive head avatar, conditioned on the avatar audio and the multimodal signals of the user. We provide an overview of our framework in Fig. 2.

In Sec. 4.1, we present our framework for achieving realtime interactive head avatar generation based on causal diffusion forcing in the motion latent space. This consists of two key steps: encoding multimodal user and avatar signals, and causal inference of avatar motion. In Sec. 4.2, we introduce a preference optimization method for the interactive motion generation, which enhances expressive interaction without the need for additional human labels.

## 4.1. Diffusion Forcing for Real-Time Interaction

Motion Latent Encoding For motion encoding, we employ the motion latent auto-encoder fromKi et al. [28]. The latent auto-encoder maps an input image S ∈ R3×H×W to a latent z ∈ Rd whose identity and motion are decomposable:

<!-- formula-not-decoded -->

where zS ∈ Rd and mS ∈ Rd are the identity and motion latents, respectively. The identity latent zS encodes identity representation of the avatar image S (e.g., appearance), while the motion latent mS encodes rich verbal and nonverbal features (e.g., facial expression, head motion). We use this latent representation to capture holistic head motion and fine-grained facial expression, which are crucial for realistic head avatar generation. We provide more details on the auto-encoder in Appendix B.

Interactive Motion Generation In Avatar Forcing, the motion latents are generated by conditioning on multimodal user signals and avatar audio. This can be formulated as an autoregressive model as follows:

<!-- formula-not-decoded -->

where each motion latent mi is predicted from past motion latents and the condition triplet ci = (aiu ,miu , ai ) consisting of user audio aiu , user motion miu , and avatar audio ai .

<!-- page-break:4 -->

Figure 2. Overall architecture of Avatar Forcing. We encode the use motion and audio, as well as avatar audio into a unified condition by Dual Motion Encoder. Causal Motion Generator infer the motion latent block of the avatar, which are then decoded into an avatar video.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000001_3e0e7b787f4cad479770b8a3aed9c5ba56dedfc788810df507ee88083d6e6bfe.png)

Figure 3. Architecture of vθ. The look-ahead causal attention mask enables a smooth transition across the blocks.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000002_cf79acf795e379da16fc3f8ecbab273bd24c82f8b9e1ce62d845ab67d297d1dd.png)

Based on this formulation, we introduce a diffusion forcing-based causal motion generator operating in the motion latent space, which is modeled using a vector field model vθ. As illustrated in Fig. 3, the model vθ comprises two main components: Dual Motion Encoder and Causal DFoT Motion Generation.

The goal of the Dual Motion Encoder is to capture the bidirectional relationship between multimodal user signals and avatar audio, and to encode them into a unified condition. As illustrated in Fig. 3(a), the encoder first takes the user signals (miu and aiu ) and aligns them through a cross-attention layer, which captures the holistic user motion. This representation is then integrated with the avatar audio using another cross-attention layer, which learns the causal relation between the user and the avatar, producing a unified user-avatar-condition. In Sec. 5.5, we empirically validate the importance of using user motion miu for generating an interactive avatar.

For the causal motion generator, we adopt the diffusion forcing transformer (DFoT) [50] with a blockwise causal structure [32, 65]. The latent frames are divided into blocks to capture local bidirectional dependencies within each block while maintaining causal dependencies across different blocks. For each block, we assign a shared noise timestep to all frames and apply an attention mask that prevents the current block from attending to any future blocks.

In Fig. 4, we compare our motion generator architecture with the standard bidirectional DiT architecture used in INFP [70]. Unlike INFP, which requires the full temporal context, our diffusion forcing allows stepwise motion generation under causal constraints and user-avatar interaction with low latency.

Figure 4. Architectural comparison between bidirectional and causal structure. (a) Bidirectional DiT used in INFP [70] requires access to the entire temporal window for motion generation. (b) Our blockwise causal DFoT predicts the next block without using future context and supports KV caching.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000003_256ec8648146bbe6950b5c8dfa9d47dce092c0a9542e6f7804732bd65d164c3f.png)

However, we observe that the strict causal mask in the blockwise causal attention fails to ensure a smooth temporal transition across blocks. To address this issue, we introduce look-ahead in the causal mask, which allows each block to attend to a limited number of future frames while preserving overall causality. We define this blockwise look-ahead causal mask M as follows (illustrated in Fig. 3(b), right):

<!-- formula-not-decoded -->

where i and j are the frame indices, B denotes the block size, and l is the look-ahead frame size. With look-ahead, we effectively mitigate severe per-frame motion jittering observed in the simple blockwise causal structure, which we provide video examples in the supplementary materials.

Based on the blockwise causal structure, the motion generator takes a noisy latent block as input, which is concatenated with the avatar motion latent mS serving as the reference motion. We use a cross-attention layer to condition on the unified condition from the Dual Motion Encoder. Additionally, we employ a sliding-window attention mask of size 2l along the time axis for temporal smooth conditioning. We provide the more details on vθ in Appendix B.

Training and Inference To train the vector field model vθ, we formulate the diffusion forcing training objective in Eq. (1) into a motion latent generation objective as follows:

<!-- page-break:5 -->

## Algorithm 1 Motion inference with KV caching

Require: ODE timesteps {tn}Tn=0, motion generator vθ, video length N, block size B, max cache size M, user inputs (au,mu), avatar audio ai , latent-to-frame decoder Dec, and id latent zS. 1: Divide the frames into L = ⌈N/B⌉ blocks 2: Initialize KV, cKV← [ ], [ ] ▷ Frame &amp; condition caches 3: for i = 1 to L do 4: Sample Noise blockmit0 ∼N(0, I), ▷mit0 ∈ RB×d 5: Acquire User inputs (aiu, miu) and avatar audio ai . 6: Set ci ← (aiu,miu, ai ) ▷ Condition triplet 7: for j = 0 to T do 8: Solve ODE: mitj+1 ← vθ(mitj , tj ; ci , KV, cKV) 9: end for 10: Decode &amp;Return xi1 ←Dec(zS,mi1) ∈ RB×3×H×W 11: Update caches kvi, ckvi ←vθ(zi1, 1; ci ,KV, cKV) 12: if |KV| = |cKV| =M then 13: KV.pop(0) and cKV.pop(0) 14: end if 15: KV.append(kvi) and cKV.append(ckvi) 16: end for

<!-- formula-not-decoded -->

where n ∈ [1,N] denotes the frame index, tn ∈ [0, 1] is the per-frame noise timestep, cn = (aiu ,miu , ai ) is the useravatar condition triplet, mntn ∈ Rd is the noisy motion latent. For simplicity, we omit the reference motion condition mS in Eq. (6).

With the trained model, we can generate the interactive avatar motion given user inputs and avatar audio in an autoregressive rollout. We provide the pseudo code for motion sampling in Algorithm 1, which adopts rolling KV cache in a blockwise manner [25, 65]. Further details on inference are provided in Appendix D.

## 4.2. Enhancing Motion Expressiveness

While our model enables real-time motion generation, we observe that it struggles to produce expressive motions crucial for natural human conversation. In contrast to achieving an accurate lip-sync that has an almost one-to-one match with the avatar audio, appropriate reaction to the user's motion and audio is highly ambiguous, as there is no single correct response [38, 40, 60].

Moreover, as shown in Fig. 5, existing listening datasets generally exhibit lower motion expressiveness compared to speaking datasets, which leads models to learn passive and non-expressive listening behaviors. While prior work attempted to address it with personalized motion [40] or external text instructions [38], these methods do not address the core challenge of learning the true interactive behavior for natural interaction.

To generate vibrant and expressive reactions, we formulate this as an alignment problem and apply the Reinforcement Learning from Human Feedback (RLHF) [41] approach. The main challenge lies in defining an explicit reward for the avatar's interactive behavior, which must account for the interplay between avatar audio, user audio, user motion, and the generated video, which are difficult to evaluate even for humans.

Figure 5. Variance visualization of the L2-norm of 3DMM expressions [16] for the speaker and listener on ViCo [68] dataset. Higher variance indicates higher expressiveness.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000004_fb81032ab4e1d37ec82e803ed6493c12c89ecc1347b298ab87e004af0b402047.png)

We adapted a reward-free method, Direct Preference Optimization (DPO) [46, 58], for fine-tuning our motion generation based on diffusion forcing. Specifically, we construct preference pairs of motion latents (mw ,ml ) as follows:

- Preferred sample mw : the motion latent from the ground-truth video, exhibiting expressive and contextually appropriate responses.
- Less preferred sampleml : themotion latent generated by a separately trained talking avatar model [28], conditioned solely on the avatar audio.

This paired design yields a clear signal for enhancing expressiveness, emphasizing active listening and reactive motion while leaving other aspects, such as lip sync or speechdriven motion, unchanged.

Building on the original DPO objective LDPO (Eq. (2)) with our constructed pairs (mw ,ml ), we fine-tune our diffusion forcing model vθ with the following objective:

<!-- formula-not-decoded -->

where λ is a balancing coefficient. As a result, our model achieves efficient preference alignment without requiring a dedicated reward model, which we validate in Sec. 5.5. We provide the detailed formulation of LDPO in Appendix C.

## 5. Experiments

## 5.1. Dataset and Preprocessing

We use dyadic conversation video datasets: RealTalk [17] and ViCo [68]. We first detect scene changes using PySceneDetect [45] to split the videos into individual clips. We then detect and track each face using FaceAlignment [5], crop and resize it to 512×512. Next, we separate and assign the speaker and listener audio using a visual-grounded speech separation model [31]. All videos are converted to 25 fps, and audio is resampled to

<!-- page-break:6 -->

| Method     | Interaction - User Input   | Interaction - Latency ↓   |   Reactiveness - rPCC-Exp ↓ |   Reactiveness - rPCC-Pose ↓ |   Motion Richness - SID ↑ |   Motion Richness - Var ↑ | Visual Quality - FID ↓   | Visual Quality - FVD ↓   |   Visual Quality - CSIM ↑ |   Lip Synchronization - LSE-D ↓ |   Lip Synchronization - LSE-C ↑ |
|------------|----------------------------|---------------------------|-----------------------------|------------------------------|---------------------------|---------------------------|--------------------------|--------------------------|---------------------------|---------------------------------|---------------------------------|
| FLOAT [28] | ✗                          | 2.4s                      |                       0.054 |                        0.182 |                     2.785 |                     2.778 | 81.297                   | 438.817                  |                     0.845 |                           8.135 |                           6.361 |
| INFP∗ [70] | ✓                          | 3.4s                      |                       0.035 |                        0.064 |                     2.343 |                     1.638 | 24.551                   | 159.000                  |                     0.867 |                           8.027 |                           6.536 |
| Ours       | ✓                          | 0.5s                      |                       0.003 |                        0.036 |                     2.442 |                     1.734 | 24.328                   | 170.874                  |                     0.833 |                           8.060 |                           6.723 |
| GT         | N/A                        | N/A                       |                       0.000 |                        0.000 |                     3.972 |                     1.658 | N/A                      | N/A                      |                     0.796 |                           7.790 |                           6.940 |

Table 1. Quantitative comparison results on RealTalk [17]. Best results highlighted in bold. ∗ denotes the reproduced version that is publicly unavailable. We also report the results from a non-interactive talking head model [28], shown in gray, for reference.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000005_9d120ee705a5b9dff43912b0ed1bca7a79f96d64909e6276b396cc799d52d301.png)

Table 3. Comparison with talking head generation models on the HDTF [67] dataset. Second-best results are underlined.

| Method         |   Visual Quality - FID ↓ |   Visual Quality - FVD ↓ |   Visual Quality - CSIM ↑ |   Lip Synchronization - LSE-D ↓ |   Lip Synchronization - LSE-C ↑ |
|----------------|--------------------------|--------------------------|---------------------------|---------------------------------|---------------------------------|
| SadTalker [66] |                   64.744 |                  342.996 |                     0.697 |                           8.046 |                           7.171 |
| Hallo3 [10]    |                   32.794 |                  184.341 |                     0.865 |                           8.498 |                           7.487 |
| FLOAT [28]     |                   25.110 |                  167.463 |                     0.881 |                           7.553 |                           8.006 |
| INFP* [70]     |                   27.155 |                  187.977 |                     0.840 |                           7.810 |                           7.325 |
| Ours           |                   20.332 |                  149.798 |                     0.870 |                           7.700 |                           7.560 |

16 kHz. Additionally, we randomly select 50 videos from the talking-head dataset HDTF [67] to evaluate the performance of talking-head generation.

## 5.2. Implementation Details

We use the Adam optimizer [29] with a learning rate of 10-4 and a batch size of 8. We retrain the motion latent auto-encoder from Ki et al. [28] on our dataset. The latent dimension is set to d = 512. For our model vθ, we use 8 attention heads with a hidden dimension of h = 1024 and 1D RoPE [52]. It is trained with N = 50 frames over B = 5 blocks (i.e., 10 frames per block) and l = 2 look-ahead frames. For audio encoding, we extract 12 multi-scale features from Wav2Vec2.0 [1]. For motion sampling, we use 10 NFEs with the Euler solver and a classifier-free guidance [22]. All the experiments are conducted on a single NVIDIA H100 GPU.

## 5.3. Interactive Avatar Evaluation

Metrics We evaluate our model across five aspects of interactive avatar generation: Latency, Reactiveness, Motion Richness, Visual Quality, and Lip Synchronization. For Latency, we assume that the pre-extracted audio features and measure motion generation time using 10 NFEs. For Reactiveness, we measure the motion synchronization between the user and the avatar by utilizing the residual Pearson correlation coefficients (rPCC) on facial expression

Table 4. Comparison with listening head generation models on the ViCo [68] dataset. † denotes results inherited from DIM [55].

| Method     |   FD ↓ - Exp |   FD ↓ - Pose |   rPCC ↓ - Exp |   rPCC ↓ - Pose |   SID ↑ - Exp |   SID ↑ - Pose |   Var ↑ - Exp |   Var ↑ - Pose |
|------------|--------------|---------------|----------------|-----------------|---------------|----------------|---------------|----------------|
| RLHG† [68] |        39.02 |          0.07 |           0.08 |            0.02 |          3.62 |           3.17 |          1.52 |           0.02 |
| L2L† [40]  |        33.93 |          0.06 |           0.06 |            0.08 |          2.77 |           2.66 |          0.83 |           0.02 |
| DIM† [55]  |        23.88 |          0.06 |           0.06 |            0.03 |          3.71 |           2.35 |          1.53 |           0.02 |
| INFP* [70] |        17.52 |          0.07 |           0.01 |            0.07 |          2.19 |           3.20 |          2.10 |           0.03 |
| Ours       |        16.64 |          0.05 |           0.01 |            0.01 |          3.12 |           3.00 |          2.80 |           0.03 |

(rPCC-exp) and head pose (rPCC-pose) [11]. For Motion Richness, we measure the Similarity Index for diversity (SID) [40] and variance (Var), following the previous studies [55, 70]. For Visual Quality, we compute the Frechet inception distance (FID) [48] and the Frechet video distance (FVD) [56] for the generated videos. We also assess the identity preservation using the cosine similarity of identity embeddings (CSIM) [12] between the reference avatar image and the generated videos. In Lip Synchronization, we compute the lip sync error distance and confidence (LSE-C and LSE-D) using the generated video and the avatar's audio. Please refer to Appendix D for details on the metrics.

Comparisonwith InteractiveHeadAvatar We compare our model with the reproduced state-of-the-art dyadic talking avatar model, INFP* [70], since its official implementation is not publicly available. For reference, we also include a talking avatar generation model [28] which does not use user motion or audio.

In Tab. 1, we provide a quantitative comparison on the RealTalk [17] dataset. Avatar Forcing significantly outperforms INFP* in terms of Reactiveness and Motion Richness, indicating that our generated avatar is much more reactive and expressive compared to the baselines. Notably, Avatar Forcing achieves a latency of 0.5s, enabling realtime interaction, and maintains Visual Quality and Lip Synchronization comparable to INFP*. In contrast, INFP*'s

<!-- page-break:7 -->

Figure 6. Qualitative comparison of interactive head avatar generation models on the RealTalk [17] dataset. Our model generates more reactive (red arrow) and expressive (red box) avatar motion compared to INFP*. We provide the videos in supplementary materials.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000006_904beeb267246ffc67529db99b927515cabfe19b7aa2bbdc04460920ae88e39a.png)

3.4s latency makes it unsuitable for real-time applications.

We visualize the generated samples of ours and INFP* in Fig. 6, where ours demonstrate more reactive (red arrow) and expressive motions (red box), compared to INFP*.

Human Preference Study We conduct a human preference study comparing our model with the interactive head avatar generation model INFP*. We recruited 42 participants to evaluate 12 video sets using five perceptual metrics: Reactiveness measures how well the avatar reacts to user motion and audio, Motion Richness evaluates expressiveness of the avatar motion, Verbal Alignment evaluates the lip synchronization with the audio, Non-verbal Alignment assesses the non-verbal behaviors such as eye contact or nodding, and Overall Preference. We provide further details of the human study in the supplementary materials.

As shown in Tab. 2, our model is strongly preferred across all metrics, achieving over 80% preference in overall quality. In particular, our generated avatars exhibit expressive motion and strong non-verbal alignment, showing the effectiveness of the preference optimization (Sec. 4.2).

## 5.4. Comprehensive Analysis

Comparisonwith TalkingHeadAvatar We evaluate the talking capability of the avatar. We compare Avatar Forcing with four state-of-the-art talking head avatar generation models: SadTalker [66], Hallo3 [10], FLOAT [28], and INFP∗ . We measure the visual quality with FID, FVD, and CSIM metrics, and assess lip synchronization using LSED and LSE-C used in Tab. 1. We provide the quantitative results on the HDTF dataset [67] in Tab. 3. Avatar Forcing shows competitive performance on all metrics and achieves the best image and video quality. We provide visual examples in Appendix D.

Comparison with Listening Head Avatar We further evaluate the listening capability of our model. We compare Avatar Forcing with the listening head avatar generation models, including RLHG [68], L2L [40], DIM [55], and INFP*, using the ViCo [68]. We measure Frechet distance (FD) for the expression and pose, and rPCC, SID, and Var metrics used in Tab. 1 but with respect to expression and head pose, respectively. Since the baseline models are not publicly available, we take the results from DIM [55]. In Tab. 4, our model outperforms the baselines on almost all of the metrics. In particular, Avatar Forcing achieves the best user-avatar synchronized motion generation (rPCC).

<!-- page-break:8 -->

Table 5. Ablation study on user motion and preference optimization. "w/ mu" indicates whether the user motion latent mu is provided as input to the model during both training and inference.

| Method - w/ mu   | Method - DPO   |   Reactiveness - rPCC-Exp ↓ |   Reactiveness - rPCC-Pose ↓ |   Motion Richness - SID ↑ |   Motion Richness - Var ↑ |
|------------------|----------------|-----------------------------|------------------------------|---------------------------|---------------------------|
| ✗                | ✗              |                       0.052 |                        0.175 |                     2.165 |                     1.586 |
| ✓                | ✗              |                       0.042 |                        0.146 |                     2.236 |                     1.408 |
| ✓                | ✓              |                       0.003 |                        0.036 |                     2.442 |                     1.734 |

## 5.5. Ablation Study

In this section, we conduct ablation studies on (i) the necessity of using user motion in the Dual Motion Encoder and (ii) the importance of preference optimization. We provide further ablation studies, including the diffusion forcing and blockwise look-ahead attention mask in Appendix D.

User Motion To validate the necessity of using user motion for interactive avatar generation, we compare our model with a variant that does not take user motion as input. As shown in Tab. 5, removing user motion leads to significantly less reactive behavior (Reactiveness) and reduced expressiveness in motion (Motion Richness).

Furthermore, as visualized in Fig. 7, the model without user motion produces static behavior whenever the user's audio is silent. This occurs even in the presence of strong non-verbal cues, such as smiling, as the model cannot perceive visual signals. In contrast, our model, which uses user motion, generates a reactive avatar that naturally smiles right after the user smiles (Fig. 7 red arrow) and becomes more focused when the user speaks (Fig. 7 green box).

Direct Preference Optimization To assess the impact of the preference optimization, we compare our model against a variant without fine-tuning. As shown in Tab. 5, preference optimization significantly improves the Reactiveness metrics (rPCC-Exp and rPCC-Pose), which quantify useravatar motion synchronization. It also substantially boosts theMotionRichnessmetrics (SIDandVar), indicatingmore expressive and varied motion. As visualized in Fig. 8, the model without preference optimization generates noticeably reduced diversity in facial expressions and head movement. It also exhibits weaker interaction, failing to respond to the user's smile.

In contrast, our fine-tuned model generates an expressive avatar that shows natural head movement (Fig. 8 red box) and smiles more broadly along with the user (Fig. 8 red arrow). We provide further analysis of our DPO method in Appendix D.

Figure 7. Ablation study on the user motion. Without mu, the avatar remains static even when the user smiles. With mu, our model reacts by smiling after the user (red arrow) and shifting to a focused expression when the user begins speaking (green box).

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000007_88bec1363266f484de73d40bc8c6eae2d52b4d3f3593264e6f813a0244938904.png)

Figure 8. Ablation study on preference optimization. Model fine-tuned with DPO produces more expressive motion (red box) and reactive (red arrow) compared to model without DPO.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000008_58d8270c1cc0ff017551d644cfaadd85788267299954fdab8853a6da56f5d2c9.png)

## 6. Conclusion

We proposed Avatar Forcing, a real-time interactive head avatar model based on diffusion forcing, that generates reactive and expressive motion using both the verbal and nonverbal user signals.Avatar Forcing takes a step toward truly interactive virtual avatars and opens new possibilities for real-time human-AI communication.

Discussion We leave further discussion, including ethical considerations, limitations, and future work, in the Appendix E.

<!-- page-break:9 -->

## 7. Acknowledgment

This work was supported by Institute for Information &amp; communications Technology Planning &amp; Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2019-II190075, Artificial Intelligence Graduate School Program (KAIST)), National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2023-00256259), Center for Applied Research in Artificial Intelligence (CARAI) grant funded by DAPA and ADD (UD190031RD), and the InnoCORE program of the Ministry of Science and ICT (No. N10250156).

## References

- [1] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in Neural Information Processing Systems, 33:12449-12460, 2020. 6
- [2] Philip J. Ball, Jakob Bauer, Frank Belletti, Bethanie Brownfield, Ariel Ephrat, Shlomi Fruchter, Agrim Gupta, Kristian Holsheimer, Aleksander Holynski, Jiri Hron, Christos Kaplanis, Marjorie Limont, Matt McGill, Yanko Oliveira, Jack Parker-Holder, Frank Perbet, Guy Scully, Jeremy Shar, Stephen Spencer, Omer Tov, Ruben Villegas, Emma Wang, Jessica Yung, Cip Baetu, Jordi Berbel, David Bridson, Jake Bruce, Gavin Buttimore, Sarah Chakera, Bilva Chandra, Paul Collins, Alex Cullum, Bogdan Damoc, Vibha Dasagi, Maxime Gazeau, Charles Gbadamosi, Woohyun Han, Ed Hirst, Ashyana Kachra, Lucie Kerley, Kristian Kjems, Eva Knoepfel, Vika Koriakin, Jessica Lo, Cong Lu, Zeb Mehring, Alex Moufarek, Henna Nandwani, Valeria Oliveira, Fabio Pardo, Jane Park, Andrew Pierson, Ben Poole, Helen Ran, Tim Salimans, Manuel Sanchez, Igor Saprykin, Amy Shen, Sailesh Sidhwani, Duncan Smith, Joe Stanton, Hamish Tomlinson, Dimple Vijaykumar, Luyu Wang, Piers Wingfield, Nat Wong, Keyang Xu, Christopher Yew, Nick Young, Vadim Zubov, Douglas Eck, Dumitru Erhan, Koray Kavukcuoglu, Demis Hassabis, Zoubin Gharamani, Raia Hadsell, Aäron van den Oord, Inbar Mosseri, Adrian Bolton, Satinder Singh, and Tim Rocktäschel. Genie 3: A new frontier for world models. 2025. 2
- [3] Antoni Bigata, Rodrigo Mira, Stella Bounareli, Michał Stypułkowski, Konstantinos Vougioukas, Stavros Petridis, and Maja Pantic. Keysync: A robust approach for leakagefree lip synchronization in high resolution. arXiv preprint arXiv:2505.00497, 2025. 2
- [4] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1839218402, 2023. 13
- [5] Adrian Bulat and Georgios Tzimiropoulos. How far are we from solving the 2d &amp; 3d face alignment problem? (and a dataset of 230,000 3d facial landmarks). In International Conference on Computer Vision, 2017. 5
- [6] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion

forcing: Next-token prediction meets full-sequence diffusion. In Advances in Neural Information Processing Systems, 2024. 2, 3, 13

- [7] Junsong Chen, JinchengYu, ChongjianGe, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 13
- [8] Zhiyuan Chen, Jiajiong Cao, Zhiquan Chen, Yuming Li, and Chenguang Ma. Echomimic: Lifelike audio-driven portrait animations through editable landmark conditions. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2403-2410, 2025. 2
- [9] Joon Son Chung and Andrew Zisserman. Out of time: automated lip sync in the wild. In Asian Conference on Computer Vision, 2016. 16
- [10] Jiahao Cui, Hui Li, Yun Zhan, Hanlin Shang, Kaihui Cheng, Yuqi Ma, Shan Mu, Hang Zhou, Jingdong Wang, and Siyu Zhu. Hallo3: Highly dynamic and realistic portrait image animation with video diffusion transformer. In Conference on Computer Vision and Pattern Recognition, 2025. 2, 6, 7, 16
- [11] RadekDanˇeˇcek, Michael J Black, and TimoBolkart. Emoca: Emotion driven monocular face capture and animation. In Conference on Computer Vision and Pattern Recognition, 2022. 6, 14
- [12] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Conference on Computer Vision and Pattern Recognition, 2019. 6, 15, 16
- [13] Yu Deng, Jiaolong Yang, Sicheng Xu, Dong Chen, Yunde Jia, and Xin Tong. Accurate 3d face reconstruction with weakly-supervised learning: From single image to image set. In Conference on Computer Vision and Pattern Recognition Workshops, 2019. 2
- [14] Nikita Drobyshev, Jenya Chelishev, Taras Khakhulin, Aleksei Ivakhnenko, Victor Lempitsky, and Egor Zakharov. Megaportraits: One-shot megapixel neural head avatars. In Association for Computing Machinery International Conference on Multimedia, 2022. 2
- [15] Nikita Drobyshev, Antoni Bigata Casademunt, Konstantinos Vougioukas, Zoe Landgraf, Stavros Petridis, and Maja Pantic. Emoportraits: Emotion-enhanced multimodal one-shot head avatars. In Conference on Computer Vision and Pattern Recognition, 2024. 2
- [16] Panagiotis P Filntisis, George Retsinas, Foivos ParaperasPapantoniou, Athanasios Katsamanis, Anastasios Roussos, and Petros Maragos. Visual speech-aware perceptual 3d facial expression reconstruction from videos. arXiv preprint arXiv:2207.11094, 2022. 5, 14
- [17] Scott Geng, Revant Teotia, Purva Tendulkar, Sachit Menon, and Carl Vondrick. Affective faces for goal-driven dyadic communication. arXiv preprint arXiv:2301.10939, 2023. 5, 6, 7
- [18] Jiazhi Guan, Zhanwang Zhang, Hang Zhou, Tianshu Hu, Kaisiyuan Wang, Dongliang He, Haocheng Feng, Jingtuo Liu, Errui Ding, Ziwei Liu, et al. Stylesync: High-fidelity

<!-- page-break:10 -->

generalized and personalized lip sync in style-based generator. In Conference on Computer Vision and Pattern Recognition, 2023. 2

- [19] Jianzhu Guo, Dingyun Zhang, Xiaoqiang Liu, Zhizhou Zhong, Yuan Zhang, Pengfei Wan, and Di Zhang. Liveportrait: Efficient portrait animation with stitching and retargeting control. arXiv preprint arXiv:2407.03168, 2024. 2, 3
- [20] Ying Guo, Xi Liu, Cheng Zhen, Pengfei Yan, and Xiaoming Wei. Arig: Autoregressive interactive head generation for real-time conversations. In International Conference on Computer Vision, 2025. 3
- [21] Hedra. Hedra realtime avatar. https://hedra.com/ app/avatar, 2025. 1
- [22] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 3, 6
- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840-6851, 2020. 12
- [24] Ailin Huang, Zhewei Huang, and Shuchang Zhou. Perceptual conversational head generation with regularized driver and enhanced renderer. In Association for Computing Machinery International Conference on Multimedia, 2022. 2
- [25] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. In Advances in Neural Information Processing Systems, 2025. 2, 5, 13, 14
- [26] Jianwen Jiang, Chao Liang, Jiaqi Yang, Gaojie Lin, Tianyun Zhong, and Yanbo Zheng. Loopy: Taming audio-driven portrait avatar with long-term motion dependency. arXiv preprint arXiv:2409.02634, 2024. 2
- [27] Taekyung Ki and Dongchan Min. Stylelipsync: Style-based personalized lip-sync video generation. In International Conference on Computer Vision, 2023. 2
- [28] Taekyung Ki, Dongchan Min, and Gyeongsu Chae. Float: Generative motion latent flow matching for audio-driven talking portrait. In International Conference on Computer Vision, 2025. 2, 3, 5, 6, 7, 12, 14, 16
- [29] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 6
- [30] Akio Kodaira, Tingbo Hou, Ji Hou, Masayoshi Tomizuka, and Yue Zhao. Streamdit: Real-time streaming text-to-video generation. arXiv preprint arXiv:2507.03745, 2025. 2
- [31] Kai Li, Runxuan Yang, Fuchun Sun, and Xiaolin Hu. Iianet: An intra-and inter-modality attention network for audiovisual speech separation. In International Conference on Machine Learning, 2024. 5
- [32] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. In Advances in Neural Information Processing Systems, 2024. 4
- [33] Gaojie Lin, Jianwen Jiang, Jiaqi Yang, Zerong Zheng, and Chao Liang. Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models. arXiv preprint arXiv:2502.01061, 2025. 2
- [34] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. 3, 12
- [35] Jin Liu, Xi Wang, Xiaomeng Fu, Yesheng Chai, Cai Yu, Jiao Dai, and Jizhong Han. Mfr-net: Multi-faceted responsive listening head generation via denoising diffusion model. In Association for Computing Machinery International Conference on Multimedia, 2023. 2
- [36] Tao Liu, Feilong Chen, Shuai Fan, Chenpeng Du, Qi Chen, Xie Chen, and Kai Yu. Anitalker: animate vivid and diverse talking faces through identity-decoupled facial motion encoding. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 6696-6705, 2024. 2
- [37] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 12
- [38] Xi Liu, Ying Guo, Cheng Zhen, Tong Li, Yingying Ao, and Pengfei Yan. Customlistener: Text-guided responsive interaction for user-friendly listening head generation. In Conference on Computer Vision and Pattern Recognition, 2024. 2, 5
- [39] Chetwin Low and Weimin Wang. Talkingmachines: Realtime audio-driven facetime-style video via autoregressive diffusion models. arXiv preprint arXiv:2506.03099, 2025. 2
- [40] Evonne Ng, Hanbyul Joo, Liwen Hu, Hao Li, Trevor Darrell, Angjoo Kanazawa, and Shiry Ginosar. Learning to listen: Modeling non-deterministic dyadic facial motion. In Conference on Computer Vision and Pattern Recognition, 2022. 2, 5, 6, 7, 15, 16
- [41] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 2022. 2, 5
- [42] William Peebles and Saining Xie. Scalable diffusion models with transformers. In International Conference on Computer Vision, 2023. 3
- [43] Pika. Pika audio-driven performance model. https:// pika.art/api, 2025. 1
- [44] KR Prajwal, Rudrabha Mukhopadhyay, Vinay P Namboodiri, and CV Jawahar. A lip sync expert is all you need for speech to lip generation in the wild. In Association for Computing Machinery International Conference on Multimedia, 2020. 2
- [45] PySceneDetect. Pyscenedetect. https://github.com/ Breakthrough/PySceneDetect, 2025. 5
- [46] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 2023. 2, 3, 5
- [47] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Conference on Computer Vision and Pattern Recognition, 2022. 2
- [48] Maximilian Seitzer. pytorch-fid: FID Score for PyTorch, 2020. Version 0.3.0. 6, 15

<!-- page-break:11 -->

- [49] Maksim Siniukov, Di Chang, Minh Tran, Hongkun Gong, Ashutosh Chaubey, and Mohammad Soleymani. Ditailistener: Controllable high fidelity listener video generation with diffusion. arXiv preprint arXiv:2504.04010, 2025. 2
- [50] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video diffusion. In International Conference on Machine Learning, 2025. 4
- [51] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. 12
- [52] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024. 6
- [53] Synthesia. Synthesia. https://www.synthesia.io/, 2025. 1
- [54] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions. In European Conference on Computer Vision, pages 244-260, 2024. 2
- [55] Minh Tran, Di Chang, Maksim Siniukov, and Mohammad Soleymani. Dim: Dyadic interaction modeling for social behavior generation. In European Conference on Computer Vision, 2024. 3, 6, 7, 8, 14, 16
- [56] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric &amp; challenges. arXiv preprint arXiv:1812.01717, 2018. 6, 15
- [57] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in Neural Information Processing Systems, 2017. 3
- [58] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Conference on Computer Vision and Pattern Recognition, 2024. 2, 3, 5, 13
- [59] Yaohui Wang, Di Yang, Francois Bremond, and Antitza Dantcheva. Latent image animator: Learning to animate images via latent space navigation. arXiv preprint arXiv:2203.09043, 2022. 2, 12
- [60] Yinuo Wang, Yanbo Fan, Xuan Wang, Guo Yu, and Fei Wang. Diffusion-based realistic listening head generation via hybrid motion modeling. In Conference on Computer Vision and Pattern Recognition, 2025. 2, 5
- [61] Cong Wei, Bo Sun, Haoyu Ma, Ji Hou, Felix Juefei-Xu, Zecheng He, Xiaoliang Dai, Luxin Zhang, Kunpeng Li, Tingbo Hou, et al. Mocha: Towards movie-grade talking character synthesis. arXiv preprint arXiv:2503.23307, 2025. 2
- [62] Sicheng Xu, Guojun Chen, Yu-Xiao Guo, Jiaolong Yang, Chong Li, Zhenyu Zang, Yizhong Zhang, Xin Tong, and Baining Guo. Vasa-1: Lifelike audio-driven talking faces generated in real time. Advances in Neural Information Processing Systems, 2024. 2
- [63] Zunnan Xu, Zhentao Yu, Zixiang Zhou, Jun Zhou, Xiaoyu Jin, Fa-Ting Hong, Xiaozhong Ji, Junwei Zhu, Chengfei Cai, Shiyu Tang, et al. Hunyuanportrait: Implicit condition control for enhanced portrait animation. In Conference on Computer Vision and Pattern Recognition, 2025.
- [64] Hongwei Yi, Tian Ye, Shitong Shao, Xuancheng Yang, Jiantong Zhao, Hanzhong Guo, Terrance Wang, Qingyu Yin, Zeke Xie, Lei Zhu, et al. Magicinfinite: Generating infinite talking videos with your words and voice. arXiv preprint arXiv:2503.05978, 2025. 2
- [65] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Conference on Computer Vision and Pattern Recognition, 2025. 2, 4, 5, 13
- [66] Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, and Fei Wang. Sadtalker: Learning realistic 3d motion coefficients for stylized audiodriven single image talking face animation. In Conference on Computer Vision and Pattern Recognition, 2023. 2, 6, 7, 16
- [67] Zhimeng Zhang, Lincheng Li, Yu Ding, and Changjie Fan. Flow-guided one-shot talking face generation with a highresolution audio-visual dataset. In Conference on Computer Vision and Pattern Recognition, 2021. 6, 7
- [68] Mohan Zhou, Yalong Bai, Wei Zhang, Ting Yao, Tiejun Zhao, and Tao Mei. Responsive listening head generation: a benchmark dataset and baseline. In European Conference on Computer Vision, 2022. 2, 5, 6, 7, 16
- [69] Yang Zhou, Xintong Han, Eli Shechtman, Jose Echevarria, Evangelos Kalogerakis, and Dingzeyu Li. Makelttalk: speaker-aware talking-head animation. Association for Computing Machinery Transactions on Graphics, 2020. 2
- [70] Yongming Zhu, Longhao Zhang, Zhengkun Rong, Tianshu Hu, Shuang Liang, and Zhipeng Ge. Infp: Audio-driven interactive head generation in dyadic conversations. In Conference on Computer Vision and Pattern Recognition, 2025. 2, 3, 4, 6, 14, 16

<!-- page-break:12 -->

## Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation

## Supplementary Material

Organization The appendix is organized as follows: In Appendix A, we provide the additional backgrounds of our work. We describe the details of our model architecture in Appendix B and the preference optimization method in Appendix C. Experimental details are presented in Appendix D, and further discussion is provided in Appendix E.

## A. Background

Flow Matching Flow matching [34, 37] is a generative model that transforms a simple prior distribution p0, for example, a Gaussian distribution, into the target data distribution p1 via an ordinary differential equation (ODE):

<!-- formula-not-decoded -->

where for fixed x0 ∼ p0 and x1 ∼ p1, the intermediate sample is a linear interpolation xt = tx1 + (1 - t)x0. The training objective of flow matching is to regress the vector field vθ toward the target vector field vt = x1 - x0:  \loss \_{FM}(\theta ) = \mathbb {E}\_{t, x\_t} \left [\| v\_{\theta }(x\_t, t) - (x\_1 - x\_0)\|\right ]. \label {eq:train\_fm} (9)

It then generates target samples by solving Eq. (8). Note that flow matching can be interpreted as a diffusion model [23, 51] where a noise schedule follows the linear trajectory between the prior and the target data.

## B. Details on Model Architecture

In Appendix B.1, we provide more details on the motion latent auto-encoder. In Appendix B.2, we provide more details on the vector field predictor vθ.

## B.1. Motion Latent Auto-encoder

In Fig. 9, we show an overview of the motion latent autoencoder. It encodes an image into a latent vector that can be decomposed into an identity representation (i.e., appearance) and the motion representation. This auto-encoder is trained to reconstruct a driving image using a source image that shares the same identity. During each training iteration, the encoder encodes two images S and D drawn from the same video clip, and computes the zS→D := zS +mD that transforms the source image into the reconstructed ˆD. This explicit decomposition yields a compact motion representation, enabling fast motion generation.

We train this auto-encoder on our dataset, following the training objective described in the original paper. For more details, including the property of this latent space, training details, please refer to [28, 59].

Figure 9. Overview of Motion Latent Encoder. It encodes an image into a latent vector that has explicit identity-motion decomposition.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000009_fd2f4f23f577ea0b257c9329c9136a23e0d75dc4f15e2fcc96230b56a0ab5a78.png)

Figure 10. Detailed architecture for Motion Generator in vθ.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000010_24caebfa3496df3fc7b15ca45579308e128843650e7e5109943b93bfca6a7beb.png)

## B.2. Vector Field Model vθ

Model Architecture The model vθ comprises two main components: the Dual Motion Encoder and the Causal DFoT Motion Generator. The Dual Motion Encoder unifies three multimodal inputs through a cross-attention layer, computed as softmax ( QK⊤ √d ) V , where

<!-- formula-not-decoded -->

and Wq,Wk,Wv ∈ Rd×d are learnable projection matrices for the query q, key k, and value v ∈ RN×d , respectively (N is the number of latents). In the first crossattention layer, the encoder captures holistic verbal and nonverbal user motion by using the user motion latent mu as the query. In the second layer, it integrates this aligned user motion with the avatar's audio by taking the avatar audio as the query. We use four attention heads, each with a hidden dimension of d = 512, for both cross-attention layers.

<!-- page-break:13 -->

In Fig. 10, we provide a detailed architecture for the motion generator. It consists of eight DFoT transformer blocks followed by a transformer head. Specifically, in each DFoT block, noisy latents (m1t1 ,m2t2 , · · · ,mNtN ) are modulated by the flow time t = (t1, t2, · · · , tN) through a shared AdaLN scale-shift coefficients (ToTimeEmb layer) [7].

For the attention modules, we use Blockwise Causal Look-ahead Mask (Eq. (5)) in self-attention and a Slidingwindow Attention Mask for aligning the driving signal c1:N to the noisy latents. Specifically, we introduce the Blockwise Casual Look-ahead Mask to ensure the causal motion generation in our motion latent space, which significantly improves the temporal consistency of the generated video, as demonstrated in Appendix D.4. Unlike the recent video diffusion models that employ a spatio-temporal compression module [25, 65] where each latent correlates to multiple video frames by the compression rate (e.g., 4× or 8×), our motion latent has one-to-one correspondence with each frame in pixel space. Under this setting, a simple (blockwise) causal mask alone produces the temporal inconsistencies across the frames or blocks.

## C. Details on Preference Optimization

Training Objective Formulation Inspired by DiffusionDPO [58], we formulate the training objective LDPO in the context of diffusion forcing [6]. Let (ml ,mw ) denote a pair of less-preferred and preferred motion latents, each consisting of N frames, where ml := (ml,n )Nn=1 and mw := (mw,n )Nn=1 . Following the per-token independent noising process of diffusion forcing, we construct the noisy latent pairs as:mw,n

<!-- formula-not-decoded -->

where n ∈ [1,N] is the frame index, tn ∈ [0, 1] is the nth flow time, and m0 := (mn0 )Nn=1 ∈ RN×d is the noise sequence. With these notations, we formulate LDPO as

<!-- formula-not-decoded -->

## Algorithm 2 Motion inference with KV caching (Detailed)

- Require: ODE timesteps {tn}Tn=0, motion generator vθ, video length N, block size B, lookahead size l, max cache size M, user inputs (au,mu), avatar audio ai , latent-to-frame decoder Dec, offset O, and id latent zS. 1: Divide the frames into L = ⌈N/B⌉ blocks 2: Initialize KV, cKV← [ ], [ ] ▷ Frame &amp; condition caches 3: for i = 1 to L do 4: Sample Noise blockmit0 ∼N(0, I), ▷mit0 ∈ RB×d 5: Acquire User inputs (aiu, miu) and avatar audio ai . 6: Set ci ← (aiu,miu, ai ) ▷ Condition triplet 7: Merge offset mit0 , ci ←Concat(mit0 , ci ; Oi ) 8: for j = 0 to T do 9: Solve ODE: mitj+1 ← vθ(mitj , tj ; ci , KV, cKV) 10: end for 11: Decode &amp;Return xi1 ←Dec(zS,mi1) ∈ RB×3×H×W 12: Update caches kvi, ckvi ←vθ(zi1, 1; ci ,KV, cKV) 13: if |KV| = |cKV| =M then 14: KV.pop(0) and cKV.pop(0) 15: end if 16: KV.append(kvi) and cKV.append(ckvi) 17: Update offset Oi+1 ← (mi1[-l :], ci [-l :]) 18: end for

where cn is the n-th unified condition, vref is the reference vector field model, the target vector fields for less-preferred and preferred samples are given by

<!-- formula-not-decoded -->

## D. Experimental Details

## D.1. Inference Details

Classifier-Free Guidance (CFG) We apply independent classifier-free guidance (CFG) [4] for multiple driving conditions. Specifically, we compute the modified vector field ˜ vθ as

<!-- formula-not-decoded -->

where c{x,y} denotes a driving condition with the conditions x and y. a denotes the avatar audio, and u = {au,mu} is the set of user audio au and user motion mu. wa and wu are the CFG scales of the avatar audio and user condition, respectively. We use 10% dropout rate for each condition during training.

Motion Inference Details We provide more details on our inference strategy in Algorithm 2.

While the lookahead attention enables to generate temporally consistent motion across the blocks, naively introducing it incurs additional latency as it requires l future frames. To tackle this problem, we introduce an offset Oi for i-th block generation. Specifically, the offset Oi consists of the last l clean motion latents and the corresponding condition from the previous block:

<!-- page-break:14 -->

<!-- formula-not-decoded -->

where i = 1, · · · , L-1. These offset motion frames are concatenated with the current noisy motion block mitj ∈ RB×d and the condition ci ∈ R3×B×d along the time axis, resulting in l + B motion frames and corresponding conditions (Line 7 in Algorithm 2). Due to the flexibility of diffusion forcing, we can assign difference flow time schedules, i.e., t = 1 for the offset and t = tj for the current noisy latent block.

As we compute the modified vector field ˜ vθ for CFG as in Eq. (14), we separately cache and update the KV of these vector fields vθ(·; c{∅,∅}), vθ(·; c{a,∅}), and vθ(·; c{∅,u}). To obtain the KV caches of the clean block, we compute all three by reusing the generated motion block along with the existing KV caches [25]. Moreover, due to the introduction of lookahead attention and offset, the KV caches are updated except the for last l frames and these frames are provided by the offset. Therefore, the maximum cache size is M = L - B - l = 38.

## D.2. Training Details

We train the vector field model vθ in Eq. (6) for 2000k steps while freezing the motion latent auto-encoder. We use L1 distance for ∥ · ∥. For fine-tuning the model using the proposed preference optimization method in Eq. (7), we set the balancing coefficient to λ = 0.1 and the deviation parameter to β = 1000. We initialize the reference model vref with the same weights as the trained vθ. We fine-tune vθ for 5k steps and observe that additional tuning does not yield further performance gains.

## D.3. Baseline Implementation

One major challenge of evaluating an interactive head avatar model is the absence of official implementation of baseline methods. To bridge this gap, we reproduce INFP [70] on the motion latent space of [28], following its core module, Motion Guider and denote the reproduced model as INFP*. Based on the description in the original paper, we adopt a bidirectional Transformer encoder for motion generation, where a single window consists of N = 75 frames and additional 10 frames serve as context frames. For Motion Guider, we set K = 64 for both verbal and non-verbal motion memory banks. We train INFP* on our dataset for 2000k steps.

## D.4. Additional Ablation Studies

Comparison with Autoregressive Diffusion We compare our motion latent diffusion forcing with standard autoregressive diffusion, where the motion generator is conditioned on clean context motion latents. Specifically, we train the motion generator in an autoregressive diffusion manner using four clean context blocks (40 frames) and one noisy block (10 frames). As shown in Fig. 11, autoregressive diffusion suffers from degraded long-horizon generation, whereas diffusion forcing is much more robust to motion drift, highlighting its necessity.

Ablation on Motion Motion block Size In Tab. 7, we provide an ablation study on motion block size under a fixed number of training frames N = 50. Increasing block size (i.e., reducing the number of frames in each block) leads to lower latency while achieving quantitative performance. Conversely, reducing the block size (i.e., increasing the number of frames in each block) can improve the temporal consistency (FVD) and Lip-sync quality (LSE-D) with higher latency.

Additional Quantitative Results In Tab. 6, we present the ablation studies on our model with additional metrics, including Visual Quality and Lip Synchronization. We provide a video results of the ablation study. Please refer to the videos "02\_ablation\_wo\_user\_motion.mp4" and "02\_ab-lation\_wo\_DPO.mp4". Moreover, we provide video ablation results on the attention mask, where each masking method is illustrated in Fig. 12. The motion jittering observed when using only the blockwise causal mask is clearly visible in the video results, yet difficult to capture with quantitative metrics. We highly recommend watching the ablation video "02\_ablation\_attention\_mask\_XX.mp4".

## D.5. Evaluation Metrics

Reactiveness and Motion Richness Reactiveness and Motion Richness are computed using the EMOCAbased [11] 3D morphable models (3DMMs) that model the facial dynamics via 50-dim expression parameters and 6dim pose parameters. We extract those parameters using an off-the-shelf 3DMM extractor, SPECTRE [16], for each video frame. Let us denote x ∈ RN×d as the ground-truth user parameters, y ∈ RN×d as the ground-truth avatar parameters, and ˆ y ∈ RN×d as the generated avatar parameters. L is the number of frames and d is the feature dimension. As reported in Tab. 1 and Tab. 4, we can compute rPCC, SID, Var, and FD for expression and pose, respectively.

- rPCC (residual Pearson Correlation Coefficients) [55] is to measure the motion synchronization between the user parameters and avatar parameters. Specifically, L1 distance is used to measure the discrepancy between generated PCC and ground-truth PCC where we define PCC as a function of z ∈ RN×d given ground-truth user

<!-- page-break:15 -->

## Training Horizon

## Long Horizon

Figure 11. Comparison of autoregressive diffusion and diffusion forcing. Autoregressive diffusion suffers from motion drift (red arrow) over the long horizon, whereas diffusion forcing maintains stable motion generation over the long horizon.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000011_24feab4a23302979f81520788908610975454cffec4bf1612f7f3d55993fe2d3.png)

Table 6. Ablation study with additional metrics on user motion and preference optimization. "w/ mu" indicates whether the user motion latent mu is provided as input to the model during both training and inference.

| Method - w/ mu   | Method - DPO   |   Reactiveness - rPCC-Exp ↓ |   Reactiveness - rPCC-Pose ↓ |   Motion Richness - SID ↑ |   Motion Richness - Var ↑ |   Visual Quality - FID ↓ |   Visual Quality - FVD ↓ |   Visual Quality - CSIM ↑ |   Lip Synchronization - LSE-D ↓ |   Lip Synchronization - LSE-C ↑ |
|------------------|----------------|-----------------------------|------------------------------|---------------------------|---------------------------|--------------------------|--------------------------|---------------------------|---------------------------------|---------------------------------|
| ✗                | ✗              |                       0.052 |                        0.175 |                     2.165 |                     1.586 |                   28.746 |                  185.593 |                     0.818 |                           8.260 |                           6.423 |
| ✓                | ✗              |                       0.042 |                        0.146 |                     2.236 |                     1.408 |                   25.600 |                  175.322 |                     0.854 |                           8.160 |                           6.803 |
| ✓                | ✓              |                       0.003 |                        0.036 |                     2.442 |                     1.734 |                   24.328 |                  170.874 |                     0.833 |                           8.060 |                           6.723 |

Figure 12. AttentionMaskComparison. (Left) framewise causal mask; (Middle) blockwise causal mask; (Right) blockwise causal look-ahead mask (Ours).

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000012_181e519d72429d925ed6e08cf7c78dc27e8310c6c47a94357cc2105242cf882a.png)

parameters x:

<!-- formula-not-decoded -->

where i ∈ [0,N] is the frame index, and ¯ z and ¯x denote the mean of z and y, respectively. Based on this notation, we can define rPCC as |PCC(y|x) - PCC(ˆ y|x)|.

- SID (Shannon Index for Diversity) [40] is to measure the motion diversity of the generated avatars using K-means clustering on 3DMM parameters. Following [40], we compute the average entropy (Shannon index) of the clusters with K = 15, 9 for expression and pose, respectively.
- Var is the variance of the parameters from generated avatars, which is computed along the time axis and then averaged along the feature axis.
- FD (Frechet Distance) [48] measures the distance between the expression and pose distributions of the gen-

Table 7. Ablation studies on motion block sizes on RealTalk.

| # blocks   |   Latency (s) ↓ |   rPCC-Exp ↓ |   rPCC-Pose ↓ |   FVD ↓ |   SID ↑ |   LSE-D ↓ |
|------------|-----------------|--------------|---------------|---------|---------|-----------|
| 10         |             0.3 |        0.012 |         0.056 |  222.47 |   2.355 |     7.290 |
| 2          |             1.5 |        0.003 |         0.031 |  155.81 |   2.145 |     6.555 |
| 5 (Ours)   |             0.5 |        0.003 |         0.036 |  170.87 |   2.442 |     6.723 |

erated avatars and the ground truth by calculating

<!-- formula-not-decoded -->

where µ and Σ are the mean and the covariance matrix, respectively.

Visual Quality We utilize FID [48] and FVD [56] to assess the image and video quality of the generated avatars, and CSIM [12] to measure the identity preservation performance of avatar generation models.

- FID (Frechet Inception Distance) measures the quality of the generated frames by comparing the distribution of image features extracted from a pre-trained feature extractor [48]. The FD computation in Eq. (17) is adopted using the extracted image features.
- FVD (Frechet Video Distance) quantifies the spatiotemporal quality of the generated videos by comparing the feature distributions of real and generated videos in a learned video feature space [56]. It reflects both framewise quality and temporal consistency. The FD computation in Eq. (17) is adopted using the extracted video features.
- CSIM (Cosine Similarity for Identity Embedding) evaluates identity preservation by computing cosine similarity between the facial embeddings from the generated

<!-- page-break:16 -->

and the source image, extracted using ArcFace [12].

Lip Synchronization We compute LSE-D and LSE-C [9] to assess the alignment between the generated lip motion and the corresponding audio.

- LSE-D and LSE-C (Lip Sync Error Distance and Confidence): Both metrics are derived from a pre-trained SyncNet-based audio-visual synchronization model. LSE-D measures the distance between the audio and lip embeddings, where lower values indicate better synchronization. LSE-C measures the confidence score of synchronized audio-visual pairs, where higher values indicate more accurate lip-audio alignment.

## D.6. Human Evaluation

In Fig. 13, we show the interface used for our human evaluation. To improve the evaluation consistency, we additionally provided participants with a reference test and answer sheet. We asked 42 participants to compare 12 videos based on 5 evaluation metrics and indicate their preference. We also provide a video test sheet. Please refer to "04\_hu-man\_evaluation\_XX.mp4".

## D.7. Supplementary Visual Results

Comparison with Interactive Head Avatar We provide the video results to further support the visual results in Fig. 6. Please refer to "01\_inter-active\_avatar\_comparison\_XX.mp4". We also provide a video comparison results using the DEMO videos of Official INFP [70]. Please refer to "01\_interac-tive\_avatar\_comparison\_demo.mp4"

Comparison with Talking Head Avatar In Fig. 14, we compare our model with SadTalker [66], Hallo3 [10], FLOAT [28], and INFP* [70] for talking head avatar generation bu dropping the user condition at inference. Avatar Forcing can generate competitive results compared to stateof-the-art models, while our model successfully reflects user signals. We also provide the video comparison results. Please refer to "03\_talking\_XX.mp4".

Comparisonwith ListeningHeadAvatar In Fig. 15, we compare our model with RLHG [68], L2L [40], DIM [55], and INFP* [70] for listening head avatar generation. Avatar Forcing can generate competitive results with more expressive facial expression. Please refer to "03\_listen-ing\_XX.mp4" for video results.

## E. Discussion

Ethical Consideration Our method can generate more engaging and interactive head-avatar videos, broadening positive applications such as virtual avatar chat, virtual education, and other communication tools by providing users with a more immersive experience. However, realistic interactive head avatar videos also pose risks of misuse, including identity spoofing or malicious deepfakes. Adding watermarks to generated videos and applying a restricted license can help mitigate these risks. We also encourage the community to use our generated data to train deepfake detection models.

Limitation and Future Work Our system focuses on modeling interactive conversations through a head-motion latent space, which enables natural and expressive interactions. This design limits the modeling of richer bodily cues, such as hand gestures, that contribute to more dynamic communication. Moreover, while our model captures userdriven conversational cues via motion latents, certain scenarios may require more explicit controllability, such as directing eye gaze or emphasizing emotional shifts. We believe that incorporating additional user signals, including eye-tracking or emotion-tracking inputs, can address these limitations. Since our framework imposes no architectural constraints on adding new conditions, such signals can be incorporated in future extensions of our system. While diffusion forcing is robust for long-horizon generation, it does not fully address exposure bias. Addressing this issue in the motion latent space remains future work.

<!-- page-break:17 -->

Figure 13. Human evaluation interface. (Left) Instructions for human evaluation; (Middle) A reference sheet for consistent evaluation; (Right) Test and answer sheet.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000013_066f4ef7385233ebab944029ed19c92a775da12cf719cb242bacf27a58c48359.png)

Figure 14. Qualitative comparison on talking head avatar generation.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000014_02742c02e1fb79aad4e6c0ca594e3c62dffeeebde3db5b1ba9728093d870ff9e.png)

<!-- page-break:18 -->

Figure 15. Qualitative comparison on listening head avatar generation.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/video_pose_driven_generation/forcing%20avatar_real%20time/images/image_000015_989390a22618ef9876d2053723483d4a1991eca97e0caf383d91687c2eb89ac4.png)