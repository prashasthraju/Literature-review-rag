<!-- page-start:1 -->

2026 International Conference on Automatic Face and Gesture Recognition (FG)

## Micro-Expression-Aware Avatar Fingerprinting via Inter-Frame Feature Differencing

Masoumeh Chapariniya1,2 , Jean-Marc Odobez3,4 , Volker Dellwo1 , Teodora Vukovi´c1,2

1Department of Computational Linguistics, University of Zurich, Zurich, Switzerland

2UZH Digital Society Initiative

3 Idiap Research Institute, Martigny, Switzerland

4École Polytechnique Fédérale de Lausanne (EPFL), Lausanne, Switzerland

masoumeh.chapariniya@uzh.ch, odobez@idiap.ch, volker.dellwo@uzh.ch,

teodora.vukovic2@uzh.ch

Abstract-Avatar fingerprinting, i.e., verifying who drives a synthetic talking-head video rather than whether it is real, is a critical safeguard for authorized use of face-reenactment technology. Existing methods rely on a fixed, non-differentiable landmark extraction stage that prevents the fingerprinting model from being optimized end-to-end from raw pixels. We propose a preprocessing-free system built on a micro-expression-aware backbone operating on raw video frames, with inter-frame feature differencing as the core design principle: consecutive feature maps are subtracted in the learned deep feature space, so that temporally stable appearance dimensions contribute zero to the output while driver-specific motion dynamics are preserved. A controlled ablation on NVFAIR confirms that temporal motion accounts for the large majority of discriminative performance, and that raw appearance features actively degrade identity separation. Both the choice of backbone and the differencing principle are essential: differencing alone is insufficient when applied to a generic encoder, as appearance-dominated features collapse to near-identical representations across adjacent frames, while the micro-expression-aware F5C backbone retains measurable motion variation that the differencing operation can exploit. Without any external preprocessing, our model achieves an overall AUC of 0.877 on NVFAIR and matches or exceeds the landmark-based baseline on the majority of cross-generator pairs.

Keywords: avatar fingerprinting, face reenactment, interframe feature differencing, temporal facial dynamics, appearance-agnostic representation, supervised contrastive learning, cross-generator generalization.

## I. INTRODUCTION

The rapid proliferation of neural talking-head generators has fundamentally changed the threat landscape for digital identity. Keypoint-based generators such as Face-vid2vid [1] and TPS [2], latent-space systems such as LIA [3] and LivePortrait [4], and the newest diffusion-based models [5] can synthesize photorealistic talking-head videos from a single image in real time. While these technologies enable immersive telepresence and accessibility applications, they simultaneously lower the barrier for unauthorized avatar impersonation: an attacker can drive another person's digital likeness without consent, at scale, using only a single photograph. Recent surveys identify AI-generated impersonation as

[Code: https://github.com/Mchapariniya/TrustFA-FG2026](https://github.com/Mchapariniya/TrustFA-FG2026)

a primary emerging threat to identity verification systems [6], and regulatory frameworks such as the EU AI Act explicitly require provenance mechanisms for synthetic face media.

Avatar impersonation presents a forensic challenge that is fundamentally different from standard deepfake detection. In deepfake detection, the goal is to distinguish real from synthetic content, where artifact-based detectors [7] exploit rendering artifacts that remain consistent within a given generator. Avatar videos, however, require a different question: who drives the avatar? All inputs are legitimate: the target face image is genuine and the generator is authorized. Only the driving identity is unauthorized. Standard deepfake detectors fail completely here because generator artifacts are identical regardless of who drives the avatar. Avatar fingerprinting [8] addresses this by extracting a biometric signature of the driving identity from the avatar's temporal facial dynamics. The core challenge is that visual appearance is identical across all drivers of the same target face. The only discriminative signal is temporal facial motion, the idiosyncratic micro-dynamics each person imprints during animation [9]-[11]. Existing avatar fingerprinting methods [8], [12], [13] extract these dynamics by first running a facial landmark detector on each frame, then training a downstream model on the resulting geometric features. While effective, this design imposes a structural constraint: the landmark stage is frozen and non-differentiable, so its feature vocabulary is fixed independently of what the fingerprinting objective actually requires.

Sub-landmark facial dynamics likely carry identity information that is too fine-grained to be reliably captured by landmark-based pipelines and too easily suppressed by generic appearance-oriented encoders. We therefore introduce a preprocessing-free avatar fingerprinting framework that combines a micro-expression-aware F5C backbone with inter-frame feature differencing. The backbone preserves coordinated fine-scale facial micro-expression directly from raw frames, whereas differencing removes temporally stable target appearance and retains only temporal variation. This combination is central to our approach: the backbone produces motion-sensitive features, and the differencing step transforms them into an appearance-agnostic representation of driving identity.

<!-- page-break:2 -->

Our contributions are as follows:

- 1) We propose a preprocessing-free, end-to-end avatar fingerprinting framework that combines a microexpression-aware F5C backbone with inter-frame feature differencing.
- 2) Through controlled ablations, we show that temporal differencing is the main source of identity signal, raw appearance features are harmful, generic backbones are poorly suited to feature-space differencing, and longer clips improve performance.
- 3) Our 0.53M-parameter model achieves 0.877 overall AUC on NVFAIR, matches or exceeds the landmarkbased baseline on most cross-generator pairs, and eliminates external landmark tracking.

## II. RELATED WORK

## A. Avatar Fingerprinting

Avatar fingerprinting was introduced by Prashnani et al. [8], who formulated the task as recovering driving identity from temporal facial dynamics in synthetic talking-head videos. Their method represents each frame using facial landmarks, converts them into pairwise normalized Euclidean distances, and learns driver-specific embeddings with a temporal CNN trained using a time-shuffling contrastive objective. Importantly, their ablations show that 3DMM and action-unit features underperform landmark distances because they oversmooth subtle facial motion, underscoring the importance of fine-grained temporal dynamics for this task. This observation directly motivates our focus on sub-landmark micro-motion and on learning motion-sensitive representations directly from raw pixels.

Pedrouzo-Rodriguez et al. [14] advance landmark-based avatar fingerprinting with a spatio-temporal graph model and broader cross-generator evaluation. Their later AVAPrintDB benchmark [12] further shows that current methods remain sensitive to synthesis-pipeline and domain shifts. Yet both approaches share a common structural constraint: the landmark extraction stage is frozen and non-differentiable, confining the learnable model to a pre-defined geometric vocabulary that cannot be jointly optimized for the fingerprinting objective.

Sahadewa et al. [13] show that avatar fingerprinting models can exploit source-dataset shortcuts rather than true driver identity, and address this issue with a raw-RGB ResNet encoder trained using bias-averse objectives. Their approach therefore targets dataset-invariant learning through the training loss and evaluation protocol. By contrast, our method focuses on representation design: we use a micro-expressionaware F5C backbone and inter-frame feature differencing to capture fine-grained facial micro-motion while suppressing appearance structurally, without requiring explicit domain supervision.

Vahdati et al. [15] address avatar fingerprinting from the latent domain, exploiting identity leakage in the pose-andexpression codes transmitted by AI-based videoconferencing systems. Their pose-conditioned contrastive encoder learns persistent identity cues directly from these latents, without reconstructing RGB frames. This setting is complementary to ours: their method assumes access to codec-level streaming latents, whereas we target fingerprinting from recorded videos using raw pixels.

## B. Deepfake Detection via Temporal Identity Features

Avatar fingerprinting is fundamentally different from deepfake detection: all inputs are synthetic by design, so appearance-based artifact cues are not informative. Nevertheless, several identity-aware deepfake detectors are relevant in how they model temporal facial dynamics. ID-Reveal [16] uses 3DMM-derived shape and motion parameters, but its reliance on coarse parametric features highlights the difficulty of capturing the subtle dynamics needed for driver identification. Agarwal et al. [17] model person-specific facial action units, but require per-identity training and are therefore unsuitable for open-set fingerprinting. Audiovisual inconsistency methods [18], [19] further assume access to reference real video, which is unavailable in our fully synthetic setting.

More broadly, recent deepfake detection has shifted toward end-to-end temporal analysis from raw frames. PhaseForensics [20] shows that temporal phase cues carry strong forensic signal, Guo et al. [21] model temporal forgery cues across multiple scales, and Kundu et al. [22] report improved crossdomain generalization without face preprocessing. These works support our emphasis on preprocessing-free, motionsensitive representation learning directly from raw video.

## C. Facial Dynamics as an Identity Signal

The theoretical basis for facial motion as an identity cue is well established. Hill and Johnston [9] show that individuals can be recognized from biological face motion alone on neutral, motion-capture-animated heads; Knappmeyer et al. [10] show that personal motion signatures remain informative even when appearance is averaged across individuals; and O'Toole et al. [11] formalize the dynamic identity signature hypothesis, arguing that idiosyncratic facial motion patterns are sufficient for recognition independently of static appearance.

This view is further supported by later work that removes appearance altogether. ExpressionAuth [23] and FacialMotionID [24] both achieve strong identity recognition from avatar blendshapes or abstract facial-motion signals alone. Chapariniya et al. [25] likewise show that expression dynamics retain substantial identity information after facial shape is explicitly separated. More broadly, dynamic behavioural cues derived from whole-body keypoints have also been shown to support person identification in natural interaction [26], although this lies outside the facial-only setting studied here.

## III. PROPOSED METHOD

Let V d t denote a synthetic video produced by animating target identity IDt with driving identity IDd, where t and d index the target and driver respectively. The visual appearance of V d t is entirely determined by IDt, while the facial motion dynamics are induced by IDd. We seek an embedding ϕ :

<!-- page-break:3 -->

V →RD , where D is the embedding dimension, such that videos sharing the same driver remain close regardless of target appearance, and videos with different drivers are wellseparated even when rendered onto the same target face:

<!-- formula-not-decoded -->

for all t1, t2, t and all d ̸= d′ . The left-hand side requires that two videos driven by the same IDd but rendered onto different target faces t1 and t2 produce similar embeddings (intra-driver distance). The right-hand side requires that two videos rendered onto the same target face t but driven by different identities d and d′ produce dissimilar embeddings (inter-driver distance). Since target appearance is identical across all videos of IDt, it carries no discriminative signal for Eq. 1: the only information that can drive intra-driver similarity and inter-driver separation is the temporal facial motion of IDd. The model must therefore learn an appearanceagnostic, driver-sensitive representation directly from raw frames.

## A. Architecture Overview

Figure 1 illustrates the complete model. Given a clip of T = 64 consecutive grayscale frames, the shared F5C backbone produces a sequence of spatial feature maps ft ∈ R128×16×16 . Inter-frame differences are then computed to form a motion tensor D ∈ R128×16×16×(T-1) , which is finally compressed by a lightweight temporal head into a single video-level identity embedding ϕ(V ).

## B. F5C Backbone

ConvStack. Each frame is first mapped from 1 × 128 × 128 to 128 × 16 × 16 using four convolutional layers with strides (2, 2, 2, 1), kernel sizes (4, 3, 2, 1), BatchNorm, and ReLU. This stage provides a compact but spatially structured representation for subsequent motion modeling.

Fully-Connected Convolution (FCC). The 128-channel feature map is split into two 64-channel halves, each processed by a separate branch. Meta-kernels of size 32 (cropped to the feature map spatial extent of 16 at runtime) give each spatial location a global receptive field across its full row and column [27]. Two asymmetric branches (H-then-W; Wthen-H) are fused by concatenation and 1×1 conv, capturing asymmetric micro-motion (e.g., one eyebrow raised more, which is a signature of individual motor control).

Channel Correspondence Convolution (CCC). CCC models inter-regional coordination by constructing a k-nearestneighbor graph over spatial positions, where similarity is measured in channel space via cosine distance. Each position then aggregates differential information from its neighbors, enabling the network to capture correlated micro-movements across distinct facial regions, such as the synchronized muscle activations that characterize individual expression style, but cannot be represented by an independent landmark trajectories.

## C. Inter-Frame Feature Differencing

A central representation step of our framework is to compute differences between consecutive feature maps:

<!-- formula-not-decoded -->

If we assume that the feature ft can approximately be decomposed according to ft = a + µt, where a is targetspecific appearance that varies slowly across the clip and µt is the time-varying driver-specific motion signal. Then we have

<!-- formula-not-decoded -->

which shows that temporally stable dimensions cancel algebraically, leaving in the feature signal only the time-varying component µt+1 -µt that encodes driver-specific motion. Appearance invariance therefore emerges as a structural consequence of the differencing operation rather than as a learned property, requiring no auxiliary objective or architectural modification beyond the subtraction itself.

This also motivates our departure from MOL's original frame-pair concatenation. In micro-expression recognition, concatenation preserves expression state because the state itself is informative for the task. In avatar fingerprinting, however, the entanglement of the state with target appearance can act as a distractor that is detrimental to the task. Even if the loss and training might in theory encourage the network to remove the useless identity information, this is quite challenging and may require huge training datasets, while still not avoiding appearance shortcuts within the training stage. In spite of its simplicity, subtraction is therefore a much better inductive bias for enforcing sensitivity to change rather than state.

## D. Temporal Identity Head and Training

Temporal head. Two Conv3D layers with stride (1, 2, 1) (temporal×H ×W) progressively downsampling H from 16→8→4, leaving W=16 unchanged, while at the same time reducing the number of channels 128→64→32 . An AdaptiveAvgPool3D (4, 4, 1) layer then equalizes the spatial dimensions to 4×4 and collapses the temporal axis, yielding a 32×4×4 volume that is flattened to a 512-d vector. A twolayer MLP (512→256→256-d, ReLU, Dropout 0.3) followed by ℓ2-normalisation produces the final 256-d embedding ϕ(V ).

Supervised contrastive learning. Supervised contrastive learning [28] is a natural fit for avatar fingerprinting: within each batch it pulls together all videos sharing the same driving identity across different target appearances, generators, and recording conditions, directly enforcing appearance invariance per gradient update. This is critical for the open-set protocol, where the embedding must generalize to identities unseen during training. Accordingly, we train our model with the following supervised contrastive loss:

<!-- formula-not-decoded -->

<!-- page-break:4 -->

Frame-wise

Fig. 1. End-to-end pipeline. Feature extraction: A video clip V composed of T grayscale frames from a generated video pass independently through the F5C backbone (ConvStack → FCC → CCC) to produce per-frame feature maps ft ∈ R128×16×16 . FCC provides global row/column receptive fields, allowing each spatial location to attend across the full face; CCC models inter-regional correlations via a k-NN graph (k=4), capturing coordinated muscle activations. Temporal differencing: Temporal structure is introduced via inter-frame differencing (dt = ft+1 - ft), which suppresses static appearance and retains only motion dynamics. This results in the motion tensor D ∈ R128×16×16×(T-1) . Identity encoder and training: The temporal identity head compresses this tensor D into the embedding ϕ(V ). The model is trained end-to-end using supervised contrastive loss, pulling embeddings from the same driver together while pushing different drivers apart. At inference, the embedding is used to assess whether the video was driven by the claimed identity.

![Image](file:///C:/Users/prash/OneDrive/Desktop/atten/ragalu/parsed/avatar_fingerprinting/Micro-Expression-Aware%20Avatar%20Fingerprinting%20via%20Inter-Frame%20Feature%20Differencing/images/image_000000_35a659626b82b7bf8c88a681faabfa6a2813fc0ed41af77a250a730de6739590.png)

TABLE I IMPLEMENTATION DETAILS. ALL COMPONENTS ARE TRAINED JOINTLY

FROM SCRATCH WITH NO PRETRAINED WEIGHTS AND NO EXTERNAL PREPROCESSORS.

| Setting                                                                                                               | Value                                                                                                                                                                                                                                                              |
|-----------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Input Clip length T Embedding dim. Optimizer Learning rate / WD Warmup SupCon temperature Batch Training Total params | T ×1×128×128 grayscale frames 64 frames (≈2.6 s at 25 fps) 256; CCC graph k = 4 AdamW (β1=0.9, β2=0.999) 10-3 cosine / 10-4 ; grad. clip 1.0 5 epochs linear 0.07 N=16 identities × M=8 videos 150 epochs × 200 steps, AMP (FP16) 0.53M (F5C: 0.06M + head: 0.47M) |

where P(i) denotes all samples in the batch sharing the same driving identity as sample i, and τ = 0.07 is the temperature. This objective explicitly pulls together videos driven by the same person across different targets and reenactment conditions, while pushing away different drivers.

Implementation details. All components are trained jointly from random initialization on the NVFAIR dataset, with no pretrained weights and no external preprocessors. The implementation details are summarized in Table I.

## IV. EXPERIMENTAL SETUP

## A. Dataset and Task Protocol

We evaluate our model on NVFAIR [8], the first large-scale benchmark specifically designed for avatar fingerprinting. The dataset contains more than 650,000 synthetic talking-head videos from 161 identities generated by three reenactment systems: Face-vid2vid [1], TPS [2], and LIA [3]. The source identities are drawn from CREMA-D, RAVDESS, and an additional videoconferencing capture collected in the original benchmark. Each identity is associated with multiple real videos, from which both self-reenactment and crossreenactment videos are synthesized.

We follow the identity-disjoint split protocol of NVFAIR, 112 identities for training, 14 for validation, and 35 for testing. Crucially, the split is identity-based, not file-based: all videos associated with a given identity belong to the same split, cross-reenactments are formed only within each split, and no test identity is ever observed during training. This setup evaluates genuine generalization to unseen drivers rather than memorization of seen identities.

We use the original benchmark terminology. The driving identity is the person in the driving video, and the target identity is the person whose appearance is rendered in the final video. When driving and target identities match, the result is a self-reenactment; when they differ, it is a cross-reenactment. The authentication goal is to verify that a video of a target identity is a self-reenactment rather than an impostor-driven cross-reenactment.

## B. Evaluation Protocol

We follow the AUC computation and identity-disjoint data split defined in Prashnani et al. [8], using the same 112/14/35 train/validation/test identity partition. For each target identity, positive pairs are formed from self-reenactment videos (driver matches target), while negative pairs compare self-reenactments against cross-reenactments (driver differs from target) for the same target face. The final score is the mean AUC across all 35 test identities.

Main evaluation (Table III). The model is trained on videos from all three generators jointly and evaluated on the combined test set, measuring overall fingerprinting performance.

Cross-generator evaluation (Tables IV and V). To assess generalization, we train three separate models, each using videos from only one generator (Face-vid2vid, TPS, or LIA). Each model is then evaluated on all three generators independently, yielding a 3×3 matrix where rows indicate training generator and columns indicate test generator. Diagonal entries reflect in-domain performance; off-diagonal entries measure cross-generator transfer.

<!-- page-break:5 -->

## C. Experimental Conditions

We consider three groups of controlled experiments: inputrepresentation conditions, clip-length ablations, and backbone ablations.

1) Input representation conditions: To isolate the roles of motion, appearance, and differencing, we evaluate four F5C-based input conditions. All of them use the same F5C backbone, temporal identity head, and supervised contrastive objective; only the representation passed to the temporal head differs.

FEAT\_DIFF (proposed). Each frame is encoded independently by F5C to produce feature maps ft, and consecutive feature maps are subtracted as in Eq. 2. This is our main appearance-suppressing, motion-preserving condition.

PIXEL\_DIFF. Frames are differenced in pixel space before encoding, so the backbone receives (It+1 - It) rather than per-frame features. Comparing PIXEL\_DIFF with FEAT\_DIFF isolates whether the gain comes from differencing itself or specifically from performing it in the learned feature space.

RAW\_FEAT. Per-frame feature maps ft are passed directly to the temporal head without differencing, retaining both appearance and motion information.

STATIC. A single center frame is processed, providing a purely spatial baseline with no temporal modeling. Together, these conditions form a controlled hierarchy: STATIC measures appearance-only capacity; RAW\_FEAT adds temporal context while retaining appearance; and PIXEL\_DIFF and FEAT\_DIFF suppress appearance by construction, differing only in where subtraction is applied.

2) Clip-length ablation: To study the effect of temporal context, we vary the clip length T ∈ {16, 32, 64, 128} while keeping the rest of the pipeline fixed. All clip-length experiments use the same F5C backbone, temporal identity head, supervised contrastive loss, joint training on all three generators, and the same 150-epoch training schedule as the main model. This isolates the effect of temporal extent from other design choices.

3) Backbone ablation: To test whether performance depends on the motion sensitivity of the encoder rather than on differencing alone, we replace F5C with ResNet18 and evaluate matched variants under our own training protocol. This comparison is motivated by recent raw-pixel avatar fingerprinting work based on ResNet encoders, especially Sahadewa et al. [13], who use a ResNet identity encoder and bias-averse objectives to learn domain-invariant facial signatures from raw face clips.

Our ResNet18 study is intentionally different from that method: it is not a reproduction of their training objective, but a controlled backbone comparison within our framework. To isolate encoder effects, we keep the data split, grayscale input format, supervised contrastive loss, embedding dimension, and 150-epoch training schedule aligned with the F5C experiments, and change only the encoder family.

For ResNet18, we evaluate practical variants that mirror the input conditions above. In particular, RESNET18-PIXEL\_DIFF computes frame differences in pixel space before encoding, whereas RESNET18-RAW\_FEAT encodes raw per-frame inputs without differencing. This setup allows us to test whether a generic appearance-oriented backbone can benefit from the same differencing strategy as F5C, or whether the gains depend on an encoder explicitly designed for fine-grained facial micro-motion.

## D. Temporal Sampling

For a chosen clip length T , we extract consecutive grayscale frames from each video rather than using strided sampling. This is important because inter-frame differencing operates on adjacent frames, and skipping frames would weaken the fine-grained temporal continuity of the motion signal.

During training, we apply random temporal cropping as data augmentation. Given a video with N frames, we sample a starting index

<!-- formula-not-decoded -->

and extract the consecutive subsequence {Is, Is+1, . . . , Is+T-1}. This exposes the model to different temporal segments across epochs and reduces sensitivity to any particular clip position.

During evaluation, we use deterministic center cropping for reproducibility:

<!-- formula-not-decoded -->

The clip is then formed as {Is, Is+1, . . . , Is+T-1}. Videos shorter than T frames are padded by repeating the final frame until the required length is reached.

## V. RESULTS

## A. Method Comparison

Table II compares our approach with prior avatar fingerprinting systems from a deployment perspective. Landmarkbased methods [8], [14] require an external per-frame keypoint extraction stage before the fingerprinting model can operate. In contrast, our method is trained end-to-end from raw frames and removes this landmark-tracking dependency entirely. Despite this simpler pipeline, our best model remains competitive with prior landmark-based systems while preserving crossgenerator generalization.

## B. Main Benchmark Performance

Table III reports performance on the full NVFAIR benchmark across all three generators. Our proposed FEAT\_DIFF model achieves an overall AUC of 0.877, outperforming the landmark-based NVFAIR baseline (0.853) by +0.024. At the per-generator level, FEAT\_DIFF is essentially on par with the baseline on Face-vid2vid (0.869 vs. 0.870), while improving substantially on LIA (+0.042) and TPS (+0.030). These gains are obtained without any external landmark extraction or preprocessing.

PIXEL\_DIFF also performs competitively, reaching 0.861 overall AUC. This shows that differencing itself is a strong design principle. However, FEAT\_DIFF consistently outperforms PIXEL\_DIFF, indicating that applying differencing in the learned feature space yields more discriminative motion representations than applying it directly in pixel space.

<!-- page-break:6 -->

TABLE II

PRACTICAL DEPLOYMENT PROPERTIES OF AVATAR FINGERPRINTING METHODS. ✓ = YES; ✗ = NO. E2E: END-TO-END TRAINABLE. KP-FREE: NO EXTERNAL KEYPOINT DETECTOR. RT: NO PER-FRAME LANDMARK-TRACKING OVERHEAD. † PER-GENERATOR AUC FROM RESPECTIVE PAPERS. ‡ MEAN AUC ACROSS GENERATORS, OUR EXPERIMENTS.

TABLE III

| Method                    | Dataset      | Input type               | E2E   | KP-free   | RT   | AUC†   | Cross-gen   |
|---------------------------|--------------|--------------------------|-------|-----------|------|--------|-------------|
| NVFAIR [8]                | NVFAIR       | Landmark dist. (126 pts) | ✗     | ✗         | ✗    | 0.87   | ✓           |
| IsItReallyYou [14]        | GAGAvatar-BM | Landmark feat. (109 pts) | ✗     | ✗         | ✗    | 0.80   | -           |
| Unmasking Puppeteers [15] | NVIDIA-VC    | Codec latents            | ✗     | ✓         | ✓    | -      | ✓           |
| ResNet18 + PIXEL_DIFF     | NVFAIR       | Raw grayscale            | ✓     | ✓         | ✓    | 0.759‡ | ✗           |
| F5C + STATIC              | NVFAIR       | Raw grayscale            | ✓     | ✓         | ✓    | 0.610  | ✗           |
| F5C + RAW_FEAT            | NVFAIR       | Raw grayscale            | ✓     | ✓         | ✓    | 0.649  | ✗           |
| F5C + PIXEL_DIFF          | NVFAIR       | Raw grayscale            | ✓     | ✓         | ✓    | 0.861  | ✓           |
| F5C + FEAT_DIFF (ours)    | NVFAIR       | Raw grayscale            | ✓     | ✓         | ✓    | 0.877‡ | ✓           |

PER-GENERATOR AUC ON NVFAIR OVER 35 TEST IDENTITIES. † REPORTED IN [8].

TABLE IV

| Method                      |   Overall |   FV2V |   LIA |   TPS |
|-----------------------------|-----------|--------|-------|-------|
| NVFAIR†                     |     0.853 |  0.870 | 0.840 | 0.850 |
| FEAT_DIFF (ours) PIXEL_DIFF |     0.877 |  0.869 | 0.882 | 0.880 |
| (ours)                      |     0.861 |  0.849 | 0.866 | 0.868 |

FEAT\_DIFF CROSS-GENERATOR AUC. DIAGONAL ENTRIES ARE IN BOLD.

| Train \ Test   |   FV2V |   TPS |   LIA |
|----------------|--------|-------|-------|
| Face-vid2vid   |  0.869 | 0.833 | 0.842 |
| TPS            |  0.838 | 0.879 | 0.872 |
| LIA            |  0.852 | 0.869 | 0.902 |

## C. Cross-Generator Generalization

Tables IV and V summarize cross-generator transfer. Both FEAT\_DIFF and PIXEL\_DIFF generalize well across unseen generators, with positive average gains on two of the three test generators relative to the landmark baseline. The strongest single result is obtained when training and testing on LIA, where FEAT\_DIFF reaches 0.902 AUC and PIXEL\_DIFF reaches 0.893.

The similar pattern across the two tables suggests that the differencing operation is the main source of crossgenerator robustness. At the same time, FEAT\_DIFF remains consistently stronger overall, showing that motion-sensitive feature-space representations provide an additional advantage beyond differencing alone. The only setting in which our method falls below the landmark baseline is when training on Face-vid2vid and testing on TPS or LIA, which mirrors an already weak transfer pattern in the NVFAIR baseline and therefore appears to reflect generator mismatch rather than a limitation specific to our architecture.

## D. Controlled Ablation Analyses

1) Representation Ablation: Table VI isolates the roles of motion, appearance, and differencing. FEAT\_DIFF performs best, while STATIC performs worst, showing that temporal PIXEL\_DIFF CROSS-GENERATOR AUC.

TABLE V

TABLE VI

| Train \ Test   |   FV2V |   TPS |   LIA |
|----------------|--------|-------|-------|
| Face-vid2vid   |  0.865 | 0.822 | 0.816 |
| TPS            |  0.839 | 0.891 | 0.873 |
| LIA            |  0.857 | 0.862 | 0.893 |

AUC ACROSS FOUR F5C-BASED REPRESENTATION CONDITIONS AT T=64.

| Condition   |   Overall |   FV2V |   TPS |   LIA |
|-------------|-----------|--------|-------|-------|
| FEAT_DIFF   |     0.877 |  0.869 | 0.882 | 0.880 |
| PIXEL_DIFF  |     0.861 |  0.849 | 0.866 | 0.868 |
| RAW_FEAT    |     0.649 |  0.719 | 0.684 | 0.682 |
| STATIC      |     0.610 |  0.685 | 0.631 | 0.630 |

motion is essential for driver identification. RAW\_FEAT also underperforms substantially, indicating that preserving appearance harms identity separation rather than helping it. Since RAW\_FEAT contains both appearance and motion whereas FEAT\_DIFF removes only the temporally stable component, this gap suggests that appearance-correlated dimensions introduce conflicting gradients during training.

The comparison between FEAT\_DIFF and PIXEL\_DIFF shows that differencing is the main source of discriminative power, but that feature-space differencing remains preferable when the backbone preserves useful motion structure.

2) Temporal Context Ablation: Table VII shows that performance improves monotonically as the clip length increases from 16 to 128 frames. This confirms that longer temporal context provides additional identity-bearing motion information. We use T=64 as the default configuration because it offers a good trade-off between accuracy, memory, and runtime while still capturing approximately 2.6 s of consecutive micro-motion.

3) Backbone Ablation: To test whether the gains arise from differencing alone or from the motion sensitivity of the encoder, we replace F5C with ResNet18 and re-evaluate the same four conditions under matched training settings. Table VIII shows that the best ResNet18 variant remains far below F5C despite using substantially more parameters

<!-- page-break:7 -->

Fig. 2. Cross-generator generalization for FEAT\_DIFF (solid) versus the NVFAIR baseline (dashed). Each panel corresponds to a model trained on a single generator and evaluated on all three generators.

![Image](file:///C:/Users/prash/OneDrive/Desktop/atten/ragalu/parsed/avatar_fingerprinting/Micro-Expression-Aware%20Avatar%20Fingerprinting%20via%20Inter-Frame%20Feature%20Differencing/images/image_000001_32942874db388bee32585a3362b56d0a16be3437694416197f607c6f1356a772.png)

TABLE VII CLIP-LENGTH ABLATION (F5C + FEAT\_DIFF, SUPCON, 150 EPOCHS, ALL GENERATORS JOINTLY). T=64 IS THE DEFAULT CONFIGURATION.

|   T |   Overall |   FV2V |   TPS |   LIA |
|-----|-----------|--------|-------|-------|
|  16 |     0.813 |  0.808 | 0.817 | 0.816 |
|  32 |     0.862 |  0.853 | 0.868 | 0.866 |
|  64 |     0.877 |  0.869 | 0.880 | 0.882 |
| 128 |     0.891 |  0.883 | 0.895 | 0.896 |

TABLE VIII

BACKBONE ABLATION: RESNET18 VARIANTS VS. F5C (FEAT\_DIFF). ALL CONDITIONS USE SUPCON (τ=0.07), T=64, 150 EPOCHS, AND ARE TRAINED JOINTLY ON ALL GENERATORS.

| Method                |   Overall |   FV2V |   TPS |   LIA |
|-----------------------|-----------|--------|-------|-------|
| RESNET18-STATIC       |     0.581 |  0.695 | 0.596 | 0.599 |
| RESNET18-RAW_FEAT     |     0.581 |  0.693 | 0.609 | 0.608 |
| RESNET18-FEAT_DIFF    |     0.547 |  0.567 | 0.553 | 0.556 |
| RESNET18-PIXEL_DIFF   |     0.759 |  0.785 | 0.763 | 0.764 |
| FEAT_DIFF (F5C, ours) |     0.877 |  0.869 | 0.880 | 0.882 |

## (11.6M vs. 0.53M).

In particular, ResNet18 with feature-space differencing collapses below even the static baseline, indicating that relying on a generic appearance-oriented encoder does not preserve useful inter-frame variation. By contrast, the FCC and CCC modules in F5C are specifically designed to encode finescale, sub-landmark motion patterns that vary measurably across frames even when global appearance remains constant. Pixel-space differencing partially recovers performance by making motion explicit at the input, but it still remains well below F5C with feature-space differencing. This confirms that effective avatar fingerprinting depends on both differencing and a backbone explicitly suited to micro-motion encoding.

## E. Limitations and Future Work

Our evaluation is currently limited to NVFAIR, which covers three pre-diffusion reenactment generators. Although the proposed method shows strong overall and cross-generator performance in this setting, its behavior on newer diffusionbased portrait generators and on subjects with very weak facial expressivity remains to be established. A natural next step is evaluation on broader benchmarks such as AVAPrintDB and GAGAvatar-based test sets, which would better assess robustness under contemporary synthesis pipelines and more diverse motion regimes. Several methodological limitations also remain. First, we do not study robustness against adaptive or adversarial attacks; an informed attacker may attempt to suppress or perturb the temporal cues exploited by inter-frame differencing. Second, while the appearancesuppression effect is supported by both the ablation results and the algebraic interpretation in Eq. 2, a more formal disentanglement analysis would strengthen the theoretical grounding of the method. Finally, extensions such as multiclip score aggregation and joint training across generators may further improve verification stability and transfer. From a deployment perspective, avatar fingerprinting should be viewed as one component of a broader trust pipeline, requiring continued evaluation under stronger attack models and careful governance of biometric embeddings.

## VI. CONCLUSION

We presented a preprocessing-free avatar fingerprinting framework that combines a micro-expression-aware F5C backbone with inter-frame feature differencing, suppressing target appearance by construction while retaining driverspecific micro-motion. On NVFAIR, our 0.53M-parameter model achieves 0.877 overall AUC, matches or exceeds the landmark-based baseline on most cross-generator pairs, and requires no external landmark tracking. Controlled ablations establish the key design principles: temporal differencing is the dominant source of identity signal, raw appearance features are actively detrimental, and feature-space differencing requires a backbone explicitly sensitive to fine-grained facial micro-motion. These findings position inter-frame feature differencing as a simple but principled mechanism for appearance-agnostic driver verification, pointing avatar fingerprinting beyond fixed geometric pipelines toward more robust verification under diverse talking-head generation systems.

<!-- page-break:8 -->

## REFERENCES

- [1] T.-C. Wang, A. Mallya, and M.-Y. Liu, "One-shot free-view neural talking-head synthesis for video conferencing," in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 10 034-10 044.
- [2] J. Zhao and H. Zhang, "Thin-plate spline motion model for image animation," in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 3657-3666.
- [3] Y. Wang, D. Yang, F. Brémond, and A. Dantcheva, "Latent image animator: Learning to animate images via latent space navigation," in International Conference on Learning Representations (ICLR), 2023.
- [4] J. Guo, D. Zhang, X. Liu, Z. Zhong, Y. Zhang, P. Wan, and D. Zhang, "LivePortrait: Efficient portrait animation with stitching and retargeting control," 2024.
- [5] L. Tian, Q. Wang, B. Zhang, and L. Bo, "EMO: Emote portrait alive - generating expressive portrait videos with audio2video diffusion model under weak conditions," in European Conference on Computer Vision (ECCV), 2024.
- [6] L. Goncharov, I. Petrov, D. Malakhov, and V. Fedotov, "Identity deepfake threats to biometric authentication systems: Public and expert perspectives," 2025.
- [7] A. Haliassos, K. Vougioukas, S. Petridis, and M. Pantic, "Lips don't lie: A generalisable and robust approach to face forgery detection," in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 5039-5049.
- [8] E. Prashnani, K. Nagano, S. De Mello, D. Luebke, and O. Gallo, "Avatar fingerprinting for authorized use of synthetic talking-head videos," in European Conference on Computer Vision (ECCV), 2024, pp. 209-228.
- [9] H. Hill and A. Johnston, "Categorizing sex and identity from the biological motion of faces," Current Biology, vol. 11, no. 11, pp. 880-885, 2001.
- [10] B. Knappmeyer, I. Thornton, and H. Bülthoff, "Facial motion can determine facial identity," Journal of Vision, vol. 1, no. 3, pp. 293-293, 2001.
- [11] A. J. O'Toole, D. A. Roark, and H. Abdi, "Recognizing moving faces: A psychological and neural synthesis," Trends in Cognitive Sciences, vol. 6, no. 6, pp. 261-266, 2002.
- [12] L. Pedrouzo-Rodriguez, L. F. Gomez, R. Tolosana, R. Vera-Rodriguez, R. Daza, A. Morales, and J. Fierrez, "Leveraging avatar fingerprinting: A multi-generator photorealistic talking-head public database and benchmark," 2026.
- [13] M. Sahadewa, M. Marchellus, and I. K. Park, "Bias-averse learning for mitigating source dataset bias in avatar fingerprinting," IEEE Access, vol. 14, pp. 34790-34801, 2026.
- [14] L. Pedrouzo-Rodriguez, P. Delgado-DeRobles, L. F. Gomez, R. Tolosana, R. Vera-Rodriguez, A. Morales, and J. Fierrez, "Is it really you? exploring biometric verification scenarios in photorealistic talking-head avatar videos," 2025.
- [15] D. S. Vahdati, T. D. Nguyen, E. Prashnani, K. Nagano, D. Luebke, O. Gallo, and M. C. Stamm, "Unmasking puppeteers: Leveraging bio-

metric leakage to disarm impersonation in AI-based videoconferencing," 2025.

- [16] D. Cozzolino, A. Rössler, J. Thies, M. Nießner, and L. Verdoliva, "ID-Reveal: Identity-aware deepfake video detection," in IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 15 10815 117.
- [17] S. Agarwal, H. Farid, Y. Gu, M. He, K. Nagano, and H. Li, "Protecting world leaders against deep fakes," in IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019, pp. 38-45.
- [18] D. Cozzolino, M. Nießner, and L. Verdoliva, "Audio-visual person-ofinterest deepfake detection," in IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2024.
- [19] B. Liu, B. Liu, M. Ding, T. Zhu, and X. Yu, "Ti2Net: Temporal identity inconsistency network for deepfake detection," IEEE Transactions on Information Forensics and Security, vol. 19, pp. 1-14, 2024.
- [20] Y. Sun, H. H. Nguyen, J. Yamagishi, and I. Echizen, "Generalizable deepfake detection with phase-based motion analysis," IEEE Transactions on Image Processing, vol. 34, pp. 100-112, 2025.
- [21] L. Guo, F. Guo, T. Lv, J. Zhang, R. Feng, and C. Lu, "Face forgery video detection via temporal forgery cue unraveling," in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.
- [22] R. Kundu, A. Saha, A. Bhattacharya, and A. Roy-Chowdhury, "Towards a universal synthetic video detector: From face or background manipulations to fully AI-generated content," in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.
- [23] T. Jitpanyoyos, Y. Sato, S. Maeda, M. Nishigaki, and T. Ohki, "ExpressionAuth: Utilizing avatar expression blendshapes for behavioral biometrics in VR," in 2024 IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (VRW), 2024, pp. 679-680.
- [24] A. Castro, S. Hanisch, M. Fallahi, and T. Strufe, "FacialMotionID: Identifying users of mixed reality headsets using abstract facial motion representations," 2025.
- [25] M. Chapariniya, P. Vuillecard, J.-M. Odobez, V. Dellwo, and T. Vukovi´c, "Investigating identity signals in conversational facial dynamics via disentangled expression features," 2025.
- [26] M. Chapariniya, T. Vukovi´c, S. Ebling, and V. Dellwo, "Beyond appearance: Transformer-based person identification from conversational dynamics," in 2025 15th International Conference on Computer and Knowledge Engineering (ICCKE). IEEE, 2025, pp. 1-6.
- [27] Z. Shao, Y. Cheng, F. Li, Y. Zhou, X. Lu, Y. Xie, and L. Ma, "MOL: Joint estimation of micro-expression, optical flow, and landmark via transformer-graph-style convolution," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 10, pp. 8756-8768, 2025.
- [28] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, and D. Krishnan, "Supervised contrastive learning," in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 18661-18673.