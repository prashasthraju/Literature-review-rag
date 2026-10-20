<!-- page-start:1 -->

## InstructAvatar: Text-Guided Emotion and Motion Control for Avatar Generation

Yuchi Wang, Junliang Guo, Jianhong Bai, Runyi Yu, Tianyu He, Xu Tan, Xu Sun, and Jiang Bian

Peking University wangyuchi@stu.pku.edu.cn

[https://wangyuchi369.github.io/InstructAvatar/](https://wangyuchi369.github.io/InstructAvatar/)

Fig. 1: InstructAvatar enables emotional talking face generation through a flexible natural language interface (top 2 rows). The generated results exhibit fine-grained expression control, excellent identity preservation, high-quality lip sync, and natural character movements. Moreover, it supports direct control of facial motion and expression without relying on audio cues, a feature absent in previous studies (bottom 2 rows). The ability of InstructAvatar to handle highly out-of-domain appearances (like cartoons, sketches, and sculptures) further highlights its generalization capabilities.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000000_27d3b02988531140e235e692e66838532093232db66d883d6bb8f836661a97bd.png)

<!-- page-break:2 -->

- 2 Yuchi Wang et al.

Abstract. Recent talking avatar generation models have made strides in achieving realistic and accurate lip synchronization with the audio, but often fall short in controlling and conveying detailed expressions and emotions of the avatar, making the generated video less vivid and controllable. In this paper, we propose a novel text-guided approach for generating emotionally expressive 2D avatars, offering fine-grained control, improved interactivity and generalizability to the resulting video. Our framework, named InstructAvatar, leverages a natural language interface to control the emotion as well as the facial motion of avatars. Technically, we design an automatic annotation pipeline to construct an instruction-video paired training dataset, equipped with a novel twobranch diffusion-based generator to predict avatars with audio and text instructions at the same time. Experimental results demonstrate that InstructAvatar produces results that align well with both conditions, and outperforms existing methods in fine-grained emotion control, lip-sync quality, and naturalness.

Keywords: Emotional Talking Avatar · Facial Motion Control · Text Guided · Diffusion Model

## 1 Introduction

Avatar generation has recently gained significant attention due to its broad applicability in film production, gaming, video conferencing, and various other domains. The primary objective of this technology is to animate portraits with synchronized speech audio. While previous studies have achieved impressive lip synchronization and head pose prediction [4, 14, 33, 35, 40, 41, 52, 56], effectively conveying and controlling detailed expressions and motions remains a challenge, resulting in less vivid and authentic videos. Previous studies have attempted to integrate emotional information through either labels [8, 12, 34, 39, 47] or example videos [18,25,26,37,49]. However, they continue to face challenges related to limited flexibility and controllability.

As a seamless interface bridging the gap between humans and computers, the textual prompt stands out as a natural solution to tackle the limitations of previous approaches, providing a versatile array of controls that encompass desired motions and expressions for the avatar. Therefore, we steer towards a textual instruction-based talking avatar generation model named InstructAvatar, which offers three key advantages: (1) Enhanced control over fine-grained details rather than just the overall style; (2) Improved generalizability compared to limited emotion or style categories; and (3) Enhanced interactivity and user-friendliness. As illustrated in Fig. 1, our framework facilitates text-guided emotional talking face generation with fine-grained control, while enabling facial motion/expression control without relying on audio cues.

To accomplish this, we meticulously design our algorithm, taking into account both data and model architecture considerations. For data preparation, existing datasets [3,39] only offer tag-level emotion annotations. To capture fine-grained facial details, we utilize Action Units (AUs) [7] to describe facial muscle movements. AUs are extracted from each video clip using an off-the-shelf model [23]. Subsequently, we prompt a large-scale multi-modal language model GPT-4V [27] to paraphrase AUs into a natural textual description. Additionally, leveraging the visual capabilities of GPT-4V, we refine the detected AUs and provide additional facial information. As a result, we generate a dataset containing detailed emotion and motion descriptions. Regarding the model architecture of InstructAvatar, we propose a diffusion model with two branches of cross-attentions to incorporate different types of instructions while generating the talking video, i.e., the emotion instructions that are high-level throughout the entire video, and the facial motions that are dynamic over timestamps. Additionally, novel techniques such as zero-convolution gate are proposed to stabilize the training and enhance the guidance, which will be detailed in Sec. 3.

<!-- page-break:3 -->

For experiments, we propose several tailored evaluation metrics to justify the model's performance on fine-grained facial emotion and motion control. Experimental results demonstrate that: (1) InstructAvatar exhibits significant improvements in emotion control, lip-sync quality, and naturalness compared to previous baselines. (2) Notably, our model has a natural language interface, enabling it to receive a much wider range of instructions. (3) Additional experimental results indicate that our model can, for the first time, effectively animate avatars directly without audio. In summary, the contributions of our paper are as follows:

(1) We introduce InstructAvatar, a diffusion-based avatar generation model featuring a fine-grained natural language instruction interface. It showcases superior flexibility, control effectiveness, and naturalness compared to previous methods. To our best knowledge, it is the first text-guided 2D-based talking face generation framework.

(2) Through meticulous design, like the implementation of a two-branch cross-attention mechanism, we integrate text-guided facial motion control into our unified framework, further enhancing the scope of avatar control.

(3) We annotate an instruction-video dataset and establish an evaluation pipeline for the fine-grained emotional talking video generation task, which may facilitate further research.

## 2 Related Works

The rapid advancement of generative models such as GANs [10] and diffusion models [16] has led to remarkable progress in avatar generation. Several studies have explored the direct generation of facial avatars [2, 20, 21, 46, 50], as well as the editing of human face images [17, 28, 38, 44]. However, these works often produce static images without dynamic effects. In contrast, talking head generation, a fundamental task in avatar generation, aims to generate a video in which an avatar speaks the provided audio. This task can be broadly categorized into video-driven and audio-driven approaches. In video-driven methods [31, 36, 42, 48], the movement of a portrait is generated based on another driving video, while in audio-driven methods [14,35,52,56], motion is predicted directly from audio inputs. To tackle the challenge of learning facial motion representations, previous talking head models have often relied on domain priors like warping-based transformations [12,22,40,56] or 3DMorphable Models [30,52,53]. Recently, [14] proposed a disentangled motion and appearance architecture and collected a large-scale dataset to directly learn the data distribution, thereby further enhancing the naturalness and diversity of the generated avatars.

<!-- page-break:4 -->

Acknowledging the constraints of prior efforts that often yield emotionless avatars, there has been growing interest in injecting emotions into talking face generation. For instance, MEAD [39] represents emotion using a one-hot vector, while EAT [8] employs a mapping network to extract emotion guidance through a latent code. EAMM [18] represents the facial dynamics of reference emotional video as displacements to motion representations. PD-FGC [37] disentangles control over specific facial organs with emotional expression, and StyleTalk [25] develops a style encoder to extract the style of a reference video. These methods either support a limited range of coarse emotion types [8,12,34,39,47] or necessitate users to seek out another desired style video [18,25,26,37,49], limiting the flexibility and controllability of the generated avatars.

More recently, some endeavors have aimed to incorporate text as an emotion control signal [24,45,54,55]. However, they typically utilize text to generate emotional talking 3D animations, which requires an external renderer to convert these animations into real talking videos. Although this animation-based approach may alleviate difficulty by introducing more domain priors, it inherently leads to indirect controls and restricted diversity [14].

In this paper, we propose to directly learn the distribution of talking videos and enable fine-grained controlling with textual prompts, improving the naturalness and controllability of generation results.

## 3 Methodology

In this section, we introduce the InstructAvatar model. In Sec. 3.1, we provide an overview of the architecture of InstructAvatar. Subsequently, Sec. 3.2 describes the process of constructing high-quality text instructions, while Sec. 3.3 delineates each component of our InstructAvatar. Lastly, in Sec. 3.4, we demonstrate the training and inference pipelines.

## 3.1 Overview

Given a sequence of audio clips A = [a1, a2, ..., aN], one portrait image I, and the text instruction T, our model is tasked with animating the portrait to utter the audio with the target style represented by the instruction. In other words, we aim to learn a mapping to generate a video V = F(A, I, T).

As illustrated in Fig. 2, we decompose F into two parts: variational autoencoder (VAE) H and diffusion-based motion generator G. The VAE follows the approach outlined in [14] to disentangle motion information from appearance, which means that we can derive V = H(M, I), where M represents purely motion information and I is the provided portrait. More details about VAE and motion latent M can be found in the Appendix D.1. Now, we can focus on learning the motion generator conditioned on audio and textual instructions, i.e., M = G(A, T ).

<!-- page-break:5 -->

Fig. 2: Method Overview: The InstructAvatar consists of two components: VAE H to disentangle motion information from the video and a motion generator G to generate the motion latent conditioned on audio and instruction. As we have two types of data, two switches in instruction and audio are designed. During inference, the motion encoder in the VAE will be dropped and we iteratively denoise Gaussian noise to obtain the predicted motion latent. Together with the user-provided portrait, the resulting video is generated by the decoder of the VAE.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000001_a3c4c94704f021f4d477253b2d96eff34133de2333c4f6a7aed1d25da5298fc8.png)

In the following sections, we will detail how to obtain fluent, diverse, and fine-grained text instructions T , as well as how to design a diffusion model-based text-guided motion generator G.

## 3.2 Construct Natural and Diverse Text Instructions

To utilize natural language as the interactive interface, it is essential to construct a dataset containing text-expression pairs. However, existing emotional talking datasets [3, 39] typically provide only tag-level annotations for talking videos, offering limited emotion categories such as "happy" or "sad" along with their corresponding videos. To diversify and refine these labels into text instructions, we will employ three procedures: Emotion Label Extension, Action Unit Extraction, and MLLM Paraphrase, aiming to generate diverse, fine-grained and natural text instructions T, illustrated in Fig. 3.

Emotion Label Extension Firstly, we adopt a straightforward and convenient method to convert emotion labels into sentences by utilizing predefined templates. Specifically, we prompt ChatGPT to generate 60 templates, such as "Talk with [EMO] emotion". We then randomly select a synonym from a predefined table and then substitute the placeholder [EMO] in the template, producing expressions like "Talk with delighted emotion". Additionally, we also utilize the emotion intensity information of the dataset in our training. For instance, for videos with high emotion intensity, we may add adverbs like "extremely" to modify emotions, resulting in "extremely delighted".

<!-- page-break:6 -->

Fig. 3: We extend the emotion label using a predefined template and incorporate intensity information by modifying the emotion with an adverb representing the degree. For fine-grained control, we extract the AUs and then prompt GPT-4V to paraphrase them into a sentence.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000002_2100ee0b1cbb366137b5a20a331347ce6bfd2b7a6c98d25a5b56f02b2a78d072.png)

Action Unit Extraction The previous method for emotion label-based extension tends to provide coarse and high-level annotations. To enable fine-grained control of facial expressions, we turn to the Facial Action Coding System (FACS) [7]. This system defines Action Units (AUs) to describe facial muscle movements, allowing for a detailed description of the local states of a talking face as shown in Fig. 3. However, AU detection is typically performed on images. Therefore, we randomly select three frames from a video and employ an off-the-shelf AU detection model [23] to extract AUs from these images. We then take the intersection of predicted action units, considering that significant facial states corresponding to a specific emotion are likely to be consistent throughout the entire video.

MLLM Paraphrase The action units obtained in the previous step are represented in a relatively incomprehensible form, such as "lid\_tightner". To transform them into more human-friendly text instructions, we capitalize on large language models' powerful paraphrase capability. We prompt GPT-4V [27] to combine these action units into sentences. Additionally, leveraging its vision capabilities, we provide GPT-4V with a frame extracted from the video and allow it to edit the action units if it disagrees with the action units extracted by the off-the-shelf model. We ask GPT-4V to provide a few diverse sentences and randomly select one during training. More analysis on the reliability of GPT-4V can be found in Appendix A.6.

<!-- page-break:7 -->

## 3.3 Text-Guided Motion Generator

We leverage the diffusion model [16], a powerful generative model, as our textguided motion generator to learn M = G(A, T) mentioned in Sec. 3.1. In the InstructAvatar, we use Conformer [11] as our diffusion model backbone. The details of our motion generator is illustrated in Fig. 4.

Basics for Diffusion Models The diffusion model is designed to fit a distribution. Basically, it is divided into two phases: the forward diffusion process and the backward denoising process. Given a data point sampled from a real data distribution x0 ∼ q(x), we define a forward process in which Gaussian noise is incrementally added to the sample, generating a sequence of noisy samples x1, ..., xT . The noise scales are controlled by βt ∈ (0, 1), and the density is expressed as q(xt|xt-1) = N(xt; √1-βtxt-1, βtI). Based on the reparameterization trick [16], we can sample at any arbitrary time step in a closed form: q(xt|x0) = N(xt; √ ¯αtx0, √1 - ¯αtI), where αt = 1 - βt and ¯αt = ∏ti=1 αi. Furthermore, from this equation, it becomes evident that as T →∞, xT converges to an isotropic Gaussian distribution.

For the reverse process, in DDPM [16], it derives a simple learning objective Lsimple = ∑Tt=1 Eq [||ϵt(xt, x0)-ϵθ(xt, t)||2 ], where ϵt is the noise added in original data x0 and ϵθ is learnable network. Recently, researchers tend to use an even simpler strategy to train a network to predict x0 directly, with the loss function defined as L = ||x0 - fθ(xt, t)||, which is also applied in our framework. During inference, following DDIM [32], we start from a Gaussian noise and iteratively denoise it to get a predicted ˆx0.

Audio-aware Input Block Basically, following the classical strategy of diffusion models mentioned above, we will train the denoising block to recover noised motion latent Mt to predicted ˆM0, denoted as ˆM0 = f(Mt, t,A, T).

To incorporate audio information into the denoising process, we first normalize the speech to an appropriate amplitude range and then apply a denoiser [6] to reduce background noise. Subsequently, we utilize Wave2Vec 2.0 [1] as audio encoder W to extract audio features. As a special case, for facial motion control absent of audio, we use pseudo empty audio with zero amplitude and a length aligned with the ground truth video. Given that audio and motion sequences are aligned in the temporal semantic, we opt to element-wise add the audio features A to the motion latent vector M. In summary, we obtain the audio-aware noisy latent MAt by:

<!-- formula-not-decoded -->

<!-- page-break:8 -->

Fig. 4: Details of motion generator. To denoise noisy motion latent, firstly we elementwise add the audio feature into it. Then, in each denoising block, we design a two-branch cross-attention module to inject emotion and motion control into the model. Lastly, we also incorporate AU and intensity losses to encourage the model to learn them.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000003_d1b88ff1ea9594a445f5baa46a781e53c5c3a61f4a18ddf6bdb3936ae8647d13.png)

Two-branch Text-aware Denoising Block Now we would use cascaded denoising blocks to denoise M At to the data ˆM0. A key component of our architecture is injecting text instruction information into the denoising procedure. To encode the text instructions T obtained in Sec. 3.2, we leverage the CLIP [29] text encoder C, which has been proven for its powerful cross-modality alignment ability and strong semantic generalization ability. We employ a cross-attention mechanism to incorporate text information, where the hidden states in the Conformer layer act as queries, and the text representation serves as keys and values.

There exist some differences between emotion and motion controls. For the emotion, the text provides style guidance throughout the entire video, ensuring that the avatar maintains the desired emotion consistently. However, this is not the case for facial motion instructions, which usually describe gradually achieved actions and transitions over time. For example, a person may gradually turn his head when receiving corresponding motion instructions. In response, we split the data flow into two branches in each denoising block when incorporating text information. For the emotion branch, we use the [EOS] token from the CLIP text encoder, which encapsulates overall information about the instruction. For the motion branch, we utilize the hidden states of all tokens from the last layer of the CLIP text encoder to capture more detailed and dynamic information. Additionally, we introduce different Adapters [9] Ae,Am to better align the distributions of these two spaces with the space that facilitates the diffusion model's learning process. Overall, the instruction representation Rep(T ) to inject into the denoising procedure for text-guided emotional talking and text-guided facial motion control could be summarized as:

<!-- page-break:9 -->

<!-- formula-not-decoded -->

In addition, to provide the generator with facial shape information, following [14], we randomly select a frame from the video's motion latent M0 as the key frame latent Mk and inject it into the denoising process through cross-attention. It's noteworthy that for emotional talking, this selected key frame may inadvertently leak emotional information. Therefore, we substitute this frame with a frame from another emotional type video featuring the same person.

Zero Convolution Mechanism for Text Condition To leverage the abundant knowledge obtained from previous talking head models, we initialize our model from pretrained emotion-unaware models. Therefore, directly inserting text instructions would significantly reduce the expressiveness of the previously learned parameters. An insight is that an emotional talking video could transform from a neutral expression video gradually. Therefore, inspired by [51], we tailor a zero convolution mechanism for text conditioning.

Specifically, suppose we have hidden states hi-1 before entering the crossattention module with text instructions. We then use

<!-- formula-not-decoded -->

to get the next hidden states. In this formula, Z is a zero convolution operation where a 1-dimensional convolutional kernel moves along the hidden states dimension, with both the weight and bias initialized to zero. Therefore, at the start of training, hi = hi-1, which corresponds to the no instruction setting. Consequently, the zero convolution layer serves as a gate to slowly inject text control instructions into the pre-trained talking face model, stabilizing the training process and leveraging the abundant knowledge obtained in previous emotionunaware models.

## 3.4 Training and Inference Pipelines

Loss Definition After cascaded denoising blocks, we obtain the predicted motion latent ˆM0. The most intuitive loss is the distance between the predicted motion latent and the ground truth M0, expressed as Lmse = ∥ ˆM0 -M0∥22 . Additionally, to enforce the model to pay attention to action units and emotion intensity, we jointly train two classifier heads. We perform mean pooling in the temporal dimension followed by two-layer MLPs to extract information from ˆM0. Lau is calculated using the binary cross-entropy (BCE) loss, treating it as a multi-label classification problem over predicted AU logits ˆa ∈ RM and groundtruth labels a, given by: Lau = - 1N ∑Ni=1 ∥a ⊙ log(ˆa) + (1 - a) ⊙ log(1 - ˆa)∥1, where M is the number of action units and N is the sample number. For the emotion intensity loss, which is a three-classification problem, we use the standard cross-entropy loss Linten = - 1N ∑Ni=1 b log(ˆb), where ˆb ∈ R3 represents predicted logits and b represents the corresponding label. Moreover, following the approach in [14], we additionally train a head pose predictor and use another mean squared error (MSE) loss Lpose = ∥ ˆP -P∥22 to measure the predicted pose and the ground truth. In summary, our loss is defined as :

<!-- page-break:10 -->

<!-- formula-not-decoded -->

where λpose, λau, λinten are hyperparameters.

Inference Pipeline During inference, we begin by sampling a Gaussian noise MT to initialize the motion latent. The audio and instructions are provided by the user. We employ the VAE motion encoder [14] to encode the user-provided portrait I, resulting in the keyframe motion latent Mk. Subsequently, we iteratively denoise MT using our trained denoising network, following the DDIM [32]. Finally, we obtain the predicted motion latent ˆM, and utilizing the VAE decoder H, we generate the RGB video ˆ V =H( ˆM, I).

## 4 Experiments

We evaluate our model for both emotional talking control and facial motion control. For emotional talking control, the input includes audio and emotional guidance (such as labels, driving videos, and text, depending on the method). For facial motion control, we use only textual instructions to drive the avatar.

## 4.1 Experimental Setups

Datasets For emotional talking control, we augment the MEAD dataset [39] following the methods outlined in Sec. 3.2. MEAD is a large-scale emotional talking face dataset featuring 8 emotion types and 3 intensity levels. We reserved 5 individuals for testing purposes and utilized the remaining data for training. For text-guided facial motion control, we leveraged the CC v1 dataset [13], which offers paired data comprising instructions and corresponding action videos. To ensure effective lip synchronization, we also incorporated the HDTF dataset [53], which has high-quality talking face recordings. Our model was trained on a combination of these three datasets, enabling it to process either audio inputs or textual instructions. The evaluation was conducted using MEAD for in-domain assessment and TalkingHead 1KH [43] for out-of-domain evaluation. In the outof-domain setting, the appearance information will be sourced from TalkingHead 1KH, while a randomly selected sample from the MEAD test set will serve as emotion guidance. Tab. 8 in the Appendix C.1 summarizes the available modalities and the corresponding tasks for each dataset. Further details about dataset statistics and preprocessing methods are also provided in the Appendix C.

<!-- page-break:11 -->

Implementation Details As detailed in Sec. 3.3, we use Conformer [11] as the backbone of our diffusion-based motion generator. Specifically, the model comprises 12 Conformer blocks, with a hidden state size of 768. For encoding textual instructions, we apply CLIP-L/14 [29], and the Adapters are two layers MLPs. We adopt the Adam [19] optimizer and train our models on 8 V100 GPUs. The learning rate starts from 1e-5 and follows the inverse square root schedule. More hyperparameters and architecture details can be found in the Appendix D.

## Evaluation Metrics

Objective Metrics To assess the effectiveness of emotion and motion control as well as the quality of talking head generation, we employ various evaluation metrics from different aspects. To measure the fine-grained controlling ability, we propose AUF1, which calculates the F1 score of action units between the generated results and the ground truth. Furthermore, we introduce AUEmo, calculated as how many action units could be recalled by typical AUs of specific emotion types, to evaluate the overall coverage of facial details w.r.t an emotion type in the generated video. For motion control, we introduce the CLIPS metric, which computes the CLIP embedding similarity between the text instruction and each frame, with the maximum value indicating the correspondence between the instruction and generated motion. Moreover, SyncD is utilized to gauge lip-sync quality using SyncNet [5], and FID [15] for both emotion and motion control to evaluate overall quality. Detailed definitions for some metrics can be found in Appendix B.

Subjective Metrics We conduct the subjective evaluation involving 15 experienced users to score the generation quality and controllability of each model. Our evaluation metric is the Mean Opinion Score (MOS). For emotional talking control, we assess the lip-sync quality (Lip.), emotion controllability (Emo.), naturalness (Nat.), and motion jittering (Jit.). For facial motion control, we measured the accuracy of instruction following (Mot.) and identity preservation (ID.). Participants were presented with one video at a time and asked to rate each video for each score on a scale of 1 to 5. We calculated the average score as the final result.

## 4.2 Experimental Results

## Emotional Talking Control

Baselines We compare InstructAvatar with several state-of-the-art methods. We consider emotion-unaware models GAIA [14] and MakeItTalk [56], and provide the portrait and audio from the MEAD test set in inference. For the emotional label-based method EAT [8], we supplement the model with additional ground truth emotion types obtained in the annotation. Reference video-based methods like StyleTalk [25] and DreamTalk [26] utilize the ground truth emotional video in MEAD as style guidance. It is worth noting that video-based methods utilize more information in generation than ours.

<!-- page-break:12 -->

Table 1: Quantitative comparison with baselines for in-domain/out-of-the-domain settings. The bold values indicate the best results, while the underlined values represent the second-best. Guid. Mod. indicates the modality of emotional guidance. Since there is no ground truth video in the out-of-the-domain setting, the FID metric is left empty. It can be observed that our model outperforms the baselines across many metrics. Notably, for SyncD, the ground truth video has a SyncD of 9.172 in the in-domain setting, which is the closest to our model.

| Method                | AUF1↑       | AUEmo↑      | FID↓     | SyncD↓        | Guid. Mod.   |
|-----------------------|-------------|-------------|----------|---------------|--------------|
| GAIA [14]             | 0.549/0.185 | 0.352/0.048 | 52.716/- | 9.542/9.776   | -            |
| MakeItTalk [56]       | 0.588/0.220 | 0.405/0.065 | 47.269/- | 11.291/10.059 | -            |
| EAT [8]               | 0.648/0.542 | 0.495/0.319 | 57.379/- | 8.757/8.962   | Label        |
| StyleTalk [25]        | 0.694/0.499 | 0.593/0.278 | 75.783/- | 12.287/12.388 | Video        |
| DreamTalk [26]        | 0.711/0.513 | 0.548/0.301 | 85.291/- | 11.967/10.967 | Video        |
| InstructAvatar (Ours) | 0.738/0.552 | 0.566/0.324 | 44.593/- | 9.412/9.653   | Text         |

Table 2: Subjective evalution results for InstructAvatar and other baselines for indomain/out-of-the-domain settings. The bold values indicate the best results, while the underlined values represent the second-best.

| Method          | Emo.↑     | Lip.↑     | Jit.↑     | Nat.↑     | Guid. Mod.   |
|-----------------|-----------|-----------|-----------|-----------|--------------|
| GAIA [14]       | 3.02/1.83 | 4.63/4.51 | 4.57/4.55 | 4.31/4.47 | -            |
| MakeItTalk [56] | 3.88/2.36 | 3.49/3.62 | 3.99/3.68 | 3.82/3.46 | -            |
| EAT [8]         | 4.48/4.18 | 4.27/4.18 | 4.40/3.27 | 4.14/4.20 | Label        |
| StyleTalk [25]  | 4.53/4.12 | 4.17/2.26 | 3.43/1.88 | 3.18/2.04 | Video        |
| DreamTalk [26]  | 4.59/4.07 | 3.74/2.39 | 3.95/2.21 | 3.72/2.26 | Video        |
| (Ours)          | 4.64/4.52 | 4.74/4.59 | 4.88/4.68 | 4.63/4.60 | Text         |

Results As shown in Tab. 1, the proposed InstructAvatar model exhibits strong performance across most evaluation metrics, both in the in-domain and out-ofdomain settings. Notably, our model demonstrates excellent fine-grained control ability, as reflected by AUF1 scores. Moreover, in the in-domain setting, our SyncD metric is closer to the ground truth video (9.172) and exhibits better FID scores. It's worth mentioning that our model infers talking emotion solely based on text inputs, which intuitively poses a more challenging task. Additionally, our model supports a broader scope of instructions beyond high-level emotion types, which is absent for most baselines.

As a generative model, qualitative metrics may not fully capture the true effect. Therefore, we conducted subjective evaluation and case studies, as shown in Tab. 2 and Fig. 5, respectively. From these results, our model exhibits strong lip-sync ability and surpasses GAIA, a recently state-of-the-art audio-driven talking head model. Additionally, our model demonstrates superior emotion control- Happy lability compared to baselines tailored for emotional talking, such as EAT. More importantly, the generated results of our model appear more natural and robust to portrait images, as reflected by the Nat. metric and the cases shown in Fig. 5. More results are in Appendix A.1.

<!-- page-break:13 -->

Fig. 5: Qualitative comparison with baselines. It shows that InstructAvatar achieves well lip-sync quality and emotion controllability. Additionally, the outputs generated by our model exhibit enhanced naturalness and effectively preserve identity characteristics.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000004_05807158b6783429edf9427cb8986acc39e033390f769fc08e9934c3d978325b.png)

Facial Motion Control To evaluate the effectiveness of the motion controllability, we establish four evaluation settings: (i) Simply repeating the portrait image (Ref.), (ii) Given a random instruction (Rand. Inst.), (iii) Given true instruction (GT Inst.), and (iv) Ground-truth video (GT video). We set up (i) as a baseline to assess whether our model can generate dynamic video, and (ii) to evaluate whether our model can follow instructions. Results for different metrics in these settings are presented in Tab. 3, and visual results are provided in Fig. 6. Note that the GT Inst. setting represents the typical inference manner. Our model exhibits accurate instruction-following ability, as evidenced by similar CLIPS metrics with the ground-truth video and a large performance gap with Rand. Inst., along with high subjective motion accuracy (Mot.) scores. Moreover, our model demonstrates excellent video generation quality, producing natural portraits while maintaining identity, as indicated by the ID. metric and Fig. 6. Additionally, it generalizes effectively to connect different actions, as shown in Fig. 6. Please refer to Appendix A.2 for more results.

<!-- page-break:14 -->

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000005_7bd976bc183bf469ff8e4b2037a8e3064eaa54c02e0e379d3bb7ff36805a419f.png)

Ref. Image

Open your mouth and close it, and then turn your head left.

Fig. 6: Examples of text-guided facial motion control. Our model can execute precise motion control and generalize effectively to connect different actions.

Table 3: Objective and subjective metrics for text-guided facial motion control.

| Methods     | Obj. Met.   |        | Subj.   | Met.   |
|-------------|-------------|--------|---------|--------|
| Methods     | FID↓        | CLIPS↑ | Mot.↑   | ID.↑   |
| Ref.        | -           | 20.500 | 0       | -      |
| Rand. Inst. | 27.782      | 21.274 | 0.19    | 4.85   |
| GT Inst.    | 25.370      | 23.043 | 4.65    | 4.86   |
| GT Video    | -           | 23.136 | 5       | 5      |

## 4.3 Ablation Study

We conduct ablation studies to verify the effectiveness of each component of our model, as presented in Tab. 4. We utilize SyncD to reflect lip-sync quality, AUF1 to gauge emotion controllability, and Mot. to assess motion controllability.

We can see that (a) In terms of data format, when no action units are provided during training, the model loses its ability to capture fine-grained details, resulting in a decrease in AUF1. (b) Combining emotion learning and motion learning into a single branch leads to a contraction to some extent, negatively impacting the performance of both controls, as evidenced by the AUF1 and Mot. metrics. (c) The Zero Convolution mechanism, detailed in Sec. 3.3, is designed to stabilize training and harvest abundant knowledge from pretrained talking face models. Removing this component dramatically deteriorates lip-sync quality, and also influences emotion and motion control. (d) To prevent emotion leakage, we substitute the key frame latent with another emotion type during training emotional talking data. We found that this method significantly improves emotion control ability during out-of-domain tests, where portraits typically exhibit neutral emotion. More ablation studies can be found in the Appendix A.5.

## 5 Conclusion

In this paper, we introduce InstructAvatar, a novel text-guided unified framework for emotion and motion control in avatar generation, significantly enhancing controllability and vividness compared to previous models. We develop an automatic annotation pipeline to construct a fine-grained and diverse instructionvideo paired dataset and propose a two-branch diffusion-based generator, with one branch focusing on emotion and the other on motion, to achieve fine-grained and accurate controls. Experimental results demonstrate InstructAvatar's exceptional lip-sync quality, fine-grained emotion controllability, user-friendly control interface, and naturalness of the generated outputs. We hope our work will inspire further research into text-guided emotional talking heads and anticipate more studies in this area.

Table 4: Ablation studies on the proposed techniques.

| Methods - InstructAvatar   |   SyncD↓ AUF1↑ - 9.653 |   SyncD↓ AUF1↑ - 0.552 | Mot.↑ - 4.52   |
|----------------------------|------------------------|------------------------|----------------|
| (a) w/o AU                 |                  9.843 |                  0.435 | -              |
| (b) w/o Branch             |                  9.994 |                  0.488 | 4.10           |
| (c) w/o Zero conv.         |                 12.832 |                  0.434 | 3.62           |
| (d) w/o Rand. sub.         |                  9.597 |                  0.339 | -              |

<!-- page-break:15 -->

Limitations Our work still has limitations. For example, our model is trained solely on a combination of action units extracted from real talking videos. This dependency between action units may limit its ability to precisely control a disentangled single action unit. Additionally, the relatively modest size of our training dataset may hinder its robustness when faced with highly out-of-domain instructions or appearances. Moreover, since almost all data in our training dataset follows a single emotion/motion pattern, it is challenging for our model to control both emotion and motion simultaneously. We leave these challenges for future exploration.

## References

1. Baevski, A., Zhou, Y., Mohamed, A., Auli, M.: wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems 33, 12449-12460 (2020) 7, 26
2. Canfes, Z., Atasoy, M.F., Dirik, A., Yanardag, P.: Text and image guided 3d avatar generation and manipulation. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 4421-4431 (2023) 3
3. Cao, H., Cooper, D.G., Keutmann, M.K., Gur, R.C., Nenkova, A., Verma, R.: Crema-d: Crowd-sourced emotional multimodal actors dataset. IEEE transactions on affective computing 5(4), 377-390 (2014) 2, 5
4. Chu, X., Li, Y., Zeng, A., Yang, T., Lin, L., Liu, Y., Harada, T.: Gpavatar: Generalizable and precise head avatar from image(s) (2024) 2
5. Chung, J.S., Zisserman, A.: Out of time: automated lip sync in the wild. In: Computer Vision-ACCV 2016 Workshops: ACCV 2016 International Workshops, Taipei, Taiwan, November 20-24, 2016, Revised Selected Papers, Part II 13. pp. 251-263. Springer (2017) 11
6. Defossez, A., Synnaeve, G., Adi, Y.: Real time speech enhancement in the waveform domain. arXiv preprint arXiv:2006.12847 (2020) 7, 26
7. Ekman, P., Friesen, W.V.: Facial action coding system. Environmental Psychology &amp; Nonverbal Behavior (1978) 3, 6
8. Gan, Y., Yang, Z., Yue, X., Sun, L., Yang, Y.: Efficient emotional adaptation for audio-driven talking-head generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 22634-22645 (2023) 2, 4, 11, 12
9. Gao, P., Geng, S., Zhang, R., Ma, T., Fang, R., Zhang, Y., Li, H., Qiao, Y.: Clip-adapter: Better vision-language models with feature adapters. International Journal of Computer Vision 132(2), 581-595 (2024) 8, 22, 27

<!-- page-break:16 -->

10. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. Advances in neural information processing systems 27 (2014) 3
11. Gulati, A., Qin, J., Chiu, C.C., Parmar, N., Zhang, Y., Yu, J., Han, W., Wang, S., Zhang, Z., Wu, Y., et al.: Conformer: Convolution-augmented transformer for speech recognition. arXiv preprint arXiv:2005.08100 (2020) 7, 11, 27
12. Gururani, S., Mallya, A., Wang, T.C., Valle, R., Liu, M.Y.: Space: Speech-driven portrait animation with controllable expression. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 20914-20923 (2023) 2, 4
13. Hazirbas, C., Bitton, J., Dolhansky, B., Pan, J., Gordo, A., Ferrer, C.C.: Towards measuring fairness in ai: the casual conversations dataset. IEEE Transactions on Biometrics, Behavior, and Identity Science 4(3), 324-332 (2021) 10, 26
14. He, T., Guo, J., Yu, R., Wang, Y., Zhu, J., An, K., Li, L., Tan, X., Wang, C., Hu, H., et al.: Gaia: Zero-shot talking avatar generation. ICLR (2024) 2, 3, 4, 9, 10, 11, 12, 25, 27
15. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017) 11
16. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840-6851 (2020) 3, 7, 27
17. Huang, Z., Chan, K.C., Jiang, Y., Liu, Z.: Collaborative diffusion for multi-modal face generation and editing. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6080-6090 (2023) 3
18. Ji, X., Zhou, H., Wang, K., Wu, Q., Wu, W., Xu, F., Cao, X.: Eamm: One-shot emotional talking face via audio-based emotion-aware motion model. In: ACM SIGGRAPH 2022 Conference Proceedings. pp. 1-10 (2022) 2, 4
19. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014) 11, 27
20. Li, S.: Instruct-video2avatar: Video-to-avatar generation with instructions. arXiv preprint arXiv:2306.02903 (2023) 3
21. Liu, R., Ma, B., Zhang, W., Hu, Z., Fan, C., Lv, T., Ding, Y., Cheng, X.: Towards a simultaneous and granular identity-expression control in personalized face generation. arXiv preprint arXiv:2401.01207 (2024) 3
22. Liu, X., Wu, Q., Zhou, H., Du, Y., Wu, W., Lin, D., Liu, Z.: Audio-driven co-speech gesture video generation. Advances in Neural Information Processing Systems 35, 21386-21399 (2022) 4
23. Luo, C., Song, S., Xie, W., Shen, L., Gunes, H.: Learning multi-dimensional edge feature-based au relation graph for facial action unit recognition. In: Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence. IJCAI2022, International Joint Conferences on Artificial Intelligence Organization (Jul 2022). https://doi.org/10.24963/ijcai.2022/173, http://dx.doi.org/10. 24963/ijcai.2022/173 3, 6
24. Ma, Y., Wang, S., Ding, Y., Ma, B., Lv, T., Fan, C., Hu, Z., Deng, Z., Yu, X.: Talkclip: Talking head generation with text-guided expressive speaking styles. arXiv preprint arXiv:2304.00334 (2023) 4
25. Ma, Y., Wang, S., Hu, Z., Fan, C., Lv, T., Ding, Y., Deng, Z., Yu, X.: Styletalk: One-shot talking head generation with controllable speaking styles. arXiv preprint arXiv:2301.01081 (2023) 2, 4, 11, 12
26. Ma, Y., Zhang, S., Wang, J., Wang, X., Zhang, Y., Deng, Z.: Dreamtalk: When expressive talking head generation meets diffusion probabilistic models. arXiv preprint arXiv:2312.09767 (2023) 2, 4, 11, 12

<!-- page-break:17 -->

27. OpenAI: Gpt-4v(ision) system card (2023) 3, 6, 23, 26
28. Patashnik, O., Wu, Z., Shechtman, E., Cohen-Or, D., Lischinski, D.: Styleclip: Text-driven manipulation of stylegan imagery. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2085-2094 (2021) 3
29. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748-8763. PMLR (2021) 8, 11, 24, 27
30. Ren, Y., Li, G., Chen, Y., Li, T.H., Liu, S.: Pirenderer: Controllable portrait image generation via semantic neural rendering. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 13759-13768 (2021) 4
31. Siarohin, A., Lathuilière, S., Tulyakov, S., Ricci, E., Sebe, N.: First order motion model for image animation. Advances in neural information processing systems 32 (2019) 3
32. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020) 7, 10, 28
33. Sun, X., Zhang, L., Zhu, H., Zhang, P., Zhang, B., Ji, X., Zhou, K., Gao, D., Bo, L., Cao, X.: Vividtalk: One-shot audio-driven talking head generation based on 3d hybrid prior. arXiv preprint arXiv:2312.01841 (2023) 2
34. Tan, S., Ji, B., Pan, Y.: Emmn: Emotional motion memory network for audiodriven emotional talking face generation. pp. 22089-22099 (10 2023). https:// doi.org/10.1109/ICCV51070.2023.02024 2, 4
35. Tian, L., Wang, Q., Zhang, B., Bo, L.: Emo: Emote portrait alive - generating expressive portrait videos with audio2video diffusion model under weak conditions (2024) 2, 3
36. Tripathy, S., Kannala, J., Rahtu, E.: Facegan: Facial attribute controllable reenactment gan. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision. pp. 1329-1338 (2021) 3
37. Wang, D., Deng, Y., Yin, Z., Shum, H.Y., Wang, B.: Progressive disentangled representation learning for fine-grained controllable talking head synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 17979-17989 (2023) 2, 4
38. Wang, J., Zhao, K., Ma, Y., Zhang, S., Zhang, Y., Shen, Y., Zhao, D., Zhou, J.: Facecomposer: A unified model for versatile facial content creation. Advances in Neural Information Processing Systems 36 (2024) 3
39. Wang, K., Wu, Q., Song, L., Yang, Z., Wu, W., Qian, C., He, R., Qiao, Y., Loy, C.C.: Mead: A large-scale audio-visual dataset for emotional talking-face generation. In: European Conference on Computer Vision. pp. 700-717. Springer (2020) 2, 4, 5, 10
40. Wang, S., Li, L., Ding, Y., Fan, C., Yu, X.: Audio2head: Audio-driven one-shot talking-head generation with natural head motion. arXiv preprint arXiv:2107.09293 (2021) 2, 4
41. Wang, S., Li, L., Ding, Y., Yu, X.: One-shot talking face generation from singlespeaker audio-visual correlation learning. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 36, pp. 2531-2539 (2022) 2
42. Wang, T.C., Mallya, A., Liu, M.Y.: One-shot free-view neural talking-head synthesis for video conferencing. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2021) 3
43. Wang, T.C., Mallya, A., Liu, M.Y.: One-shot free-view neural talking-head synthesis for video conferencing. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10039-10049 (2021) 10

<!-- page-break:18 -->

44. Xia, W., Yang, Y., Xue, J.H., Wu, B.: Tedigan: Text-guided diverse face image generation and manipulation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2256-2265 (2021) 3
45. Xu, C., Zhu, J., Zhang, J., Han, Y., Chu, W., Tai, Y., Wang, C., Xie, Z., Liu, Y.: High-fidelity generalized emotional talking face generation with multi-modal emotion space learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6609-6619 (2023) 4
46. Yu, C., Lu, G., Zeng, Y., Sun, J., Liang, X., Li, H., Xu, Z., Xu, S., Zhang, W., Xu, H.: Towards high-fidelity text-guided 3d face generation and manipulation using only images. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15326-15337 (2023) 3
47. Zhai, S., Liu, M., Li, Y., Gao, Z., Zhu, L., Nie, L.: Talking face generation with audio-deduced emotional landmarks. IEEE Transactions on Neural Networks and Learning Systems (2023) 2, 4
48. Zhang, B., Qi, C., Zhang, P., Zhang, B., Wu, H., Chen, D., Chen, Q., Wang, Y., Wen, F.: Metaportrait: Identity-preserving talking head generation with fast personalized adaptation. arXiv:2212.08062 (2022) 3
49. Zhang, C., Wang, C., Zhang, J., Xu, H., Song, G., Xie, Y., Luo, L., Tian, Y., Guo, X., Feng, J.: Dream-talk: Diffusion-based realistic emotional audio-driven method for single image talking face generation. arXiv preprint arXiv:2312.13578 (2023) 2, 4
50. Zhang, L., Qiu, Q., Lin, H., Zhang, Q., Shi, C., Yang, W., Shi, Y., Yang, S., Xu, L., Yu, J.: Dreamface: Progressive generation of animatable 3d faces under text guidance. arXiv preprint arXiv:2304.03117 (2023) 3
51. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3836-3847 (2023) 9
52. Zhang, W., Cun, X., Wang, X., Zhang, Y., Shen, X., Guo, Y., Shan, Y., Wang, F.: Sadtalker: Learning realistic 3d motion coefficients for stylized audio-driven single image talking face animation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8652-8661 (2023) 2, 3, 4
53. Zhang, Z., Li, L., Ding, Y., Fan, C.: Flow-guided one-shot talking face generation with a high-resolution audio-visual dataset. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3661-3670 (2021) 4, 10, 26
54. Zhao, Q., Long, P., Zhang, Q., Qin, D., Liang, H., Zhang, L., Zhang, Y., Yu, J., Xu, L.: Media2face: Co-speech facial animation generation with multi-modality guidance. arXiv preprint arXiv:2401.15687 (2024) 4
55. Zhong, Y., Wei, H., Yang, P., Wang, Z.: Expclip: Bridging text and facial expressions via semantic alignment. arXiv preprint arXiv:2308.14448 (2023) 4
56. Zhou, Y., Han, X., Shechtman, E., Echevarria, J., Kalogerakis, E., Li, D.: Makelttalk: speaker-aware talking-head animation. ACM Transactions On Graphics (TOG) 39(6), 1-15 (2020) 2, 3, 4, 11, 12

<!-- page-break:19 -->

## A More Experimental Results

## A.1 More Results about Emotional Talking

Additional outcomes concerning text-guided emotional talking control are presented in Fig. 7. We can see that our model exhibits precise emotion control ability, with the generated results appearing natural. Furthermore, InstructAvatar supports fine-grained control and demonstrates reasonable generalization ability beyond the domain.

Ref. Image

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000006_a5525ca064c7cfd43702afda5f0052f0f5a141aaf20c44063289228ca0ab3e6a.png)

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000007_96cc3cc352f7af82ad794233704ef3d33c70f38a4f9ea08acab9f8a344763d9d.png)

Ref. Image

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000008_43c95ca32de622e8fd0213614a29bc6130ab32e2d56150f15b7ea49ef14d64a1.png)

Speak as if your emotion was astonished.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000009_2c9ff251c4ddb373741f786bbb8822cf9c6d2340b286df8184fc230adb003f51.png)

Express yourself with surprised emotion.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000010_54f6ffecad22ecf06a6ea4cbdbd070b5758dfd9db43d789c0b2418535b43eddc.png)

Ref. Image

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000011_c366d293616710caafe8a75337501868354c6dd8c7155b88374f05cc3f113829.png)

Ref. Image

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000012_0f2eb56f1b8e40f0164cd1637e25837680b174d7db4c47cb58604c6232045891.png)

Talk with happy emotion.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000013_f2950c48e0b997ba2f5c180e5c86b22b03dee49421781b17d8c414603a22a2d5.png)

Be angry when talking.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000014_2a662e30c28b8ab4abdd02703e3db100423737e684f0149a9d319c154ba112d1.png)

Ref. Image

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000015_df77b87c5711edc6bd8bbf3d6d495c8d9391ab8d603d3a69cf238f98499a06c5.png)

Ref. Image

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000016_655b50b4e8978f345c3a79147b9348692f9fc8563ce40d6776d9e5868cd4b79c.png)

Talk with a mixture of happy and surprised emotions.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000017_10dbab8d850a483c1c1e365d9f5f44b53f6eb8255c4aa2002ae71a687f62b4b4.png)

Talk with your lip corner depressed and brow lowered.

Fig. 7: More examples of text-guided emotional talking control.

<!-- page-break:20 -->

Ref. Image

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000018_9c18a43ada7de87d97a08d233f464fafb1ea37a9407b4b3e73dd63705988aa5d.png)

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000019_0c130191cb6b98b911545d4302a1e7a7d830ddb0bbfb96808abe3fa93d3000cc.png)

Slowly turn your head to your left side.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000020_f5975ee941b3abe41bd4c5bf8998cb21dec52c78a3da291d8a7b42cb670a4449.png)

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000021_a77cf12c8a38acaf600d0d8f5bc038736e1e17a5a653e703693048168c6df77d.png)

Ref. Image

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000022_b728afdd5131b5cd0910b4e8bdb7f81d08cc920c65a16fccc18b891c97e568f1.png)

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000023_4eaee97d479cf76b783fb411f864386c6725212a66a434c4a88340b40310239a.png)

Give me a surprised emotion.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000024_b325341a3648ad07c79ae67a53ad62aa26916ae8f357add69fbab0207c3bd11a.png)

Ref. Image

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000025_5c5816f62a6f1de4428bd972320122fd6c4eafc460e2435c7db33bb1195cd63b.png)

Smile.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000026_733f02568263c950a2c5d2f52ef00507c3107ba16d0b6c43f6c78458c56a4c7b.png)

Ref. Image

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000027_24e8b037b484278dcf57f8f70a20886f147255f807189360c00057a85a9dcf13.png)

Be neutral.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000028_2af5ab6d8c6060c5683ce716e644513b2ada2391d1901fbfd9365b9724567c48.png)

Ref. Image

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000029_cd64392a3cbeb2afaf6d513673af1cd3f45d33fd63881506188f28593df76278.png)

Lower your brow and depress your lip corner.

Ref. Image

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000030_014f4f7edfe36055ed161960666fce3ad920f045cb0344f1b91eee3ee8d6d2c4.png)

Ref. Image

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000031_12036b53634c65c7ff2b78df712636780cf11e3d7658da97ab3bf2d76a404272.png)

Sad.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000032_2f4e17d22c8438de9fcc5677fc710c0f4251ec0deb1c929c8c8b3ca3aab3ae7d.png)

Angry, please.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000033_2533858ae8bf565b68a089c32c508ba1c9cc00b75f64ca357bad4c1565b07b9d.png)

Ref. Image

Open your mouth.

Fig. 8: More examples of text-guided facial motion control.

## A.2 More Results about Facial Motion Control

We show more results about facial motion control in Fig. 8. It is evident that InstructAvatar exhibits remarkable proficiency in following instructions and preserving identity. Furthermore, the generated results appear natural and robust with variations in the provided portrait, including tilting or inherent expressions. Moreover, our model demonstrates fine-grained control capability and performs effectively in out-of-domain scenarios, as depicted in the last row of Fig. 8.

## A.3 The Effectiveness of Textual Instructions

To animate the avatar in our model, we input three conditions: portrait, audio, and textual instructions. We acknowledge that in real life, all these conditions can convey emotion. Therefore, a natural question arises: Does the emotion depicted in our generated videos primarily stem from the textual instructions rather than from the portraits or the inherent emotion conveyed in the audio? We address this question in Fig. 9, where all videos are generated using identical neutral portraits and neutral audio. The results demonstrate that our model can still produce distinct emotional talking videos, highlighting the effectiveness of textual instructions.

## A.4 Emotion Intensity

InstructAvatar demonstrates the capability to generate results with varying levels of emotion intensity. We illustrate a case in Fig. 10. It is evident that our model distinguishes between different emotion intensities based on specific descriptors such as "extremely" and "slightly".

<!-- page-break:21 -->

Ref. Image

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000034_140e8fcc0249a7268dfff6c7471a2b4f34409c84cb35bf5275e10a4f990c2c10.png)

happy

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000035_1413f74e194774e95cf40952e736dbb13cae82af583258fa7af2bc1a0166fc2d.png)

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000036_6bb04f32a1ed388948d7247d1f9d4a7005f023e29558991f198a5070d1c8d872.png)

surprised

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000037_4899e4a2788ecd5f6d125b2c52a9aebc6ee90ba555154c95e86da7486e8b46eb.png)

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000038_09800a845d9e9231c9fca20f8647f0a04005e394bd5fd022bf7f25c1c34927ea.png)

fear sad

contempt

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000039_81419724f2992a9dd86f7c13e7e12923eda63d02464634368d2d67514d90eaf0.png)

neutral

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000040_1a3a29b44822cece92c41df8f8e70263fc0d67cf5634ff22989bc81a435faef7.png)

Fig. 9: Illustration of the effectiveness of textual instructions. All videos are generated utilizing identical portraits and neutral audio, with variations only in the textual instructions.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000041_f42fdf0cdab2cb7cc45f445dc39edd58054261d79c8c356ad83b7ec198cc915e.png)

Ref. Image

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000042_6c241e1c286fcc0c4df53e7bda2eb66a428cf837198c19943069739ede3d58bd.png)

Talk with slightly sad emotion.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000043_504d980dbfce8b040f0fe1b85c22c7a3661e2aea8aeb526c39973813ca4c3226.png)

Talk with moderately sad emotion.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000044_ffed34e4865e9d956ea2ec19fc27013ba2f58f6c0fc3f8448c415ed07a629d2b.png)

Talk with extremely sad emotion.

Fig. 10: Illustration of emotion intensity control.

<!-- page-break:22 -->

Table 5: More ablation studies on the proposed techniques.

| Methods - InstructAvatar   | SyncD↓ AUF1↑ - 9.747   | SyncD↓ AUF1↑ - 0.537   | Mot.↑ - 4.46   |
|----------------------------|------------------------|------------------------|----------------|
| (a) w/o CLIP Adapter       | 9.903                  | 0.512                  | 4.31           |
| (b) w/o AU loss            | 9.753                  | 0.504                  | -              |
| (c) w/o Empty noise        | 10.427                 | 0.469                  | 3.23           |
| (d) w/ Label input         | 9.763                  | 0.469                  | -              |
| (e) All tokens for Emo.    | 9.842                  | 0.417                  | -              |
| (f) [EOS] token for Mot.   | -                      | -                      | 4.03           |

## A.5 More Ablation Results

We provide more ablation results in Tab. 5. To reduce the training cost, we randomly select 40% of the samples from the training pool. Given the substantial number of samples (over 60k), the statistical distribution differences would be marginal, thus representing the entire dataset. We observe that: (a) When the CLIP Adapter [9] is removed, the model loses a converter from the CLIP text space to the space required by the denoising block. This results in a decrease in both AUF1 and Mot. metrics. (b) When the AU loss is removed, the model loses some strict guidance on capturing fine-grained action unit details, leading to a decrease in AUF1. (c) To integrate facial motion control where no audio is provided into our unified framework, we use pseudo-empty audio as a placeholder. However, upon switching this strategy to employ another pseudo audio feature, such as tensors with all 0s, we observe a rapid deterioration in facial motion control, which also impacts emotional control, as indicated by Mot. and AUF1. We attribute this to the inherent physical meaning carried by pseudoempty audio, symbolizing silence and resulting in a still avatar. Conversely, the use of fake audio features like tensors with all 0s lacks meaningful interpretation. Consequently, when combined with normal audio during training, the model becomes confused due to this misaligned setting. (d) Beyond the limited emotion categories used in previous models, we adopt natural language for an openvocabulary emotion guidance approach. This method has several advantages as discussed in Sec. 1: (1) Enhanced control over fine-grained details rather than just the overall style; (2) Improved generalizability compared to limited emotion categories; and (3) Enhanced interactivity and user-friendliness. Experimentally, when training our model with emotion labels, as shown in Tab. 5 (d), the analysis of AUF1 indicates that text guidance offers better fine-grained controllability. (e) The [EOS] token is well-suited for the emotional talking task, acting as a general style guide. Replacing it with all instruction tokens leads to a performance drop, demonstrating that additional tokens may distract the model from extracting overall information. (f) Conversely, the hidden states of all tokens are more suitable for the motion control task, which requires temporally dynamic guidance. Replacing these with only [EOS] tokens fails to reflect dynamic information, resulting in suboptimal performance.

<!-- page-break:23 -->

## A.6 The Reliability of GPT-4V

In Sec. 3.2 in the main text, we mentioned that we would use GPT-4V [27] to transform the detected action units into coherent sentences and to reexamine the action units detected by off-the-shelf models, leveraging its visual capabilities. We provide an example wherein GPT-4V converts action units into a sentence, accurately rectifying erroneous action units and supplementing omitted ones, as depicted in Fig. 11. Furthermore, we randomly examined 20 examples from the dataset and observed that in 16 instances, there was no significant difference in the action units detected with or without leveraging visual capabilities. In the remaining 4 cases, we favored the results obtained by GPT-4V, and in none of these cases did GPT-4V generate an incorrect response.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000045_ca5f25a857054fe38af7357df221c030db065f14062bdf820e76862ef756d12a.png)

## AU detection

"outer\_brow\_raiser", "lid\_tightener", "upper\_lip\_raiser", "inner\_brow\_raiser", "lips\_part", "chin\_raiser", "lip\_corner\_puller"

## GPT-4V Paraphrase

Elevate the outer and inner brows, tighten the eyelids, raise the upper lip, part the lips, lift the cheek, drop the jaw, and pull the lip corners outward for a complex facial expression.

Fig. 11: Illustration of the effectiveness of GPT-4V paraphrase. We can see that GPT4V generates a fluent sentence while rectifying incorrect action units and supplementing omitted ones.

We also conduct a more extensive user study to evaluate the reliability of LLMs. We enlist 10 experienced volunteers to assess the outputs. For the 60 templates generated by ChatGPT, the volunteers found all to be satisfactory. For the paraphrased sentences generated by GPT-4V, we randomly select 30 instances (each with 3 sentences) and ask the volunteers to evaluate their accuracy, fluency, conciseness, diversity, and the correctness of overriding action units generated by the off-the-shelf model, rating them from 0 to 5. The results, shown in Tab. 6, indicate that LLMs not only help us quickly gather extensive training data but are also reliable.

## B Metrics Definition

We provide formal definitions for self-defined metrics in our paper.

<!-- page-break:24 -->

Table 6: Human evaluation for the application of GPT-4V on the data construction.

| Accuracy Fluency Conciseness Diversity Correctness of overriding   |   Accuracy Fluency Conciseness Diversity Correctness of overriding |   Accuracy Fluency Conciseness Diversity Correctness of overriding |   Accuracy Fluency Conciseness Diversity Correctness of overriding |   Accuracy Fluency Conciseness Diversity Correctness of overriding |   Accuracy Fluency Conciseness Diversity Correctness of overriding |
|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|
| Average Score                                                      |                                                                4.7 |                                                                4.9 |                                                                4.7 |                                                                4.5 |                                                                4.8 |

## B.1 AUF1 and AUEmo

To evaluate the fine-grained controllability of InstructAvatar, we introduce AUF1 and AUEmo. Let us consider action units extracted from generated sample j, denoted by ˆy(j) ∈ RM , alongside corresponding ground truth action units y(j) ∈ RM , where M represents the number of action units (in our paper M = 41). Both ˆy(j) and y(j) are vectors composed of 0s and 1s, with 1 indicating activation of the action unit and 0 indicating its inactivation.

To quantify the concordance between the action units of generated results and ground truth, assuming we have n samples, we compute the F1 score for this multi-label classification problem as follows:

<!-- formula-not-decoded -->

Moreover, to evaluate the overall coverage of facial details with respect to an emotion type, we define AUEmo. Firstly, we identify typical and representative action unit combinations for each emotion, as shown in Tab. 7. Then, for each generated result ˆy(j) , suppose the corresponding action units for such desired emotion are y(j) emo, we calculate how many action units could be recalled by the typical action units:

<!-- formula-not-decoded -->

## B.2 CLIPS

We employ CLIPS to measure the accuracy of text-guided motion control, as shown in Sec. 4.1 in the main text. Let t denote the text instruction and v denote the generated video. We denote vi as the i-th frame of video v. We use the CLIP [29] text encoder Et to encode the text and the image encoder Ev to encode each frame. Since the CLIP model is trained on paired image-text data, it possesses powerful modality alignment ability. We utilize cosine similarity to calculate the matchness of each frame with the instruction. Considering that motion control is a dynamic process, we consider it successful if it matches the instructions for a period of time. Therefore, we take the maximum of the similarity scores si as our final result:

<!-- formula-not-decoded -->

<!-- page-break:25 -->

Table 7: Typical Action Units for Different Emotions.

| Emotion Type   | Typical Action Units                                               |
|----------------|--------------------------------------------------------------------|
| Angry          | Brow Lowerer, Jaw Drop, Nose Wrinkler, Lid Tightener               |
| Fear           | Inner Brow Raiser, Jaw Drop, Upper Lid Raiser, Outer Brow Raiser   |
| Happy          | Cheek Raiser, Lip Corner Puller, Jaw Drop, Lid Tightener           |
| Contempt       | Inner Brow Raiser, Chin Raiser, Lip Corner Puller, Lid Tightener   |
| Disgusted      | Brow Lowerer, Cheek Raiser, Lip Corner Depressor, Nose Wrinkler    |
| Sad            | Brow Lowerer, Chin Raiser, Inner Brow Raiser, Lip Corner Depressor |
| Surprised      | Inner Brow Raiser, Jaw Drop, Outer Brow Raiser, Lid Tightener      |

## C Data

## C.1 Modalities of Each Dataset

In Tab. 8, We list the available modalities and corresponding tasks of each dataset utilized in our paper.

Table 8: Available modalities and corresponding tasks of the collected dataset. * indicates that for TalkingHead 1KH, we only use the portrait images to measure the out-of-domain generative ability of our model.

| Dataset         | Task                      | Video Audio   | (Speech)   | Instruction         |
|-----------------|---------------------------|---------------|------------|---------------------|
| MEAD HDTF       | Emotinoal Talking Control | ✓ ✓           | ✓ ✓        | ✓ ✓(Always neutral) |
| CC v1           | Facial Motion Control     | ✓             |            | ✓                   |
| TalkingHead 1KH | Evaluation*               | ✓             | ✓          |                     |

## C.2 Data Preprocessing

We preprocess all datasets and establish filter policies to discard low-quality samples.

For video data, we standardize each clip into a portrait-centered talking head video with dimensions of 256 by 256 pixels and a frame rate of 25 fps. Our filtering strategy follows [14] summarized as follows: (1) We maintain consistency in the orientation of individuals facing the camera throughout video clips.

<!-- page-break:26 -->

Frames exhibiting significant deviations, potentially obscuring lip movements, are excluded. (2) We monitor the positions of faces across frames, ensuring minimal displacement over consecutive timestamps to achieve smooth facial motion in video clips. (3) Frames featuring individuals wearing masks or remaining silent are identified and removed. Additionally, to minimize domain gaps across datasets, we estimate the distribution of talking head position and scale in the HDTF [53] videos and adjust the other datasets accordingly to this standard.

For each video clip obtained, we extract the audio and resample it to a 16kHz sampling rate. We normalize the speech and apply a denoiser [6] to reduce background noise. In the CC v1 dataset [13], where the audio comprises off-screen instructional speech unrelated to lip movement, we generate pseudo-empty audio with zero amplitude and a duration matching that of the corresponding ground truth video clip. Subsequently, we extract audio features using Wave2Vec 2.0 [1].

An important aspect of our model is its integration of textual information as a supervised signal. For emotional talking control, we outline the process of constructing textual instructions in Sec. 3.2 in the main text, along with listing the templates and prompts used in Appendix C.4. For facial motion control, the CC v1 dataset provides annotations consisting of off-screen instructional speeches obtained via ASR(Automatic Speech Recognition), along with corresponding timestamps. We extract the instructional annotations and prompt GPT-4V [27] to paraphrase them into fluent sentences, eliminating incomplete forms due to ASR detection. Based on these timestamps, we extract the corresponding action videos. Regarding the HDTF dataset, which lacks explicit instructions, and considering that the majority of videos in this dataset exhibit neutral emotions, we provide pseudo instructions such as "Talk with neutral emotion" or "Talk with an emotionless face". etc.

## C.3 Data Statistic

We provide the statistics of each dataset in Tab. 9.

Table 9: Statistics of the collected dataset.

| Datasets                  | Raw #IDs   |   #Hours | Filtered #IDs   |   #Hours |
|---------------------------|------------|----------|-----------------|----------|
| HDTF                      | 362        |       16 | 359             |       14 |
| CC v1 (Motion Control) 2, | 412        |       29 | 2, 412          |       23 |
| MEAD                      | 47         |       42 | 47              |       31 |
| Total                     | 2, 821     |       87 | 2, 818          |       68 |

<!-- page-break:27 -->

## C.4 Instruction Templates and GPT-4V Prompt

We present a portion of the templates utilized for transforming emotion types into sentences in Tab. 10, along with the prompts used to query GPT-4V in Tab. 11.

## D Implementation Details

## D.1 Illustration of VAE

In our model, we employ a Variational Autoencoder (VAE) based on the framework outlined in [14]. The VAE is designed to disentangle motion information from video data and consists of two encoders: the motion encoder and the appearance encoder, along with a single decoder.

To prevent the leakage of appearance information in reconstruction, they utilize the appearance information from the i-th frame and the motion information from the j-th frame to reconstruct the j-th frame by the VAE. Therefore, in cases where the i-th and j-th frames from one video clip contain the same appearance but different motion information (e.g., the same person speaking different words), the VAE model learns to first extract the pure appearance feature from the i-th frame. Subsequently, it combines this feature with the pure motion feature of the j-th frame to accurately reconstruct the original j-th frame.

Once the VAE is well trained, it can encode a video into disentangled appearance and motion latents. For a video with l frames, we can randomly select a frame for appearance encoding, resulting in an appearance latent of shape (1, dapp), where dapp is 768 for the GAIAbase model. This corresponds to the flattened dimensions (3, 16, 16), capturing spatial information. For the motion latent, we obtain a latent of shape (l, dmot) by encoding all frames with the motion encoder, where dmot is also 768.

## D.2 Hyperparameters and Model Architectures

The VAE comprises cascaded traditional convolutional residual blocks for both appearance and motion encoders, with downsampling factors of 8 and 16, respectively. The hidden size and number of layers are set to 256 and 4, resulting in approximately 700M parameters. The motion generator, based on diffusion models [16], consists of 12 Conformer [11] blocks with a hidden state size of 768. The total number of parameters for the motion generator is 409M, including around 100 million for the non-trainable CLIP-L/14 [29] text encoder and approximately 300 million for the denoising backbone. The adapter [9] utilizes a two-layer MLP with skip connections, where the hidden state size of the middle layer is 4 units smaller than the input dimension.

During training, the Adam [19] optimizer is employed with β1 = 0.9 and β2 = 0.98. The learning rate starts at 1e-5 and follows an inverse square root schedule with 8000 warmup steps. For the diffusion model, a quadratic β schedule is set with βmin = 0.05 and βmax = 20. During inference, the model follows the DDIM [32] approach and samples 150 steps. The loss weights are set to λpose = 1, λau = λinten = 0.1.

<!-- page-break:28 -->

## E Ethical Consideration

For ethical considerations, InstructAvatar is designed to advance AI research on talking avatar generation. Responsible usage is strongly encouraged, and we discourage users from employing our model to generate intentionally deceptive content or engage in other inauthentic activities. To prevent misuse, adding watermarks is a common approach. Moreover, as a generative model, our results can be utilized to construct artificial datasets and train discriminative models.

<!-- page-break:29 -->

```
========= Instruction Templates ========= "Talk with [EMO] emotion" , "Please talk with a [EMO] expression. " , "Please talk with a [EMO] face. " , "Please talk with a [EMO] look. " , "Please engage in a conversation using an [EMO] expression. " , "Converse with the help of [EMO] expression. " , "Make use of [EMO] expression as we discuss. " , "Could you express [EMO]ly in your speech?" , "speak with the expression of [EMO]" , "Please convey your thoughts with the emotion of [EMO]. " , "I'd like you to talk as if you were feeling [EMO]. " , "Incorporate [EMO] into your speech. " , "Let your words reflect the sentiment of [EMO]. " , "Let [EMO] be your guide in speaking. " , "Express yourself as if you were experiencing [EMO]. " , "Speak as if your emotions were [EMO]. " , "Try conveying your message with [EMO] as your tone. " , "Let [EMO] be your emotion as you speak. " , "Speak with the emotion of [EMO] guiding your words. " , "Let your words with the feeling of [EMO]. " , "I'd like you to communicate using [EMO] expression, please. " , "Could you chat employing [EMO] expression?" , "Discuss the topic while embracing the spirit of [EMO]. " , "Let's discuss things while incorporating [EMO] expression. " , "Feel free to use [EMO] expression while we talk. " , "Speak as if you were surrounded by [EMO]. " , "Let the essence of [EMO] guide your words. " , "Let [EMO] set the tone for your dialogue. " , "Could you communicate with the spirit of [EMO]?" , "Express your thoughts with the [EMO] in mind. " , "Let [EMO] be the guiding principle in your conversation. " , "Try conveying your message while being [EMO]. " , "Let [EMO] set the mood for your conversation. " , "Speak as if your emotions were [EMO]. " , "Let [EMO] shape the texture of your conversation. " , "Try conveying your message while being [EMO]. " , "Let [EMO] guides your dialogue. " , "Speak as if [EMO]. " , "Express your ideas with the emotional of [EMO]. " , ......
```

Table 10: The templates used in transforming emotion types into a sentence.

<!-- page-break:30 -->

Table 11: The prompts used to query GPT-4V.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_driven_generation/Instructavatar/images/image_000046_cf8388e75eb11f573ed0496250dc32f6017ffbe8d1fba441562a0dcb0cf41da0.png)