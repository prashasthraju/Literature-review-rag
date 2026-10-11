## Deepfake Video Detection Using Facial Feature Points and Ch-Transformer

RUI YANG, RUSHI LAN, ZHENRONG DENG, and XIAONAN LUO, The Guangxi Key

Laboratory of Image and Graphic Intelligent Processing, Guilin University of Electronic Technology,

Guilin, China

XIYAN SUN, The Nanning Research Institute, Guilin University of Electronic Technology,

Nanning, China

With the development of Metaverse technology, the avatar in Metaverse has faced serious security and privacy concerns. Analyzing facial features to distinguish between genuine and manipulated facial videos holds significant research importance for ensuring the authenticity of characters in the virtual world and for mitigating discrimination as well as preventing malicious use of facial data. To address this issue, the Facial Feature Points and Class-head-Transformer (FFP-ChT) deepfake video detection model is designed based on the clues of different FFPs distribution in real and fake videos and different displacement distances of real and fake FFPs between frames. The face video input is first detected by the BlazeFace model, and the face detection results are fed into the FaceMesh model to extract 468 FFPs. Then, the Lucas-Kanade (LK) optical flow method is used to track the points of the face, the face calibration algorithm is introduced to re-calibrate the FFPs, and the jitter displacement is calculated by tracking the FFPs between frames. Finally, the Ch is designed in the transformer, and the FFPs and FFP displacement are jointly classified through the ChT model. In this way, the designed ChT classifier is able to accurately and effectively identify deepfake videos. Experiments on open datasets clearly demonstrate the effectiveness and generalization capabilities of our approach.

CCS Concepts: · Security and privacy ! Privacy protections; Social network security and privacy; · Theory of computation ! Structured prediction;

Additional Key Words and Phrases: Transformer, deepfake detection, facial feature, FFP-ChT, facial feature points

This work was partially supported by the Guangxi Science and Technology Project (Grant Nos. AB22035052 and AD20159034), the National Natural Science Foundation of China (Grant Nos. 62172120, 62002082, and 6202780103), the Guangxi Key Laboratory of Image and Graphic Intelligent Processing Project (Grant No. GIIP2211), and the Innovation Project of GUET Graduate Education (Grant No. 2023YCXB09). Authors' Contact Information: Rui Yang, the Guangxi Key Laboratory of Image and Graphic Intelligent Processing, Guilin University of Electronic Technology, Guilin, China; e-mail: rainayoung@foxmail.com; Rushi Lan (Corresponding author), the Guangxi Key Laboratory of Image and Graphic Intelligent Processing, Guilin University of Electronic Technology, Guilin, China; e-mail: rslan2016@163.com; Zhenrong Deng, the Guangxi Key Laboratory of Image and Graphic Intelligent Processing, Guilin University of Electronic Technology, Guilin, China; e-mail: 799349175@qq.com; Xiaonan Luo, the Guangxi Key Laboratory of Image and Graphic Intelligent Processing, Guilin University of Electronic Technology, Guilin, China; e-mail: luoxn@guet.edu.cn; Xiyan Sun, the Nanning Research Institute, Guilin University of Electronic Technology, Nanning, China; e-mail: sunxiyan1@163.com.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

© 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM 1551-6865/2024/12-ART49

[https://doi.org/10.1145/3672566](https://doi.org/10.1145/3672566)

## ACM Reference format:

Rui Yang, Rushi Lan, Zhenrong Deng, Xiaonan Luo, and Xiyan Sun. 2024. Deepfake Video Detection Using Facial Feature Points and Ch-Transformer. ACM Trans. Multimedia Comput. Commun. Appl. 21, 2, Article 49 (December 2024), 22 pages.

[https://doi.org/10.1145/3672566](https://doi.org/10.1145/3672566)

## 1 Introduction

The rapid advancement of Artificial Intelligence (AI) technology has resulted in the widespread integration of Metaverse technology into various aspects of entertainment, daily life, and education. Within this Metaverse environment, avatars confront substantial security and privacy challenges. Due to the close correlation between facial features and identity information, it is crucial to establish effective privacy protection measures through the analysis of facial features. By detecting forged faces in the virtual world, it is possible to identify manipulated faces with malicious and discriminatory intent, aiming to prevent gender, race, ethnic, and other imbalances. Simultaneously, Deepfake (DF) detection can be utilized to authenticate avatars, preventing identity theft or the deceptive misuse of avatars. If an avatar is identified as fake or manipulated, appropriate actions can be taken to mitigate potential risks.

Currently, DF videos, generated through adversarial networks, demonstrate an unprecedented level of authenticity. In the Metaverse, these DF creations typically fall into three primary categories: audio, video, and image manipulations, with video manipulations being the most widely recognized. This technique involves recording an individual and seamlessly replacing their facial features with those of another person using pre-existing video footage. This intricate process heavily relies on deep learning algorithms, which effectively blend facial images and videos to produce counterfeit content that is often imperceptible to the naked eye. It's worth noting that current DF technology primarily relies on Generative Adversarial Networks (GANs) and its optimization models for its operation, representing a critical aspect of its functionality. ProGAN [10] and StyleGAN [16] are the two primary methods used to generate new synthetic faces. Fake videos are mainly divided into face replacement and expression exchange. Representative face replacement models include DeepFaceLab [34], FaceSwap (FS) [20], Face Swapping GAN [32], and DFs [19]. These models generate new synthetic videos by blending the image of a target person with existing video footage. DF, also known as facial reproduction, is currently popular with techniques, such as Face2Face (F2F) [44] and NeuralTextures (NTs) [43], where a facial expression from another person's face is replaced with the target's facial expression, such as changing the expression and movement to complete a fake "speech." The extensive application of DF technology, as evident from this, may pose significant security and privacy risks to virtual characters within the Metaverse.

Given the aforementioned background, DF videos can be utilized to steal the identities of virtual characters in the Metaverse. They can impersonate individuals for fraudulent activities, damage others' reputations, and initiate malicious cybersecurity attacks. This may ultimately lead to a crisis of trust in virtual world technology. Many experts and scholars have conducted research in this field. Currently, there are two main methods for detecting DF videos. The first method is based on analyzing single frames of the video to determine whether it is real or fake by examining details, such as hair, pupil, and facial features. The work presented in this article is inspired by Sun et al. [41], who demonstrated that the jitter of facial key points between real and fake video frames differs, as illustrated in Figure 1. Thus, a combination of Facial Feature Points (FFPs) distribution and displacement jitter is utilized to predict whether a video is real or fake.

We design a DF detection model FFPs and Class-head-Transformer (FFP-ChT) for the Metaverse, which focuses on identifying inconsistencies between the distribution of FFPs on a face and the jitter displacements in real and fake videos. Our model classifies videos based on the distribution of FFPs and the jitter displacements between those points in each video frame, thereby determining whether the face in the video is real or fake. FFP-ChT consists of three parts, first, a FFPs extraction model is used to extract 468 individual FFPs in each frame of the video. Secondly, the movement displacement of FFPs is calculated using the LK optical flow tracking method. Finally, the ChT is used to jointly predict FFPs and their jitter displacement in each video frame. Experimental results show that our DF detection model is robust and achieves higher Accuracy (ACC) not only on simple videos but also on higher resolution video datasets. The main contributions of our work are as follows:

Fig. 1. Jitter frequency of key points in different parts of real and fake faces. There are four broken lines in the figure, with the facial feature points of shaking taken from the eyes, nose, mouth, and outline. The green broken line represents the jitter frequency of the real FFPs, and the red broken line represents the jitter frequency of the fake FFPs.

![Image](parsed/deepfake%20detection%20using%20facial%20feature%20points/images/image_000000_b1f31fa6487f95b2ca3f3d0fd23952dcdfc3110368c6ef1387aa8c9fa7d0d58a.png)

- (1) We propose a new DF detection model for the Metaverse, FFP-ChT, based on transformer for joint prediction of FFPs and FFP displacements.
- (2) We engineer a 468 FFPs extraction model that fuses target detection neural network and FFP extraction neural network, focusing on facial features and contours with high shaking rates.
- (3) We propose a ChT model specifically designed to classify video frames with FFPs. ChT trains FFPs and their displacements separately and jointly predicts whether a video is real or fake based on the two results.

## 2 PriorWorks

As DF technology continues to evolve, the detection methods for DF videos are becoming increasingly sophisticated. Currently, DF video detection can be broadly categorized into two approaches. The first method involves analyzing individual frames of video to detect any discrepancies between real and fake faces [1, 8, 9, 31]. The second approach involves examining differences in timing characteristics and frame-by-frame variations to classify videos as real or fake.

## 2.1 Detection of DF Videos Based on Single-Frame Images

Similar to other AI tasks, the initial detection of fake face videos using single frame images has evolved from traditional methods to deep learning methods. Traditional methods include support vector machines [30], co-occurrence matrices [29], residual domain differences [21], and spectral differences [33] and have progressed to optical response heterogeneity and analysis of spatial domain features in images [37].

In the deep learning-based models, Jeon et al. introduced the Transferable GAN-images Detection framework, which applied transfer learning to GAN for the first time in generating image detection [14]. Further, Jeon proposed the fine-tuning neural network architecture, Fake Detection Fine-tuning Network, that incorporated a self-attention module to focus on the module and the undersampling layer. This method significantly enhanced the ACC [15]. Chen et al. found that both the luminance and chrominance components play important roles in detecting GAN-generated facial images. Therefore, considering both the luminance and chrominance components in the dual-color spaces can enhance the robustness of the model [5]. Hsu et al. utilized pair learning to detect deep forged images and transformed DenseNet into a two-flow network structure, which inputted paired information to train the common pseudo-feature network. The paired information then connected to the classification layer to detect the authenticity of the input image. It was noted that images generated by different GAN may have unknown image distortion, such as blur, noise, and JPEG compression [13]. Liu et al. found that large texture information is more robust to image distortion and proposed the GramNet model, which focuses on texture information as an essential clue for deep forgery image detection [27]. When new DF generation methods become available, many DF detection models' performance deteriorates. Thus, Khalid et al. proposed One-Class FakeDetect, which uses variational autoencoder to encode inputs as distributions in potential spaces [18]. Hinton et al. proposed a robust architecture composed of multiple capsules to solve the limitations of neural networks on inverse image problems. Capsule networks were applied to deep forgery detection [36]. Dual Spatial Pyramid for Exposing Face Warp Artifacts (DSP-FWA) model of Li et al. was limited by computational resources and production time, as the DF algorithm could only synthesize face images of fixed size and had to be affine distorted to match the face configuration of the source. This distortion can leave noticeable artifacts due to the inconsistent resolution between the distorted facial area and the surrounding environment. And real face videos and fake face videos can be classified by artifact analysis [23].

Liu et al. proposed a new block shuffling framework that consists of intra-block and inter-block transfer learning modules. The former improves the robustness of the network by intra-block transformation and overfits the problem by adversarial learning, while the latter encourages the network to learn local features through interblock shuffling and to model the semantic relevance of local features through refactoring [26]. The model high-frequency proposed by Luo et al. in 2021 improved the detection ACC by designing a multi-scale high-frequency feature extraction module, residual-guided spatial attention module, and cross-modal attention module [28]. Liu et al. proposed the Spatial-Phase Shallow Learning (SPSL) model, which captures upsampled artifacts of face forgery, improving the portability of face forgery detection [25]. Chen et al. proposed a Multi-Scale Patch Similarity Module (MPSM) to measure the similarity between local regional features and form a robust generalized similarity model [7]. They also proposed enhancing the "diversity" of forgery by synthesizing enhanced forgery with a pool of forgery configurations and enhancing the "sensitivity" to forgery by forcing models to predict forgery configurations [6]. Li et al. proposed the face X-ray model, which can classify true and false face videos by capturing the mixed boundary of forged images and the non-mixed boundary of real images. DF detection has typically been described as a classification problem based on cross-entropy loss, which emphasizes class-level differences rather than essential differences between real and fake faces, limiting the model's generalization in the invisible domain [22]. Sun et al. proposed a new face forgery detection framework called Dual Contrast Learning (DCL), which addresses the problem of local content inconsistency by constructing local region pairs in the instance. The issue of local content inconsistency is addressed by constructing local region pairs within the instance [39]. Gu et al. proposed the Hierarchical Contrastive Inconsistency Learning model, which can distinguish real faces from fake faces by deeply contrasting images [12]. In the 2022 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Shiohara proposed a novel composite training data called Self-Blended Images (SBIs) for detecting deep forgeries. The main idea behind SBIs is to enhance the robustness of the detection model by generating fake samples that are challenging to identify [38]. Chen et al. proposed a privacy-preserving model, Secure DeepFake Detection Network, which can detect DF faces without disclosing any private inputs, thus enhancing the privacy protection of facial data [4].

## 2.2 Detection of DF Videos Based on Spatio-Temporal Features of Videos

Compared to DF detection based on image detail features, there have been fewer studies on DF detection based on video timing features, and the research in this area is relatively new. In light of the spatio-temporal inconsistency present in generated forged videos, Zhang et al. proposed a time-domain separation three-dimensional convolutional neural network, Temporal Dropout 3DCNN, which utilizes the inconsistent cues between video frames. This method exhibits excellent generalization ability and ACC compared to popular benchmarks [46]. Zheng et al. proposed an end-to-end framework comprising a full-time convolutional network. They focused on making the network learning time-agnostic and proposed to limit the ability of the network to handle spatial correlation. By limiting the spatial dimension, the network's evolution concentrates on time dependence. This method achieves better results in dataset training without any pre-training or manual operation [48]. In 2022, Zhang et al. proposed a simple yet effective patch-level method, spatio-temporal dropout Transformer, to fully utilize the spatio-temporal inconsistencies present in local facial areas of different frames in DF videos. The input video is extracted into a facial frame sequence, which is then reassembled into a patch pack and introduced into the visual converter to learn the differential representation of dynamic spatio-temporal cues in local facial areas describing different frames [47]. In 2021, Sun et al. proposed a Recurrent Neural Network (RNN) classification algorithm model, LRNet, based on 68 human face landmarks, and introduced a new plug-and-play landmark calibration module to improve the ACC of facial geometric features and the effectiveness of time modeling [41]. Gu et al. studied the local motion of the image extensively and proposed a new sampling unit, Intra-Snippet Inconsistency Module, which uses bidirectional time difference operation and a learnable convolution kernel to mine short-term motion within each fragment. Their Snippet Inconsistency Module model achieved robust results in detecting DF videos [11]. In 2023, Sun proposed the FTDN model, which is based on facial areas such as displacement path to detect manipulated videos. This method can robustly identify displacement information [40].

## 3 Methodology

Our proposed DF video detection model FFP-ChT consists of four components as shown in Figure 2. Firstly, the input video is divided into frames, and a set of frames of the video image is fed into the BlazeFace model for face detection. Then, the face detection results are fed into the FaceMesh model to extract 468 FFPs. The extracted FFPs are repositioned by the feature point Re-Calibration algorithm. Finally, the FFPs and the displacement sequences of frame are fed into the ChT designed in this article for classification and prediction. The prediction results of FFPs and the prediction results of FFPs displacement sequences are combined to predict whether the input video is real or fake.

Fig. 2. General diagram of the FFP-ChT model. The input video frames are processed as a group and passed through BlazeFace for face detection and then through FaceMesh for the extraction of 468 FFPs. The extracted feature points are further calibrated by the FFP Re-Calibration algorithm, which computes a sequence of feature point displacements between frames. The FFP and FFP displacement sequences are separately fed into the ChT for training and the resulting predictions are combined for the final output. Conv, Convolutional layer; Conv2d, Two-Dimensional (2D) Convolutional layer; PReLU, Parametric Rectified Linear Unit.

![Image](parsed/deepfake%20detection%20using%20facial%20feature%20points/images/image_000001_80810b214838f1c8ffe6fd7c4ff2c0ae837e70964bec4fb933295704aaa4082c.png)

Fig. 3. Single BlazeBlock is shown on the left, and Double BlazeBlock is shown on the right.

![Image](parsed/deepfake%20detection%20using%20facial%20feature%20points/images/image_000002_853a499a4d6f1aeae6814059f534b218203a9e8890141faea3f499b63e85a4bc.png)

## 3.1 Face Detection and FFPs Extraction

To obtain FFPs with high ACC and relatively small computational cost, the face detection model adopts BlazeFace, a lightweight face detection model proposed by Bazarevsky et al. in 2019 [2]. The BlazeFace structure is shown in Figure 2, which mainly consists of one convolutional layer, five Single BlazeBlocks, and six Double BlazeBlocks. The Single BlazeBlock and Double BlazeBlock structures are shown in Figure 3.

Single BlazeBlock initially splits the input feature map into two branches for processing. The first branch undergoes a Depthwise (DW) convolutional layer with a 5 � 5 convolution kernel, followed by a convolutional layer with a 1 � 1 convolution kernel. The second branch is optional and involves parameter reduction by max pooling. The outputs of the last two branches are then added using an activation function. Double BlazeBlock builds upon the Single BlazeBlock structure by adding a 5 � 5 DW convolutional layer and a 1 � 1 convolutional layer on top of it. The face feature map detected by BlazeFace is fed into the FaceMesh key point extraction model, which was proposed by Google in the MediaPipe library in 2020 [17]. The structure diagram of FaceMesh is depicted in Figure 4.

Fig. 4. FaceMesh model. It is mainly extracted by backbone, coord\_head outputs the prediction result, conf\_head outputs the confidence of the result, and finally adds the joint prediction.

![Image](parsed/deepfake%20detection%20using%20facial%20feature%20points/images/image_000003_75259486f444608a8734adb660b103f32ab062d8c516f4a7be804029808d1bcc.png)

The backbone of the FaceMesh key point extraction model consists primarily of a TwoDimensional (2D) convolutional layer (Conv2d) and Parametric Rectified Linear Unit (PReLU) activation functions, and a FaceMesh Block structure. The training results are predicted through two heads, with the coord\_head directly outputting the prediction results and with the conf\_head outputting the confidence level. Finally, the two results are combined to produce the final FFPs prediction result. The FaceMesh Block shown on the right of Figure 4. The feature images are passed into the FaceMesh Block structure. If the step size is 2, the MaxPooling layer is adopted; if the step size is not 2, the feature images are expanded by padding, inputted into two 2D convolutional Conv2d layers, and finally outputted by the PReLU activation function.

The BlazeFace model is used to detect the entire face in each frame of the input video and to compute the face position in the image. The face landmark model is then employed to extract FFPs with a detection confidence threshold of 0.5. Each FFP is defined by its position coordinates "G� ~", and the facial contour is outlined by these FFPs, with the landmark points aligned to their corresponding preset positions via affine transformation, as illustrated in Figure 5. The 468 extracted FFPs are primarily located in the eyes, nose, mouth, and face regions. The nasal region features the highest density of FFPs in the 2D feature point distribution after the affine transformation. The model analysis depicted in Figure 5 confirms that the selected FFPs conform to a normal distribution.

## 3.2 FFP Re-Calibration Algorithm

To eliminate the influence of jitter noise on the displacement calculation of FFPs, we use the LucasKanade (LK) algorithm to track FFPs [3]. Then re-calibrate FFPs using the LK operation-based face calibration method proposed by Zekun Sun in 2021 [41]. The current frame is defined as %8 . The next frame is defined as %"8,1" . The core of LK algorithm is to track FFPs, which needs to minimize the pixel displacement 3 = »3G� 3~…) of each FFP in %8 and %"8,1" . The feature points of %8 are defined as G8 = »G�~…) , so the prediction of tracking points in %"8,1" is defined as G"8,1" = G8 , 3, where 8 � 480. The displacement vector is represented by

Fig. 5. FFPs extraction image. The figure on the left is the original video image, the figure in the middle is the drawing of the key parts of face extraction, and the image on the right is the objective display and distribution statistics of 468 FFPs.

![Image](parsed/deepfake%20detection%20using%20facial%20feature%20points/images/image_000004_47813025f30b835b0940c565a4a6a5e52cffd6a1978d234886d47eccb1aa1f36.png)

<!-- formula-not-decoded -->

where the pixel displacement 3 is initialized to 3 = »0� 0…) , and 3 is updated by iteration 3 "3 , �3" until converge. � represents the set of all small FFPs centered on G8 , 0G = exp � � kG�G8 k 2 2f 2 � , where exp is an exponential function based on the natural constant 4. 0G is used to reduce the weight away from the center position and make a smooth prediction, and �3 is achieved by

<!-- formula-not-decoded -->

where � = � ) �� 2 R 2�2 is Hessian matrix [42]. � 2 R�j�j�2 is generated by vertical connection. � "G" 2 R��2 "G 2 �" is Jacobian matrix [45]. � represents the number of channels for %8 , and � is a diagonal matrix whose elements are constituted by 0G . The Jacobian matrix is used to calculate the weighted eigenpoint G. �3 is updated until 3 converges, thus 3 predicting the FFPs G8,1 = G8 ,3 for %8,1 in the next 5 A0&lt;48,1.

The Kalman filter [41] designed in 2021 is used to fuse the detected FFPs and the predicted FFPs to complete the calibration of FFPs. The FFPs obtained through FFP extraction model is defined as - 34C 8,1 , the FFPs predicted by LK algorithm are defined as - ?A 43 8,1 , and the optimal FFPs obtained through the corresponding weighted average of the Kalman filter are defined as - &gt;?C 8,1 by Equation (3)

<!-- formula-not-decoded -->

where 8,1 is the Kalman gain for evaluating - &gt;?C 8,1 , which can be obtained by

<!-- formula-not-decoded -->

In Equation (4), %8,1 is the variance of - ?A 43 8,1 during LK prediction, �8,1 is the variance of - 34C 8,1 during FFP.

Fig. 6. Comparison between before and after calibration.

![Image](parsed/deepfake%20detection%20using%20facial%20feature%20points/images/image_000005_503198ae54e43bce2bfd43a00912ce8fc9d6436b6d8b5ccf92f2872eecd2f49c.png)

Fig. 7. FFPs and FFP displacements classification model: ChT.

![Image](parsed/deepfake%20detection%20using%20facial%20feature%20points/images/image_000006_d6e5e58c40cff366f8c2dcf15ca39d7e09f908bd1686a60e82c619961e437608.png)

As shown in Figure 6, calibrated FFPs mitigate the impact of facial boundary jitter, while a stable optical flow of these points reduces the likelihood of misclassification.

## 3.3 ChT

To comprehensively comprehend the characteristics of FFPs and their displacement, ChT is devised to train and categorize FFPs and FFP displacements in a set of video frames individually. The designed ChT model is shown in Figure 7.

A set of video frames are imported from a video. In this study, the block size is fixed at 60 and the facial landmarks of each frame are individually subjected to linear transformation. The classification head I 0 is designed to record the prediction information, and then the blocksize added to the classification head is encoded for position, and the encoding result is input into Transformer encoder. Finally, the classification result is output through the Ch.

3.3.1 FFPs Classification. Due to the excellent performance of Transformer in image classification, we propose a video frames-based FFPs classification model: ChT. One complete video is defined as + = �E 1� E 2� E 3� � � � � E 8 � � � � � E = � , where E 8 is the 8th frame of the video, and = is the total number of the video frames. In a video, the FFPs are defined as ! = �; 1� ; 2� ; 3� � � � � ; 8 � � � � � ; = � after the FFP calibration algorithm is used, where ; 8 is the corresponding FFPs coordinate of the 8th frame. Each frame contains 468 FFPs coordinates which are defined as ;8 = »"G 1�~ 1 "� "G 2�~ 2 "� "G 3�~ 3 "� � � � � "G 8 �~8 "� � � � � "G 468�~ 468 "…. # frames of FFPs are taken from the video as input for the ChT. Experiments in this article take # = 60 frames as a set of input&lt; = »; 9 � ; 9,1� � � � � ; 9,#�1 …, where 9 is the number of current frame, &lt; 2 R#�"468�2" .

A linear transformation is applied to the input &lt; to reduce the input dimension and speed up the computation, which can be obtained by

<!-- formula-not-decoded -->

� 2 R "468�2"�38&lt; , . is the result of linear transformation of FFPs displacement. Since only the Transformer encoder is used, the randomly initialized learnable vector ; 2;0BB 2R 1�38&lt; is concatenated in front of . for predicting the class of real and fake faces, which can be obtained by

<!-- formula-not-decoded -->

where / 2 R "#,1"�38&lt; is the sequence representation after adding ; 2;0BB . Position encoding is used for / to represent the temporal information of the video, which can be obtained by Equation (7)

<!-- formula-not-decoded -->

�?&gt;B = »%�0� %�1� %�2� � � � � %�C � � � � � %�# …, �?&gt;B 2 R "#,1"�"38&lt;" , I 0 = ; 2;0BB ,%�0. We use � as input to Transformer encoder, and the calculation of Transformer encoder is as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where �; = »I 0 ; � � � � � I # ; …. "(� represents the multi-head self-attention. !# is the layer normalization. ; 2 f1� � � � � !g, and ! is the number of encoder layers. Experiments in this article ! is set to 1. After multi-head self-attention calculation, I 0 ; becomes a vector which aggregated global feature information of �; . Due to the fact that I 0 ; does not contain video frame content itself during initialization, it can avoid bias in the prediction results of a certain video frame. And the position encoding of I 0 ; is fixed, avoiding interference from the position encoding on the output. Therefore, I 0 ; is used to express the classification representation of a set of video frames with # = 60. I 0 is obtained by the following equations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Algorithm 1: Facial Feature Points Displacement Classification Algorithm

```
INPUT<0 = »; 9 � ; 9,1� � � � � ; 9,#�1 … Linear transformation<0 by Equation (5) / add ; 2;0BB by Equation (6) Position encoding � /,�?>B by Equation (7) �?>B = »%�0� %�1� %�2� � � � � %�C � � � � � %�# … I 0 = ; 2;0BB , %�0 Transformer encoder I 0 ; by Equations (10) to (15) return I 0 ; Class Head%0BB8=6 I 0 ; to the Class head
```

OUTPUT B&gt;5 C&lt;0G

where *&lt;B0 2 R:����� are learnable parameters, � = 38&lt;. (� is self-attention layer, : is the total number of (� layers, we set = 8 in this article. The (� is calculated as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

�0 2 R 1�"#,1" is attention score of I 0 , @� :� E are transformed from �, which is defined as Equation (15)

<!-- formula-not-decoded -->

where *@�:�E 2 R��3�� , @ = �@0� � � � � @#,1 � .

As shown in Figure 7, after �; = »I 0 ; � � � � � I # ; … is obtained, I 0 ; is inputted into the designated classification header, Ch, to derive the probability of facial authenticity. The implementation of Ch is an layer normalization, which first normalizes vectors in 1,024 dimensions, followed by a linear transformation to obtain the probability of real and fake faces with dimension 2. Finally, B&gt; 5 C&lt;0G is used to predict the real or fake of FFPs in this group of frames.

3.3.2 FFPs Displacement Classification. The LK algorithm is used to track and calibrate the FFPs. The FFPs of frame %8 are defined as ;8 = »"G 1�~ 1 "� "G 2�~ 2 "� � � � � "G 8 �~8 "� � � � � "G 467�~ 467 "…, and the FFPs of frame %8,1 are defined as ;8,1 = »"G 2�~ 2 "� "G 3�~ 3 "� � � � � "G 8 �~8 "� � � � � "G 468�~ 468 "…. To categorize the displacement of FFPs, it is imperative to compute the displacement distance for each individual point across frames. The displacement distance is determined by calculating the difference between the coordinate points of the current and previous frames, as expressed in the following formula:

<!-- formula-not-decoded -->

Each frame contains the displacement of 468 FFPs, and a group of frames' FFP displacements in the video serve as input for the ChT model. In the displacement dataset B8 , &lt;0 represents the set of FFP displacements for # = 60 frames, and&lt;0 = »; 9 � ; 9,1� � � � � ; 9,#�1 …, where 9 is the current frame number, and&lt;0 2 R#�"468�2" . The classification calculation process is similar to FFPs, which is shown in Figure 7. The FFPs displacement classification algorithm is as Algorithm 1.

3.3.3 Classification of ChT. The FFPs in a video are defined as ! = »; 1� ; 2� ; 3� � � � � ; 8 � � � � � ; = …, the FFP displacements in a video are defined as ( = »B 1� B 2� B 3� � � � � B 8 � � � � � B = …, calculated by designing ChT classification separately. For the FFPs set !, the set of FFPs &lt; = »; 9 � ; 9,1� � � � � ; 9,#�1 … with # = 60 are taken. Each video ! includes multiple&lt; that is defined as ! = f&lt;1� � � � �&lt;8� � � � �&lt;=g and combined all &lt;8 2 ! for prediction by Equation (17)

<!-- formula-not-decoded -->

where ?! is the probability that the video ! predicted by FFPs is false. = is the number of &lt;8 in a video, and ?&lt;8 represents the probability that the prediction for sample&lt;8 is incorrect, which calculated using a softmax layer with # = 60 frames within the ChT model. Accordingly, the set of FFPs displacement&lt;0 = »; 9 � ; 9,1� � � � � ; 9,#�1 … are taken from the FFPs displacement set (. Each video ! also includes multiple &lt;0 which is defined as ! = �&lt;0 1� � � � �&lt;08 � � � � �&lt;0= � and combined all &lt;08 2 ! for prediction by Equation (18)

<!-- formula-not-decoded -->

where ?( is the probability that the video ! predicted by FFPs displacement is false, ?&lt;0 8 is the probability that the prediction of sample &lt;08 is false with # = 60 frames which is computed by B&gt; 5 C&lt;0G layer through ChT. Finally, the probability of detected video being a fake facial video is defined as Equation (19)

<!-- formula-not-decoded -->

The performance of the designed model is evaluated by the evaluation indexes ACC and AUC.

## 4 Experiments

## 4.1 Datasets

The Faceforensics,, (FF,,) dataset [35] consists of 1,000 raw videos downloaded from the internet, along with four corresponding versions of fake faces. The dataset comprises of videos with male and female subjects, with a ratio of 1 : 3 and 2 : 3, respectively. The resolution distribution is as follows: approximately 1/2 of the videos are in 480p, 3�8 in 720p, and 1�8 in 1080p. All videos contain a traceable, unobstructed front face that is used for automatic tampering to generate a real fake face. The videos were processed using one of four automatic face tamper methods, namely DFs, F2F, FS, NTs, and FaceShifter. As the primary aim is to assess the DF detection performance of the designed model, the DF subset is utilized for experimentation in this study.

Celeb-DF [24] is a recently proposed dataset with high-visual quality. The Celeb-DF dataset comprises of two versions: Celeb-DF(v1) and Celeb-DF(v2). In this experiment, Celeb-DF(v2) version was used. Celeb-DF(v1) contains 158 real videos and 795 fake videos, while Celeb-DF(v2) is a significant expansion over the v1 version, with 5,639 fake videos and 590 real videos based on publicly available YouTube clips from 59 celebrities of various genders, ages, and races. The average length of all videos is about 13 seconds, with a standard frame rate of 30 frames per second; 56.8% of the videos feature male subjects and 43.2% feature female subjects. The age distribution is as follows: 8.5% of the subjects are aged 60 or older, 30.5% are aged 50-60, 26.6% are in their 40s, 28.0% are in their 30s, and 6.4% are under 30. The racial distribution is as follows: 5.1% of the subjects are Asian, 6.8% are African American, and 88.1% are Caucasian. Additionally, the dataset features a wide range of variations in face size (in pixels), orientation, lighting conditions, and backgrounds. The fake face videos were generated by swapping each pair of faces of 59 subjects using an improved DF algorithm. As a result, the overall visual quality of the Celeb-DF dataset is significantly improved compared to the existing datasets.

## 4.2 Experimental Environment and EvaluationMetrics

4.2.1 Experimental Environment. The experiments presented in this article are conducted on the Ubuntu 18.04.5 LTS operating system, utilizing a GeForce RTX 2080 Ti graphics processing unit with a graphics memory capacity of 12 GB. The CUDA version 9.2 and PyTorch 1.7.1 framework are utilized. The FF,, dataset is divided into a training set and a validation set using an 8:2 ratio, similar to the basic model. For the CelebDF dataset, the division method provided by the original authors is adopted. Additionally, the parameters used for all datasets are the same. The specific experimental parameters are set as follows: batch size of 1.024, learning rate of 0.001, 400 epochs for training the detection network based on the absolute location of key points, and 300 epochs for training the detection network based on the relative location of FFPs. During training, each video is divided into fixed 60 frame segments as a training sample. The key point information of each frame is mapped to a 128 dimensional vector, and position coding is carried out for each 60 frame training sample with a droprate of 0.1 for the position coding layer. Then, a Learnable embedding vectors (cls\_token) is added, and the data are passed through a Transformer encoder with a dropout layer of droprate = 0.1 for encoding. The specific parameters for the Transformer encoder are set as follows: the encoder had 1 layer, 8 attention heads, 512 dimensions for each attention head, 128 for the last layer normalization, and 2 for the output of the linear transformation layer.

4.2.2 Evaluation Metrics. To assess the efficacy of the model in discerning authentic and counterfeit facial videos, the evaluation metrics ACC and Area Under the Curve (AUC) are used. ACC represents the proportion of correctly predicted samples out of the total samples, while AUC represents the probability that a positive sample score is greater than a negative sample score. The ACC metric is as follows:

<!-- formula-not-decoded -->

In Equation (20), True Positive (TP) indicates that the real category of the sample is positive, and the final predicted result is also positive. False Positive (FP) indicates that the real category of the sample is negative, but the final predicted result is positive. False Negative (FN) indicates that judgment is negative, but the judgment is wrong. True Negative (TN) indicates that the real category of the sample is positive, but the predicted result is negative at last.

Before calculating the AUC, we need to draw a ROC curve (receiver operating characteristic curve), also known as a sensitivity curve, and set a threshold that should be classified as 1 if it exceeds this threshold, and 0 if it is lower than this threshold. When the threshold starts at 0 and slowly moves to 1, we can get a set of values for False Positive Rate (FPR) and True Positive Rate (TPR), and we plot them on a coordinate system, that is called an ROC curve. FPR represents how many of the predictions are wrong among all the TNs, while TPR indicates how many of the predictions are correct among all the TPs. The formula is as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Table 1. Comparison of Experimental Results of Different Improvement Points of FFP-ChT

| Methods          |   DF(c23) - ACC | DF(c23) - AUC   |   Celeb-DF(v2) - ACC | Celeb-DF(v2) - AUC   |
|------------------|-----------------|-----------------|----------------------|----------------------|
| Baseline (LRNet) |           96.83 | 97.31           |                65.47 | 70.32                |
| ,468 FFPs        |           96.92 | 98.67           |                74.81 | 80.56                |
| ,ChT             |           97.56 | 98.32           |                73.26 | 71.13                |
| g1               |           92.75 | n               |                68.61 | n                    |
| g2               |           97.58 | n               |                74.00 | n                    |
| FFP-ChT (ours)   |           98.52 | 99.70           |                76.39 | 82.24                |

The following data are obtained in the experimental setup described in this article.

After the ROC curve is produced, the AUC can be calculated, and the value range of AUC is [0.5,1). The ROC curve must pass through two points (0,0) and (1,1). (0,0) indicates that all the forecasts are positive, and (1,1) indicates that all the forecasts are negative.

## 4.3 Ablation Experiment

In this section, we conduct an ablation experiment to evaluate the effectiveness of the proposed FFP-ChT. As a baseline, we choose the LRNet model [41], which is proposed using 68 FFPs for classification, while we propose to use 468 FFPs for classification and use ChT for joint prediction of FFP displacement. In Table 1, ",468 facial feature points" represents a model that extracts 468 FFPs instead of the original 68 points. ",Ch-Transformer" denotes the utilization of 68 feature points with the classification model being ChT. "g1" signifies a model in this article's FFP-ChT framework that solely employs the branch using 468 FFPs for prediction, while "g2" indicates prediction solely based on the branch using 468 FFPs Displacement.

As shown in Table 1, In the FF,,(DF) dataset (hereinafter referred to as DF), each video has an original version (raw), a slightly compressed version (c23), and a heavily compressed version (c40). The baseline ACC on DF(raw) reached 0.99, and our model's ACC also reached 0.99 (results on the DF(raw) dataset are shown in Table 2), with no significant difference in ACC. Therefore, ablation experiments are conducted on DF(c23) and Celeb-DF(v2) to demonstrate the effectiveness of the proposed innovations. On the DF(c23) dataset using ",468 facial feature points," ACC and AUC improved by 0.09 and 1.36 over the baseline, respectively, but the improvements were significant on Celeb-DF(v2), with ACC and AUC increasing by 9.34 and 10.24, respectively. After incorporating the designed ",Ch-Transformer," ACC and AUC increased by 0.73 and 1.01 over the baseline on the DF(c23) dataset, respectively, and by 7.79 and 0.81 on the Celeb-DF(v2) dataset, respectively. For the models proposed in this article, the ACC using only the g1 model with FFPs for prediction on both datasets is 92.75 and 68.61, respectively. The ACC using only the g2 model with FFP displacements for prediction is 97.58 and 74.00, respectively. The proposed FFP-ChT model demonstrated improvements over the baseline, with an increase of 1.69 in ACC and 2.39 in AUC on the DF(c23) dataset, and a remarkable increase of 10.92 in ACC and 11.92 in AUC on the Celeb-DF(v2) dataset.

The reason for the much higher improved ACC on the Celeb-DF(v2) dataset than on DF is that the portrait videos in the DF dataset are relatively simple, with most of them being portraits with a simple background, so both our model and baseline can achieve good ACC. However, Celeb-DF is a dataset of high visual quality, based on publicly available YouTube video clips, and is much The data for "LRNet" and "SBIs" are results obtained using their respective open-source models in the experimental environment described in this article, while data for other models are cited from their original articles. Bold and italic values represent the best and second-best results, respectively. PPG, Photoplethysmography; STDT, Spatiotemporal Dropout Transformer.

Table 2. A Comparison of the Results with Other Models on the DF(raw) Dataset

| Models         | ACC   | AUC   |
|----------------|-------|-------|
| DSP-FWA [23]   | n     | 64.6  |
| X-ray [22]     | n     | 99.1  |
| STDT [47]      | 97.9  | 99.7  |
| PPG cell [9]   | 93.6  | n     |
| Xception [8]   | n     | 99.7  |
| FTDN [40]      | 99.4  | 99.8  |
| LRNet [41]     | 99.2  | 99.9  |
| SBIs [38]      | 99.4  | 99.9  |
| FFP-ChT (ours) | 99.5  | 99.9  |

richer and more complex. The original classification of only 68 FFPs of the human face cannot represent the feature difference of the entire face. Therefore, using 468 FFPs of the human face intensively and extracting feature points from the four key parts of the eyes, nose, mouth, and face can expand the difference between true and false faces. For example, the stacking distance of key points displacement of the human face is higher. Furthermore, the ChT model proposed specifically for FFP video frames is also dedicated to face learning and training for each group of frames (60 frames as a group). Our model can learn the fine difference between true and false face videos better than the baseline classification with RNN. Hence, the ACC of the proposed model can be significantly improved on the Celeb-DF(v2) dataset.

## 4.4 Comparative Experiment

As facial video anti-counterfeiting technology advances, many experts and scholars have made significant research progress in DF detection. To verify the effectiveness of the model proposed in this article, experimental results were compared with different models in the DF and Celeb-DF(v2) datasets, using evaluation indices, such as ACC and AUC.

4.4.1 FF,,(DF) Dataset. Table 2 presents a comparison of the results with other models on the DF(raw) dataset. The DF(raw) dataset is uncompressed and the original version of the dataset. As shown in the Table 2, compared with recent DF detection models, our model achieves good detection results on DF(raw). When compared with the classical DF detection model "X-Ray" proposed by Microsoft Research Asia in CVPR 2020, our model achieves an increase of 0.8 percentage points in the AUC evaluation index and an increase of 0.1 percentage points compared with the "SBIs" ACC model proposed by Shiohara et al. in CVPR2022. The "FTDN" model was included in WACV2023, and our model achieves a 0.1 percentage point increase in ACC and AUC on the DF(raw) dataset.

To validate the detection performance of the model on compressed videos, experiments were conducted on the compressed version of the DF(c23). As shown in Table 3, Our model achieves the highest performance compared to recent models in terms of the ACC and AUC evaluation index. The FTDN model, also based on facial region displacement trajectory for DF detection, was published in WACV2023, and our model outperforms it by 0.45 percentage points in terms of ACC.

Table 3. A Comparison of the Proposed Method with Other Models on the DF(c23) Dataset

| Models              | ACC   | AUC   |
|---------------------|-------|-------|
| DSP-FWA [23]        | n     | 57.7  |
| X-ray [22]          | 98.1  | 87.3  |
| Xception [8]        | n     | 93.3  |
| DCL [39]            | n     | 99.6  |
| LRNet [41]          | 96.8  | 97.3  |
| High frequency [28] | n     | 99.2  |
| SBIs [38]           | 97.7  | 98.9  |
| FTDN [40]           | 98.0  | n     |
| FFP-ChT (ours)      | 98.5  | 99.7  |

The data for "LRNet" and "SBIs" are results obtained using their respective open-source models in the experimental environment described in this article, while data for other models are cited from their original articles. Bold and italic values in the table indicate the best and second-best values.

Table 4. A Comparison of the Proposed Method with Other Models on the DF(c40) Dataset

| Models               | ACC   | AUC   |
|----------------------|-------|-------|
| DSP-FWA [23]         | n     | 47.2  |
| Block shuffling [26] | n     | 93.13 |
| X-ray [22]           | n     | 61.6  |
| Xception [8]         | n     | 86.5  |
| MesoNet [1]          | 87.27 | n     |
| DCL [39]             | n     | 96.9  |
| LRNet [41]           | 92.4  | 95.7  |
| SBIs [38]            | 92.1  | 95.3  |
| FTDN [40]            | 83.55 | n     |
| FFP-ChT (ours)       | 93.7  | 96.8  |

The data for "LRNet" and "SBIs" are results obtained using their respective open-source models in the experimental environment described in this article, while data for other models are cited from their original articles. Bold and italic values in the table indicate the best and second-best values.

To verify the detection performance of the model on highly compressed video, experiments were carried out on DF(c40) compressed version. As shown in Table 4, compared to the network model FTDN in 2023, our model achieves approximately a 10-percentage-point higher ACC, demonstrating the robustness of the model in detecting forged facial videos at high compression rates.

Table 5. A Comparison of the Experimental Results with Other Models on the Celeb-DF(v2) Dataset

| Models               |   AUC |
|----------------------|-------|
| LRNet [41]           |  65.4 |
| Block shuffling [26] |  69.1 |
| X-ray [22]           |  74.7 |
| High-frequency [28]  |  79.4 |
| SPSL [25]            |  76.8 |
| MPSM [7]             |  78.2 |
| FFIW [6]             |  78.3 |
| DCL [39]             |  81.0 |
| SBIs [38]            |  81.9 |
| FFP-ChT (ours)       |  82.2 |

The data for "LRNet" and "SBIs" are results obtained using their respective open-source models in the experimental environment described in this article, while data for other models are cited from their original articles. Bold and italic values correspond to the best and second-best results.

- 4.4.2 Celeb-DF(v2) Dataset. It should be noted that the ACC evaluation metric is not an ideal metric when the sample distribution is imbalanced, as a high ACC rate may not guarantee effective detection. To illustrate, if there are 100 videos in total, 99 of which are fake and only 1 is real, any model that predicts fake videos for all samples would achieve an ACC rate of 99%. However, this high ACC rate would not indicate the model's ability to detect real videos. In the Celeb-DF(v2) dataset, there are 5639 fake videos and only 590 real videos, making the ACC evaluation index less reliable. Hence, we only use the AUC index to compare our model's experimental results on this dataset.

In the Table 5, our model obtains the highest AUC value in the Celeb-DF(v2) dataset, The second prize DCL model is a DCL for general face proposed by Sun et al. at the 2022 36th AAAI Conference on Artificial Intelligence Forgery Detection model. The AUC of our model in the CelebDF(v2) dataset is 1.2 higher. In summary, the proposed model in this article is more robust and can achieve high ACC rates not only in low-pixel videos but also in videos with more complex structural information.

- 4.4.3 Cross-Dataset Experiments. To validate the model's cross-dataset generalization ability, we separately fine-tuned the weights trained on DF dataset on the Celeb-DF(v2) dataset and tested on the DF dataset with weights trained on Celeb-DF(v2) dataset. The experimental results are shown in Table 6.

According to Table 6, training on Celeb-DF(v2) and testing on DF yields higher performance, while training on DF and testing on Celeb-DF(v2) results in lower performance. This is because the Celeb-DF(v2) dataset is more complex than the DF dataset. It is noteworthy that DF achieves an AUC of 89.7 when tested on Celeb-DF(v2), whereas training and testing on Celeb-DF(v2) itself yield an AUC of 82.2. It can be concluded that training the model on a more complex dataset leads to higher performance when tested on a simpler dataset.

Table 6. Cross-Dataset Experiments

| Training dataset   |   ACC |   AUC |
|--------------------|-------|-------|
| DF                 |  52.7 |  64.9 |
| Celeb-DF(v2)       |  84.6 |  89.7 |

Fig. 8. FFPs distribution. (a) The distribution density of FFPs in a frame of facial image. Darker blue squares indicate more FFPs. (b) The result of visualization of FFPs through quantile-quantile chart. The more points are in a straight line, the more normally distributed they are.

![Image](parsed/deepfake%20detection%20using%20facial%20feature%20points/images/image_000007_a0fda2459b0cca7f96d74045c625dfc6be4c628b82b98c12fca76e2edab979e5.png)

## 4.5 Model Analysis

The model in this article classifies real and fake faces by extracting 468 facial key points from each image frame. To compare the influence of different number of FFPs on the evaluation index. In each frame of a face video, 50 FFPs are extracted each time based on the 68 FFPs extracted, formula as shown in the following:

<!-- formula-not-decoded -->

where � 2 »0� 8…. %&gt;8=CB = f68� 118� 168� 218� 268� 318� 368� 418� 468g. Randomly selecting the corresponding number of FFPs from among 468 points, random probability function as follows:

<!-- formula-not-decoded -->

� "G� ?" represents the probability distribution function of random variable G� 5 "G; ?" represents the parameter % of the distribution function. Through the quantile-quantile plot verification of 468 FFPs, the FFPs meet the normal distribution as shown in Figure 8.

On the x-axis, we have the expected overall distribution quantile. On the y-axis, we have got the quantile of the sample empirical distribution. We conducted a normality test, but instead of testing the mean and variance graphically, we transformed the expected distribution into a standard normal distribution and tested whether the sample distribution falls on a line with a mean intercept and a standard deviation slope. As shown in the Figure 8, the 468 FFPs fall on a straight line, indicating that each frame face satisfies a normal distribution. We calculated the mean and variance of the FFPs using ' and f 2 , respectively. The random data generated by

Fig. 9. Visual comparison of different numbers of FFPs.

![Image](parsed/deepfake%20detection%20using%20facial%20feature%20points/images/image_000008_b2e345fe47f11903ca0654334a6e540d821cc66de7355a8c5a611e4cddc8deea.png)

Table 7. Comparison of Parameters

| Models       | Parameters   |
|--------------|--------------|
| DSP-FWA [23] | 28 M         |
| Xception [8] | 20.8 M       |
| SBIs [38]    | 67 M         |
| ChT          | 4.5 M        |

<!-- formula-not-decoded -->

To intuitively feel the distribution of different numbers of FFPs in the original video frame, Figure 9 shows the comparison between 68 FFPs and 468 FFPs.

The experimental results for the DF(c23) dataset and the Celeb-DF(v2) dataset were obtained by taking the FFPs 68, 118, 168, 218, 268, 318, 368, 418, and 468 using a random function. As shown in the Figure 10, both the ACC and AUC results increase as the number of FFPs increases, but the rate of increase slows down as the number of FFPs continues to increase. However, it is worth noting that the memory consumption also increases as the number of key points increases. For instance, the memory consumption of the 68 FFPs file in each video is 2.65 GB for Celeb-DF(v2) and 0.4 GB for FF,,. On the other hand, the memory consumption of the 468 FFPs file in each video is 18.3 GB for Celeb-DF(v2) and 7.5 GB for DF. After taking into account both model performance and ACC, we concluded that the optimal solution for both DF(c23) dataset and Celeb-DF(v2) dataset is 468 FFPs.

To demonstrate the lightweight nature of the designed ChT classifier, we compared its parameter count with that of other backbone networks, as shown in Table 7, The designed classifier has fewer parameters.

Fig. 10. Experimental results of different FFPs. On the left, we can see the ACC and AUC results of the model with different numbers of FFPs on the DF(c23) dataset, while on the right are the results on the Celeb-DF(v2) dataset. (a) DF(c23). (b) Celeb-DF(v2).

![Image](parsed/deepfake%20detection%20using%20facial%20feature%20points/images/image_000009_3ea42db98c4b8aa351c0091415a1784714dc01605c9beb726970fb275fab38dd.png)

## 5 Conclusion

In this article, to improve face privacy protection in the virtual world, we presented a new method for detecting DF videos that focuses on the inconsistency between the distribution of FFPs and their jitter displacements in real and fake videos. Our experimental results demonstrate that the proposed DF detection model has good robustness, as it achieves higher ACC in video datasets with higher resolution and obtains better detection results in simple videos. With the evolution of the Metaverse, in the future, DF detection can assist in establishing a series of identity verification standards in the virtual world to ensure the authenticity of users' identities. Simultaneously, it is possible to explore DF detection that combines various sensors and data sources to enhance the validation of real identities, thereby improving the capability to detect forged faces and reducing deception in virtual spaces.

## References

- [1] Darius Afchar, Vincent Nozick, Junichi Yamagishi, and Isao Echizen. 2018. Mesonet: A compact facial video forgery detection network. In Proceedings of the IEEE International Workshop on Information Forensics and Security (WIFS '18). IEEE, 1-7.
- [2] Valentin Bazarevsky, Yury Kartynnik, Andrey Vakunov, Karthik Raveendran, and Matthias Grundmann. 2019. BlazeFace: Sub-millisecond neural face detection on mobile gpus. arXiv:1907.05047. Retrieved from https://arxiv.org/pdf/1907.05047
- [3] Andrés Bruhn, JoachimWeickert, and Christoph Schnörr. 2005. Lucas/Kanade meets Horn/Schunck: Combining local and global optic flow methods. International Journal of Computer Vision 61, 211-231.
- [4] Beijing Chen, Xin Liu, Zhihua Xia, and Guoying Zhao. 2023. Privacy-preserving DeepFake face image detection. Digital Signal Processing 143, 104233.
- [5] Beijing Chen, Xin Liu, Yuhui Zheng, Guoying Zhao, and Yun-Qing Shi. 2021a. A robust GAN-generated face detection method based on dual-color spaces and an improved Xception. IEEE Transactions on Circuits and Systems for Video Technology 32, 6 (2021), 3527-3538.
- [6] Liang Chen, Yong Zhang, Yibing Song, Lingqiao Liu, and Jue Wang. 2022. Self-supervised learning of adversarial example: Towards good generalizations for deepfake detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18710-18719.
- [7] Shen Chen, Taiping Yao, Yang Chen, Shouhong Ding, Jilin Li, and Rongrong Ji. 2021b. Local relation learning for face forgery detection. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35. 1081-1088.
- [8] François Chollet. 2017. Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 1251-1258.

- [9] Umur Aybars Ciftci, Ilke Demir, and Lijun Yin. 2020. How do the hearts of deep fakes beat? Deep fake source detection via interpreting residuals with biological signals. In Proceedings of the IEEE International Joint Conference on Biometrics (IJCB '20). IEEE, 1-10.
- [10] Hongchang Gao, Jian Pei, and Heng Huang. 2019. Progan: Network embedding via proximity generative adversarial network. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 1308-1316.
- [11] Zhihao Gu, Yang Chen, Taiping Yao, Shouhong Ding, Jilin Li, and Lizhuang Ma. 2022a. Delving into the local: Dynamic inconsistency learning for deepfake video detection. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 36. 744-752.
- [12] Zhihao Gu, Taiping Yao, Yang Chen, Shouhong Ding, and Lizhuang Ma. 2022b. Hierarchical contrastive inconsistency learning for deepfake video detection. In Proceedings of the Computer Vision-ECCV 2022: 17th European Conference, Part XII. Springer, 596-613.
- [13] Chih-Chung Hsu, Yi-Xiu Zhuang, and Chia-Yen Lee. 2020. Deep fake image detection based on pairwise learning. Applied Sciences 10, 1 (2020), 370.
- [14] Hyeonseong Jeon, Youngoh Bang, Junyaup Kim, and Simon S. Woo. 2020b. T-gd: Transferable gan-generated images detection framework. arXiv:2008.04115. Retrieved from https://arxiv.org/abs/2008.04115
- [15] Hyeonseong Jeon, Youngoh Bang, and Simon S. Woo. 2020a. Fdftnet: Facing off fake images using fake detection fine-tuning network. In Proceedings of the ICT Systems Security and Privacy Protection: 35th IFIP TC 11 International Conference (SEC '20). Springer, 416-430.
- [16] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. 2020. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8110-8119.
- [17] Yury Kartynnik, Artsiom Ablavatski, Ivan Grishchenko, and Matthias Grundmann. 2019. Real-time facial surface geometry from monocular video on mobile GPUs. arXiv:1907.06724. Retrieved from https://arxiv.org/abs/1907.06724
- [18] Hasam Khalid and Simon S. Woo. 2020. Oc-fakedect: Classifying deepfakes using one-class variational autoencoder. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Workshops. 656-657.
- [19] Jan Kietzmann, Linda W. Lee, Ian P. McCarthy, and Tim C. Kietzmann. 2020. Deepfakes: Trick or treat? Business Horizons 63, 2 (2020), 135-146.
- [20] Iryna Korshunova, Wenzhe Shi, Joni Dambre, and Lucas Theis. 2017. Fast face-swap using convolutional neural networks. In Proceedings of the IEEE International Conference on Computer Vision. 3677-3685.
- [21] Haodong Li, Bin Li, Shunquan Tan, and Jiwu Huang. 2020b. Identification of deep network generated images using disparities in color components. Signal Processing 174, 107616.
- [22] Lingzhi Li, Jianmin Bao, Ting Zhang, Hao Yang, Dong Chen, Fang Wen, and Baining Guo. 2020a. Face x-ray for more general face forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5001-5010.
- [23] Yuezun Li and Siwei Lyu. 2018. Exposing deepfake videos by detecting face warping artifacts. arXiv:1811.00656. Retrieved from https://arxiv.org/abs/1811.00656
- [24] Yuezun Li, Xin Yang, Pu Sun, Honggang Qi, and Siwei Lyu. 2020c. Celeb-df: A large-scale challenging dataset for deepfake forensics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3207-3216.
- [25] Honggu Liu, Xiaodan Li, Wenbo Zhou, Yuefeng Chen, Yuan He, Hui Xue, Weiming Zhang, and Nenghai Yu. 2021. Spatial-phase shallow learning: Rethinking face forgery detection in frequency domain. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 772-781.
- [26] Sitong Liu, Zhichao Lian, Siqi Gu, and Liang Xiao. 2022. Block shuffling learning for deepfake detection. arXiv:2202.02819. Retrieved from https://arxiv.org/abs/2202.02819
- [27] Zhengzhe Liu, Xiaojuan Qi, and Philip H. S. Torr. 2020. Global texture enhancement for fake face detection in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8060-8069.
- [28] Yuchen Luo, Yong Zhang, Junchi Yan, and Wei Liu. 2021. Generalizing face forgery detection with high-frequency features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16317-16326.
- [29] Scott McCloskey and Michael Albright. 2018. Detecting GAN-generated imagery using color cues. arXiv:1812.08247. Retrieved from https://arxiv.org/abs/1812.08247
- [30] Lakshmanan Nataraj, Tajuddin Manhar Mohammed, Shivkumar Chandrasekaran, Arjuna Flenner, Jawadul H. Bappy, Amit K. Roy-Chowdhury, and B. S. Manjunath. 2019. Detecting GAN generated fake images using co-occurrence matrices. arXiv:1903.06836. Retrieved from https://arxiv.org/abs/1903.06836
- [31] Huy H. Nguyen, Junichi Yamagishi, and Isao Echizen. 2019. Capsule-forensics: Using capsule networks to detect forged images and videos. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP '19). IEEE, 2307-2311.

- [32] Yuval Nirkin, Yosi Keller, and Tal Hassner. 2019. FSGAN: Subject agnostic face swapping and reenactment. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7184-7193.
- [33] EhsanNowroozi and YassineMekdad. 2023. Detecting high-qualityGAN-generated face images using neural networks. In Big Data Analytics and Intelligent Systems for Cyber Threat Intelligence. River Publishers, 235-252.
- [34] Ivan Perov, Daiheng Gao, Nikolay Chervoniy, Kunlin Liu, Sugasa Marangonda, Chris Umé, Mr Dpfks, Carl Shift Facenheim, Luis R. P., Jian Jiang, Sheng Zhang, Pingyu Wu, Bo Zhou, and Weiming Zhang. 2020. DeepFaceLab: Integrated, flexible and extensible face-swapping framework. arXiv:2005.05535. Retrieved from https://arxiv.org/abs/2005.05535
- [35] Andreas Rossler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Nießner. 2019. Faceforensics,,: Learning to detect manipulated facial images. In Proceedings of the IEEE/CVF Conference on Computer Vision. 1-11.
- [36] Sara Sabour, Nicholas Frosst, and Geoffrey E. Hinton. 2017. Dynamic routing between capsules. Advances in Neural Information Processing Systems 30, 3856-3866.
- [37] Ulrich Scherhag, Luca Debiasi, Christian Rathgeb, Christoph Busch, and Andreas Uhl. 2019. Detection of face morphing attacks based on PRNU analysis. IEEE Transactions on Biometrics, Behavior, and Identity Science 1, 4 (2019), 302-317.
- [38] Kaede Shiohara and Toshihiko Yamasaki. 2022. Detecting deepfakes with self-blended images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18720-18729.
- [39] Ke Sun, Taiping Yao, Shen Chen, Shouhong Ding, Jilin Li, and Rongrong Ji. 2022. Dual contrastive learning for general face forgery detection. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 36. 2316-2324.
- [40] YuYang Sun, ZhiYong Zhang, Isao Echizen, Huy H. Nguyen, ChangZhen Qiu, and Lu Sun. 2023. Face forgery detection based on facial region displacement trajectory series. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 633-642.
- [41] Zekun Sun, Yujie Han, Zeyu Hua, Na Ruan, and Weijia Jia. 2021. Improving the efficiency and robustness of deepfakes detection through precise geometric features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3609-3618.
- [42] William Carlisle Thacker. 1989. The role of the Hessian matrix in fitting models to measurements. Journal of Geophysical Research: Oceans 94, C5 (1989), 6177-6196.
- [43] Justus Thies, Michael Zollhfer, and Matthias Niener. 2019. Deferred neural rendering: Image synthesis using neural textures. Acm Transactions on Graphics (TOG) 38, 4 (2019), 1-12.
- [44] JustusThies, Michael Zollhofer, Marc Stamminger, ChristianTheobalt, and Matthias Nießner. 2016. Face2face: Realtime face capture and reenactment of rgb videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2387-2395.
- [45] Yoji Umetani and Kazuya Yoshida. 1989. Resolved motion rate control of space manipulators with generalized Jacobian matrix. IEEE Transactions on Robotics and Automation 5, 3 (1989), 303-314.
- [46] Daichi Zhang, Chenyu Li, Fanzhao Lin, Dan Zeng, and Shiming Ge. 2021. Detecting deepfake videos with temporal dropout 3DCNN. In Proceedings of the International Joint Conference on Artificial Intelligence. 1288-1294.
- [47] Daichi Zhang, Fanzhao Lin, Yingying Hua, Pengju Wang, Dan Zeng, and Shiming Ge. 2022. Deepfake video detection with spatiotemporal dropout transformer. arXiv:2207.06612. Retrieved from https://arxiv.org/abs/2207.06612
- [48] Yinglin Zheng, Jianmin Bao, Dong Chen, Ming Zeng, and Fang Wen. 2021. Exploring temporal coherence for more general video face forgery detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 15044-15054.

Received 12 December 2023; revised 10 April 2024; accepted 25 May 2024