<!-- page-start:1 -->

## TEXT2AVATAR: TEXT TO 3D HUMAN AVATAR GENERATION WITH CODEBOOK-DRIVEN BODY CONTROLLABLE ATTRIBUTE

Chaoqun Gong1† , Yuqin Dai2† , Ronghui Li1 , Achun Bao1 , Jun Li2 , Jian Yang2 , Yachao Zhang1⋆ , Xiu Li1⋆

1Shenzhen International Graduate School, Tsinghua University, China

2School of Computer Science and Engineering, Nanjing University of Science and Technology, China

## ABSTRACT

Generating 3D human models directly from text helps reduce the cost and time of character modeling. However, achieving multi-attribute controllable and realistic 3D human avatar generation is still challenging due to feature coupling and the scarcity of realistic 3D human avatar datasets. To address these issues, we propose Text2Avatar, which can generate realistic-style 3D avatars based on the coupled text prompts. Text2Avatar leverages a discrete codebook as an intermediate feature to establish a connection between text and avatars, enabling the disentanglement of features. Furthermore, to alleviate the scarcity of realistic style 3D human avatar data, we utilize a pre-trained unconditional 3D human avatar generation model to obtain a large amount of 3D avatar pseudo data, which allows Text2Avatar to achieve realistic style generation. Experimental results demonstrate that our method can generate realistic 3D avatars from coupled textual data, which is challenging for other existing methods in this field.

Index Terms- 3D Avatar, Decoupling Control, Crossmodal Generation, Deep Learning

## 1. INTRODUCTION

3D human body modeling has wide-ranging application prospects in film production, video games, human-machine interaction, and content creation. Traditional 3D human body modeling is a complex and costly process, which can take thousands of hours to produce modeling products to meet requirements. Consequently, the utilization of text prompts in cross-modal 3D avatar generation frameworks has emerged as a practical and accessible modeling method with lower entry barriers.

There have been several instances [1, 2] which can generate reasonably matching 3D avatars using prompt words in recent years. To achieve better controllability, some research [3, 4] enables manipulating NeRF [5] using either a short text prompt or an exemplar image. TeCH [6] achieves 2D-to-3D human body reconstruction by using coupled text as assistance. However, research focused on generating 3D human bodies using prompt words solely is still scarce. Notably, to the best of our knowledge, due to the absence of realisticstyle 3D datasets and generation resolution limitations, all of the current text-to-3D avatar generators without additional information can only produce anime-style results.

† These authors contributed equally to this work.

⋆ Corresponding authors: yachaozhang@sz.tsinghua.edu.cn, li.xiu@sz.tsinghua.edu.cn

© 2024 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.

Moreover, it is difficult to decouple the generator's latent space, therefore simultaneously satisfying multiple human attributes in a single generated result is challenging. StyleFlow [7] enables decoupled face editing by modifying human face attributes through a reverse inference process. InterFaceGAN [8] achieves multi-attribute face control by altering the projection direction of the vectors in the subspace latent space. However, existing research has primarily focused on face editing, with limited work on human body decoupling editing due to the more complex spatial structure and the scarcity of datasets,

In this paper, we propose a novel framework, named Text2Avatar, which can generate 3D avatars from multiattribute prompts containing human clothing information. Different from the generation of objects [9], the generation of avatars focuses more on the rationality and controllability of body elements. Unlike [6], we only rely on textual prompts without the need for additional image inputs, thereby possessing higher levels of difficulty and a broader range of application prospects. To realize cross-modal generation, we proposed the Multi-Modal Encoder, which can be used as a plugin to assist in unconditionally generating models for textual cross-modal tasks. Inspired by prior works [10, 11, 12], we employ discrete attribute codes to express the 3D human body, realizing decoupled representation. By using the existing cross-modal model CLIP(Contrastive Language-Image Pre-Training) [13], which provides a paired semantic-consistent text-image encoder, we are able to encode text/image features into the discrete codebook. The codebook contains the human body feature and serves as a mediator to obtain the matching latent code, which controls the 3D avatar generation. To achieve high-accuracy attribute matching encoding, we employed a segmentation module [14] to support the CLIP model. In addition to the segmentation module, we utilized inherent image information(e.g., RGB) to facilitate the matching process.

<!-- page-break:2 -->

Fig. 1. Framework of Text2Avatar.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_to3dmodel_orfullbody/TEXT2AVATAR/images/image_000000_919d7679cbd80df1cc6aab38319aae33dd947900504e50e0647016c4fe07c237.png)

Fig. 2. Multi-Modal Encoder.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_to3dmodel_orfullbody/TEXT2AVATAR/images/image_000001_daf75fd7f5132c06c220b2b5dc6f74c98b4659bf01ec095ca93b0934fe7f94d7.png)

We extensively evaluate our proposed Text2Avatar and demonstrate that when presented with coupled textual prompts, our framework can generate high-quality 3D clothed avatars that satisfy complex attribute requirements.

## 2. METHOD

## 2.1. Structural Composition

Overall, our structure consists of three neural networks: the 3D-aware GAN generator G(·), segmentation module Seg(·) the multi-modal encoder E(·). The input of the generator G(·) includes camera parameters distribution ptext and the latent code z ∼ pz. The ptext includes the shape parameter β, the pose parameter θ, and the perspective attitude ξ. The multi-modal encoder E(·) consists of crossmodal text encoder Etext(·) and image encoder Etext(·), an attribute mapping network M(·), and a predefined text library{tpredefinedij |i = 1, ..., n, j = 1, ..., niattr , where n is the number of the attribute, and nattr,i represents the number of predefined categories for the i-th attribute. Our Framework is shown in Figure 1.

It should be emphasized that the text library is primarily designed to complement the encoders provided by CLIP [13] for attribute matching. Due to the strong data support of CLIP, this text library can theoretically be easily expanded without the need for additional training steps. The segmentation module [14] is employed to convert the local information of an image into global information in order to enhance the performance of the CLIP model. Motion generation methods [15] can be applied to the generation of β so that the avatar presents a variety of poses or dances.

## 2.2. Multi-Modal Encoder

The Multi-modal Encoder can serve as a plugin to assist unconditional generation models in textual cross-modal tasks. The Framework is shown as Figure 2. Given a 2D human body rendering segmented set {I i seg |i = 1, ..., n}, we utilize CLIP to extract the decoupled attribute feature. Specifically, by utilizing pre-trained image encoder Eimg(·) and text encoder Etext(·), which can encode visual and textual information into paired features, we identify the most relevant textual description for each I i seg within the given text library {tij predefined |i = 1, ..., n, j = 1, ..., niattr }. We encode both the textual and visual content, and compute their cosine similarity to select the best matching pair. The index of the tjpredefined is set to be the attribute value ai . The formula for discrete attribute encoding is as follows:

<!-- formula-not-decoded -->

The attribute mapping network will then utilize the aforementioned codebook to obtain the corresponding zgen, enabling control over unconditional generative models.

## 2.3. Training Setup

For the training of generators and discriminators, we follow the training methods of Hong et al [11].

We train an attribute mapping network based on MLP, which is mainly used to map the image human-attribute space to the latent space of the generative model. Unlike the video generation [16], avatar appearance is affected by the latent code z. Specifically, we first use the generative model to obtain paired latent variables zgen and the corresponding 3D avatar. We use volume rendering to obtain the corresponding image Igen. Then, utilizing the image encoder Eimg(·) provided by CLIP and the segmentation module [14], we encode the image Igen with a carefully-designed text library {tij predefined |i = 1, ..., n, j = 1, ..., niattr } into discrete codebook. The codebook corresponds to the attribute information with the corresponding number in the text library. After that, we use the mapping network to map the codebook to the latent space of the generative model, obtaining the predicted latent code zpre. We compare this zpre with the true zgen, and optimize the mapping network using MSE loss. The objective function L(ϕ) is shown below:

<!-- page-break:3 -->

Table 1. Comparison of different baselines w.r.t. attribute accuracy and R-Precision.

| Methods     |   Attribute Accuracy - Gender | Attribute Accuracy - Sleeve-length   | Attribute Accuracy - Top-color   | Attribute Accuracy - Top-type   | Attribute Accuracy - Pants-length   | Attribute Accuracy - Pants-color   | Attribute Accuracy - Pants-type   |   R-Precision - ViT-B/32 |   R-Precision - ViT-L/14 |
|-------------|-------------------------------|--------------------------------------|----------------------------------|---------------------------------|-------------------------------------|------------------------------------|-----------------------------------|--------------------------|--------------------------|
| DreamFusion |                          1.00 | -                                    | -                                | -                               | -                                   | -                                  | -                                 |                    74.71 |                    79.64 |
| 3DFuse      |                          1.00 | 0.30                                 | 0.65                             | 0.20                            | 0.55                                | 0.40                               | 0.15                              |                    77.83 |                    82.76 |
| AvatarCLIP  |                          1.00 | -                                    | 0.60                             | -                               | -                                   | 0.40                               | -                                 |                    76.66 |                    81.15 |
| Text2Avatar |                          1.00 | 1.00                                 | 0.80                             | 0.55                            | 0.85                                | 0.90                               | 0.60                              |                    78.52 |                    83.30 |

Fig. 3. Generation results from coupled textual prompts.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_to3dmodel_orfullbody/TEXT2AVATAR/images/image_000002_f1b1fde02f57fee9cc3060d38afe121e9b42d52e5ed00d4ec47f13c119776236.png)

<!-- formula-not-decoded -->

where ϕ represents the parameters of the mapping network, and N denotes the number of training instances.

## 2.4. Decoupling Control Generation

During the model inference phase for 3D avatar generation, the model takes as input text describing human attributes. The input text is first decoupled into various human attributes. These attributes are then fed into the multi-modal encoder that utilizes CLIP for matching with a text library to obtain human attribute code a. Then the attribute code are encoded into latent code z using the attribute mapping network M(·) trained during the training phase.

The latent code z, along with shape parameters β and pose parameters θ and camera viewpoint ξ, are input into a generator to produce an image of the avatar from a certain perspective during rendering. The mesh of avatar can be obtained by using the offsets inferred by a special NeRF within the generator.

## 3. EXPERIMENT

## 3.1. Implementation Details

We implemented code using pytorch on one NVIDIA RTX 3090 GPU. The training of the generator and discriminator follows the method of EVA3D [11]. They were trained on the deep fashion image dataset [17], along with the estimated SMPL [18] model parameters and camera perspective. The model was trained 400,000 iterations with a learning rate of 0.002 and a batch size of 64. We adopt the Adam optimizer.

In the GAN inversion step, we first used a trained generator to generate 50,000 images by random sampling of latent codes under the condition that the camera Angle of view was positive and the human body attitude parameters were fixed, and recorded the correspondence between the images and latent codes. Then, we perform decoupling image encoding according to the method mentioned in the section 2.2. The CLIP model used for attribute decoupling is ViT-L/14.

The predefined text library encompasses seven distinct attributes: gender, sleeve-length, top-color, top-type, pantslength, pants-color, and pants-type. The number of the attributes' category is decided by the segmentation performance. By merging the segmentation results of similar components such as arms and sleeves, seven distinct subgraph segments can be obtained.

## 3.2. Qualitative Results

Our generation result is shown as Figure 3. The results demonstrate that our approach can generate a 3D avatar matching the input text, incorporating various attributes of the human body. More vivid results can be found on the project page 1 . We compared our method with other text-to3D baselines as Figure 4 shows. Results demonstrate that existing methods struggle to deal with coupled cues, while our approach achieves better decoupling of high-quality generation. It can be observed that Dreamfusion [19] has a missing upper part of the body, while 3DFuse [20] has redundant human structures. Avatars generated by AvatarCLIP [1] and Text2Avatar both have complete human structures and exhibit the best performance. Comparing AvatarCLIP and Text2Avatar more carefully, it can be seen that Text2Avatar can generate more accurate results (green box) and higherresolution faces (red box) under coupled textual prompt.

1 project page: https://iecqgong.github.io/text2avatar/

<!-- page-break:4 -->

Fig. 4. Comparison between baselines and Text2Avatar.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_to3dmodel_orfullbody/TEXT2AVATAR/images/image_000003_159b4f00aa64e8e38bed5af7ff7f94c9b4bd1bf58d60e21d982b0f4167d2cdd9.png)

## 3.3. Quantitative Comparisons

We compare our method with existing text-to-3D methods[19, 20, 1]. We highlight that these existing text-driven crossmodal methods are limited in their capability to handle coupled instructions.

Attribute Accuracy. Due to the long time required and high cost involved of our baselines, we only allowed each model to generate 20 samples and manually compared their semantic matching accuracy as attribute accuracy. However, considering the obvious limitations of the comparative models in terms of visual results as Figure 4 shows, we believe that this number is sufficient to demonstrate the superiority of our model.

R-Precision. We employ volumetric rendering to converted 3D samples to 2D. We utilize the CLIP model [13] to calculate the correlation between the images and the textual features, and take the average value as the R-Precision. Considering that the samples generated by the model are labeled as "human", we include the match degree of the generated results with the word "human" in the calculation of R-Precision.

As shown in Table 1, we omitted the attribute accuracy of some models for difficult-to-measure attributes, which are unable to discern accurately due to low generation quality, or for which the model did not generate the corresponding body part information (e.g., Dreamfusion [19] did not generate the upper body of the avatar).

The results demonstrate that our model exhibits significant superiority in attribute accuracy across various attributes, while also achieving the best R-Precision.

Table 2. Ablation study result.w/o stands for without.

| Methods          |   Attribute Accuracy - Pants-color |   Attribute Accuracy - Sleeve-length |   R-Precision - ViT-B/32 |   R-Precision - ViT-L/14 |
|------------------|------------------------------------|--------------------------------------|--------------------------|--------------------------|
| w/o codebook     |                               0.55 |                                 1.00 |                    77.71 |                    82.65 |
| w/o segmentation |                               0.45 |                                 0.30 |                    76.64 |                    81.66 |
| origin           |                               0.80 |                                 1.00 |                    78.52 |                    83.30 |

## 3.4. Ablation Studies

To validate the effectiveness of our codebook design approach, we conducted ablative experiments by removing the codebook and segmentation module separately. In fact, the attributes that we define can be roughly divided into three categories: gender attribute, length attribute, and color attribute. As the accuracy of gender is consistently high, we have chosen the indicative length-related attribute (Sleeve-length) and the color-related attribute (Pants-color) to illustrate the superiority of our method, along with the R-Precision.

The experimental results are shown in Table 2, indicating that the segmentation operation and codebook significantly improve the recognition accuracy and R-precision. This is because segmentation converts local information of human attributes into global information, helping CLIP overcome the disadvantage of local information confusion, while codebook effectively increases the controllability between mappings.

## 4. CONCLUSIONS

We propose Text2Avatar, a method for generating realisticstyle 3D Avatars from coupled multi-attribute description text. We highlight that the Multi-Modal Encoder module can serve as a plugin after retraining, therefore providing flexibility. In this way, different clothing of the human body can be easily obtained from textual information solely.

## Acknowledgements

This research was partly supported by Shenzhen Key Laboratory of next generation interactive media innovative technology (Grant No: ZDSYS20210623092001004), the China Postdoctoral Science Foundation (No.2023M731957), the National Natural Science Foundation of China under Grant 62306165, 62072242 and 62361166670.

<!-- page-break:5 -->

## 5. REFERENCES

- [1] Fangzhou Hong, Mingyuan Zhang, Liang Pan, Zhongang Cai, Lei Yang, and Ziwei Liu, "Avatarclip: Zeroshot text-driven generation and animation of 3d avatars," arXiv preprint arXiv:2205.08535, 2022.
- [2] Yukang Cao, Yan-Pei Cao, Kai Han, Ying Shan, and Kwan-Yee K Wong, "Dreamavatar: Text-and-shape guided 3d human avatar generation via diffusion models," arXiv preprint arXiv:2304.00916, 2023.
- [3] Can Wang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao, "Clip-nerf: Text-and-image driven manipulation of neural radiance fields," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 3835-3844.
- [4] Nasir Mohammad Khalid, Tianhao Xie, Eugene Belilovsky, and Tiberiu Popa, "Clip-mesh: Generating textured meshes from text using pretrained image-text models," in SIGGRAPH Asia 2022 conference papers, 2022, pp. 1-8.
- [5] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng, "Nerf: Representing scenes as neural radiance fields for view synthesis," in ECCV, 2020.
- [6] Yangyi Huang, Hongwei Yi, Yuliang Xiu, Tingting Liao, Jiaxiang Tang, Deng Cai, and Justus Thies, "Tech: Text-guided reconstruction of lifelike clothed humans," arXiv preprint arXiv:2308.08545, 2023.
- [7] Rameen Abdal, Peihao Zhu, Niloy J Mitra, and Peter Wonka, "Styleflow: Attribute-conditioned exploration of stylegan-generated images using conditional continuous normalizing flows," ACM Transactions on Graphics (ToG), vol. 40, no. 3, pp. 1-21, 2021.
- [8] Yujun Shen, Jinjin Gu, Xiaoou Tang, and Bolei Zhou, "Interpreting the latent space of gans for semantic face editing," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 9243-9252.
- [9] Yukang Lin, Haonan Han, Chaoqun Gong, Zunnan Xu, Yachao Zhang, and Xiu Li, "Consistent123: One image to highly consistent 3d asset using case-aware diffusion priors," arXiv preprint arXiv:2309.17261, 2023.
- [10] Shengqu Cai, Anton Obukhov, Dengxin Dai, and Luc Van Gool, "Pix2nerf: Unsupervised conditional p-gan for single image to neural radiance fields translation," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 3981-3990.
- [11] Fangzhou Hong, Zhaoxi Chen, Yushi LAN, Liang Pan, and Ziwei Liu, "EVA3d: Compositional 3d human generation from 2d image collections," in International Conference on Learning Representations, 2023.
- [12] Yuming Jiang, Shuai Yang, Haonan Qiu, Wayne Wu, Chen Change Loy, and Ziwei Liu, "Text2human: Textdriven controllable human image generation," ACM Transactions on Graphics (TOG), vol. 41, no. 4, pp. 111, 2022.
- [13] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., "Learning transferable visual models from natural language supervision," in International conference on machine learning. PMLR, 2021, pp. 8748-8763.
- [14] Peike Li, Yunqiu Xu, YunchaoWei, and Yi Yang, "Selfcorrection for human parsing," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 6, pp. 3260-3271, 2020.
- [15] Ronghui Li, Junfan Zhao, Yachao Zhang, Mingyang Su, Zeping Ren, Han Zhang, Yansong Tang, and Xiu Li, "Finedance: A fine-grained choreography dataset for 3d full body dance generation," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 10234-10243.
- [16] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Ying Shan, Xiu Li, and Qifeng Chen, "Follow your pose: Pose-guided text-to-video generation using posefree videos," arXiv preprint arXiv:2304.01186, 2023.
- [17] Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang, "Deepfashion: Powering robust clothes recognition and retrieval with rich annotations," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.
- [18] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black, "Smpl: A skinned multi-person linear model," ACM transactions on graphics (TOG), vol. 34, no. 6, pp. 1-16, 2015.
- [19] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall, "Dreamfusion: Text-to-3d using 2d diffusion," arXiv preprint arXiv:2209.14988, 2022.
- [20] Junyoung Seo, Wooseok Jang, Min-Seop Kwak, Jaehoon Ko, Hyeonsu Kim, Junho Kim, Jin-Hwa Kim, Jiyoung Lee, and Seungryong Kim, "Let 2d diffusion model know 3d-consistency for robust text-to-3d generation," arXiv preprint arXiv:2303.07937, 2023.