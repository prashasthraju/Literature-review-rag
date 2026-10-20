<!-- page-start:1 -->

## 𝐸 3 Gen: Efficient, Expressive and Editable Avatars Generation

Weitian Zhang Shanghai Jiao Tong University Shanghai, China weitianzhang@sjtu.edu.cn Yichao Yan Shanghai Jiao Tong University Shanghai, China yanyichao@sjtu.edu.cn Yunhui Liu Lenovo Research Shanghai, China liuyhp@lenovo.com

Xingdong Sheng

Lenovo

Shanghai, China

shengxd1@lenovo.com Xiaokang Yang Shanghai Jiao Tong University Shanghai, China xkyang@sjtu.edu.cn

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_to3dmodel_orfullbody/e3%20gen%20avatars/images/image_000000_aafb6f35c886571814fc3ea6c1d0a656fff7051e3935d16920833d9b9673fd0b.png)

Figure 1: Our novel method, 𝐸 3 Gen, demonstrates its capability of generating high-fidelity avatars with detailed cloth wrinkles and achieves a rendering resolution of 1024 2 in real time. Furthermore, our method offers comprehensive control over camera views, and full-body poses, and supports attribute transfer and local editing.

## ABSTRACT

This paper aims to introduce 3D Gaussian for efficient, expressive, and editable digital avatar generation. This task faces two major challenges: 1) The unstructured nature of 3D Gaussian makes it incompatible with current generation pipelines; 2) the expressive

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

MM'24, October 28 - November 1, 2024, Melbourne, Australia.

© 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/18/06 https://doi.org/XXXXXXX.XXXXXXX

animation of 3D Gaussian in a generative setting that involves training with multiple subjects remains unexplored. In this paper, we propose a novel avatar generation method named 𝐸 3 Gen, to effectively address these challenges. First, we propose a novel generative UV features plane representation that encodes unstructured 3D Gaussian onto a structured 2D UV space defined by the SMPL-X parametric model. This novel representation not only preserves the representation ability of the original 3D Gaussian but also introduces a shared structure among subjects to enable generative learning of the diffusion model. To tackle the second challenge, we propose a part-aware deformation module to achieve robust and accurate full-body expressive pose control. Extensive experiments demonstrate that our method achieves superior performance in avatar generation and enables expressive full-body pose control and editing. Our project page is https://olivia23333.github.io/E3Gen.

<!-- page-break:2 -->

## CCS CONCEPTS

· Computing methodologies → Image-based rendering.

## KEYWORDS

Avatar Generation, Diffusion

## ACM Reference Format:

Weitian Zhang, Yichao Yan, Yunhui Liu, Xingdong Sheng, and Xiaokang Yang. 2018. 𝐸 3 Gen: Efficient, Expressive and Editable Avatars Generation. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (MM'24)Proceedings of the 32nd ACM International Conference on Multimedia (MM'24), October 28-November 1, 2024, Melbourne, Australia. ACM, New York, NY, USA, 10 pages. https://doi.org/XXXXXXX. XXXXXXX

## 1 INTRODUCTION

Digital avatars, we refer to as 3D clothed human characters, have extensive applications [2, 11] in various fields such as virtual and augmented reality, filmmaking, telecommunication, and more. Traditional graphics-based pipelines require weeks of labor from experienced 3D artists, utilizing sophisticated equipment [12, 20, 56] and software, to construct a single digital avatar. This manual and time-consuming process poses a significant obstacle to the creation of digital avatars at scale. Consequently, there is a pressing need for efficient methods that can generate digital avatars in a fast and autonomous way.

To enable the aforementioned applications, it would be desirable for digital avatars to meet the 𝐸 3 standard: (1) Efficient, i.e., digital avatars are expected to enable real-time, high-resolution, and realistic rendering. (2) Expressive, i.e., the avatars should allow animation not only by global body poses but also by local facial expressions and hand gestures. (3) Editable, i.e., digital avatars should support easy editing, including local editing and partial attribute transfer, as shown in Fig 1 and Fig 6. The key to achieving the 𝐸 3 standard is to design a powerful generative representation that can cover all these features.

Implicit and explicit representations are the main streams to model digital avatars. Implicit-function-based representations [19, 41] such as neural radiance field [37] can achieve photorealistic rendering results. However, due to the reliance on computationally expensive and time-consuming volume rendering, this representation struggles to support high-resolution and real-time rendering which is crucial in practical applications. The editing ability is also constrained due to the entanglement of geometry and texture. Explicit-mesh-based representations [13, 17, 48], on the other hand, enable high-resolution and real-time rendering through rasterization-based renderer [31] while struggling to represent thin structures like hair which alleviates the realism. A promising alternative for avatar representation is 3D Gaussian [29]. This representation offers realistic rendering quality, real-time high-resolution rendering, and supports editing and easy animation.

With the flourishing of 3D generative models [4, 16, 17, 25, 34, 38, 49, 54, 59], implicit representations [8, 22, 30, 39, 57] have been widely applied in the generation of digital avatars. 3D-aware GANs [5] have been employed to generate avatars in canonical pose space [3, 7, 15, 60], followed by a deformation module to transform the avatars into various body poses. These methods can generate avatars under the control of camera views and body poses, but due to the diverse topology and complex texture of avatars, they cannot produce realistic rendering results. Recently, diffusion-based methods [14, 21, 45, 53] have surpassed GAN-based methods [26-28] in 2D generation tasks, achieving high-quality generation results in complex and diverse scenes, promising for utilization in 3D content generation. Several attempts [10, 23] have been made to introduce diffusion-based methods for animatable avatar generation. However, the performance is still restricted by the current representation of avatars due to the aforementioned reasons.

It is natural to ask a question: Can we marry the generation power of diffusion model with the representation ability of 3D Gaussian to achieve the 𝐸 3 standard? Our answer is YES, but this is a non-trivial task due to the following reasons. 1) The unstructured nature of the 3D Gaussian poses challenges for its integration into diffusion-based generation pipelines which primarily consists of 2D-CNN-based networks [46]. 2) It is an unexplored problem for the expressive animation of 3D Gaussian in a generative setting.

To address the first challenge, our key insight is to project the unstructured 3D Gaussian into a structured 2D space. Following this idea, we propose a novel avatar representation, i.e., generative UV features plane. Specifically, we employ the parametric human model SMPL-X [42] as an initial template to provide a shared structure for the human body, and the corresponding 2D UV map is utilized to encode the attributes of 3D Gaussian. We assign the initial position of each 3D Gaussian primitive on the surface of a densified SMPL-X mesh. By decoding the UV feature maps, we obtain the attribute UV maps of 3D Gaussian. Each pixel within these maps represents the attributes of the corresponding 3D Gaussian anchored on the SMPL-X surface. This novel representation offers several benefits. First, it preserves the efficient advantage of the original 3D Gaussian representation while obtaining a shared structure UV space. Second, by utilizing the semantic information of SMPL-X, this representation supports editing. Third, it is compatible with generative models, enabling an efficient and editable avatar generation process.

The remaining question is how to enable expressive body control of 3D Gaussian in the generative setting. Animating the hands and face regions is particularly challenging due to their small volume and complex deformations, which make it difficult to learn or query accurate skinning weights. Leveraging the observation that these regions have limited topology changes compared to the body, we develop a part-aware deformation module that enables expressive full-body pose control. Specifically, we intuitively assign blendshapes and skinning weights derived from SMPL-X to the 3D Gaussian primitives in the face and hands regions, based on their barycentric coordinates. This ensures precise and robust control over facial expressions and hand gestures. The animation of body parts relies on precomputed K-nearest neighbors(KNN)-based skinning weights, which effectively handle deformations under large topology changes [15]. For animation in a generative setting, we further enhance the part-aware deformation module to allow for the disentanglement of body shape factors. This disentanglement enables training with multiple subjects possessing varied body shapes.

<!-- page-break:3 -->

Extensive experiments demonstrate that our method 𝐸 3 Gen achieves superior efficiency and generation quality compared to current state-of-the-art methods. Ablation studies have been conducted to evaluate the design choices of our method. Furthermore, 𝐸 3 Gen supports expressive pose control and editing, including attribute transfer (such as face shape and clothing) and local editing. In summary, our method has the following contributions:

- We propose 𝐸 3 Gen, an efficient avatar generation pipeline that enables real-time rendering at high resolution (1024 × 1024). It also supports expressive pose control and editing.
- We propose a novel 3D avatar presentation called generative UV features plane to combine the unstructured 3D Gaussian with current diffusion generation pipeline.
- Our 𝐸 3 Gen achieves superior generation quality on THuman2.0 Dataset [58] and supports editing including attribute transfer and local editing.

## 2 RELATEDWORK

## 2.1 3DDigital Human Generation

Various approaches [3, 8, 57] attempt to create animatable avatars by combining 3D-Aware GANs [4, 5, 49] or diffusion models [53] with implicit human representations. ENARF-GAN [39] utilizes an articulated neural representation based on tri-planes [5] but struggles to produce high-quality generation results. Other approaches [3, 15, 60] build upon EG3D [5] and employ super-resolution module to enhance the resolution of the generated avatars which leads to view inconsistency issues. To address this challenge, EVA3D [22] represents the digital human as a compositional part-based human representation. This approach achieves impressive rendering results with resolutions of 512. However, it falls short in enabling real-time rendering and faces difficulties in editing due to the entanglement of geometry and texture. Implicit human representations mainly rely on inverse skinning technique for avatar animation. However, this technique has certain drawbacks. Firstly, it tends to produce artifacts in joint regions and areas of contact. Secondly, for learnable skinning weights fields, it lacks generalization capabilities. To solve these issues, AG3D [15] adopts a forward-skinning technique [9] but faces the costly process of root finding. Another category of methods [17, 18] attempt to incorporate explicit mesh models to achieve real-time high-resolution rendering while providing better support for editing and animation. These mesh-based methods often rely on parametric models such as SMPL [36] to learn offsets or employ neural networks [50] for mesh optimization. However, meshes representations face challenges in accurately representing thin structures, such as hair, and may be constrained by the topology limitations imposed by models like SMPL, resulting in generated avatars that lack realism in their rendering results.

Recently, 3D Gaussians [29] serve as an explicit representation that supports both editing and animation capabilities while enabling real-time high-resolution rendering. This representation offers greater representation capability compared to meshes. A concurrent work, GSM [1], combines 3D Gaussians with EG3D [5] using shell maps [44]. In comparison, our generative UV features plane representation facilitate expressive animation that includes facial expressions and gestures, while also supporting editing capabilities. Additionally, GSM [1] control body-only poses and faces challenges in editing local regions of the avatar. We do not compare to it since the training code have not been released.

## 2.2 DiffusionModel

The diffusion model has recently achieved significant success in the field of generation, surpassing GANs in tasks such as text-to-image generation. Consequently, there is a growing interest in extending the success of the diffusion model from 2D to 3D generation. One approach [43, 55] leverage prior knowledge encoded in pre-trained latent diffusion models for text-to-3D generation. However, these methods often employ per-subject optimization, which can take hours to generate a single sample, thus reducing efficiency.

Another approach [40, 51] focuses on directly generating 3D representations, enabling faster inference. However, many of these methods adopt a two-stage training scheme. This can introduce noisy patterns and artifacts in the latent code due to the uncertain nature of inverse rendering. Consequently, these noisy patterns can distract denoising networks and affect the quality of the generated outputs. To solve this, SSDNeRF [6] proposes a single-stage diffusion model that trains the fitting and denoising parts together, leveraging the diffusion priors to constrain the latent codes. However, SSDNeRF focuses on static object generation and adopts an implicit representation, which limits the rendering resolution to only 128x128. In comparison, our proposed method can achieve high-resolution rendering at 1024x1024 in real time while generating articulated digital avatars.

## 3 PRELIMINARY

In this section, we provide a brief introduction about SMPL-X human model in Sec. 3.1 and 3D Gaussians Splatting in Sec. 3.2.

## 3.1 SMPL-X

SMPL-X is an animatable parametric human model that represents human body (without cloth) with a parameterized deformable mesh 𝑀(𝛽, 𝜃,𝜓). This model consists of 10,475 vertices and 54 joints, allowing for control over hand gestures and facial expressions. The deformation process can be formulated as follows:

<!-- formula-not-decoded -->

where 𝛽, 𝜃 and𝜓 represent shape, pose and expression parameters respectively. The linear blend skinning (LBS) function, denoted as 𝐿𝐵𝑆(·), is used to transform the canonical template 𝑇𝑃 the given pose 𝜃 based on the skinning weights W and joint locations 𝐽 (𝛽). The canonical template 𝑇𝑃 can be computed as:

<!-- formula-not-decoded -->

where, 𝑇𝐶 denotes the mean shape template. 𝐵𝑆 (𝛽; S), 𝐵𝐸 (𝜓; 𝜖) and 𝐵𝑃 (𝜃 ; P) represent per-vertex displacements calculated by the blend shapes S , P and 𝜖 with their corresponding shape, pose and expression parameters.

## 3.2 Gaussian Splatting

3D Gaussian Splatting is an explicit point-based representation for 3D static scenes, involving a collection of 3D Gaussian primitives MM'24, October 28 - November 1, 2024, Melbourne, Australia.

<!-- page-break:4 -->

Weitian Zhang, Yichao Yan, Yunhui Liu, Xingdong Sheng, and Xiaokang Yang

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_to3dmodel_orfullbody/e3%20gen%20avatars/images/image_000001_19915a6df270b7c5420b01aa88e3d9db46ea832cc6f15c47f139f30535206f39.png)

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_to3dmodel_orfullbody/e3%20gen%20avatars/images/image_000002_cc49dd913f9d3797fc2e83fb0e962c5067f56e856efc576e070a7ff7a59b0cdc.png)

Canonical Space

2," = { % /", %0", ̂*", ̂)", ̂+"} Targeted Pose Space Figure 2: Method Overview. Our approach utilizes a single-stage diffusion model to simultaneously train the denoising and fitting process. The UV features plane, 𝑥𝑖 , is randomly initialized and optimized by both processes. In the denoising process, noise is added to the UV features plane and then denoised following a v-parameterization scheme using a denoising UNet. In the fitting process, the UV features plane is decoded into Gaussian Attribute maps, which are used to generate a 3D-Gaussian-based avatar in canonical space by fetching the corresponding attributes for the initialized Gaussian primitive. Finally, a part-aware deformation module is employed to deform the avatar into targeted pose based on SMPL-X parameters.

denoted as G. These primitives enable real-time rendering through differentiable rasterization. Each 3D Gaussian G𝑘 comprises five attributes: position 𝜇, scaling matrix 𝑆, rotation matrix 𝑅, opacity 𝛼, and view-dependent color 𝑐, which is represented by coefficients of spherical harmonics. In practice, we employ RGB color instead of spherical harmonics coefficients for simplicity and utilize the diagonal vector s ∈ R 3 and axis-angle r ∈ R 3 , to represent the scaling and rotation matrix, respectively.

The 3D Gaussians are projected onto the 2D image plane during the rendering process. The resulting pixel color 𝐶 is computed by blending the 𝑁 projected 3D Gaussians primitives within that pixel. This process can be formulated as:

<!-- formula-not-decoded -->

where 𝑐𝑖 denotes the color of the 𝑖-th projected 3D Gaussians primitive, and 𝛼𝑖 represents the blending weight calculated from the learned opacity and probability density.

## 4 METHOD

In this work, we propose 𝐸 3 Gen, a generative model designed for efficient, expressive, editable digital avatar generation. An overview of 𝐸 3 Gen is illustrated in Fig 2.

To achieve efficient, expressive and editable avatar generation, we propose a novel generative UV features plane representation (Sec 4.1). This representation ensures compatibility between the 3D Gaussian and the generative diffusion model while preserving efficiency. In Sec 4.2, we present the part-aware deformation module. This module offers full-body pose control, including facial expressions and gestures, allowing for expressive avatar animations. The training scheme of our method is detailed in Sec 4.3. Furthermore, in Sect 4.4, we discuss the editing capability of our method.

## 4.1 Generative UV Features Representation

Get inspiration from previous work for adopting 3D Gaussians in animatable avatar reconstruction tasks, we aim to introduce 3D Gaussians as a target space for diffusion model. Different from reconstruction tasks focusing on per-subject optimization, we have to enable 3D Gaussian training among multiple subjects, thus a shared structure among subjects is needed. To be compatible with the 2DCNN-based denoising network in diffusion model, the shared structure is expected to be a 2D representation. Therefore, we propose to represent 3D Gaussian based digital avatars in the 2D UV space defined by the SMPL-X parametric model.

Given a generated UV features plane 𝑥𝑖 , we extract a set of 𝐾 3D Gaussian primitives from it to obtain a 3D Gaussian-based generated avatar G:

<!-- formula-not-decoded -->

Each Gaussian primitive G𝑘 is parameterized by a 3D position 𝜇𝑘 ∈ R 3 , an opacity 𝛼𝑘 ∈ R, a rotation matrix 𝑅 represented by the axis angle representation 𝑟𝑘 ∈ R 3 , a scale matrix 𝑆 represented by a diagonal vector 𝑠𝑘 ∈ R 3 , and a rgb color 𝑐𝑘 ∈ R 3 .

The input generated UV features plane is first separated evenly into two parts and then decoded with two light-weight shared decoders 𝐷𝑎 and 𝐷𝑔, respectively. The separation of the UV features plane enables the disentanglement of geometry and texture, facilitating capability in editing. 𝐷𝑔 predicts geometry related attributes of 3D Gaussians: position 𝜇 and opacity 𝛼, while 𝐷𝑎 predict appearance related attributes of 3D Gaussians: scale s, rotation r and color 𝑐. As demonstrated by Li et al. [32], convolutional-based decoder provide realistic results with more details than MLP-based decoder. Thus, we construct 𝐷𝑔 and 𝐷𝑎 as shallow convolutional-based networks for fast inference and high generation quality.

<!-- page-break:5 -->

With these decoded attribute maps, we can extract a 3D Gaussianbased avatar in canonical pose space. Specifically, we query the attributes of those Gaussian primitives G𝑘 by projecting them according to their initial position ˆ 𝜇𝑘 onto each of the five attribute planes, retrieving the corresponding attributes via bilinear interpolation, thus obtaining a set of Gaussian primitives G which represent a digital avatar. In practice, the position 𝜇𝑘 , scale s𝑘 and rotation r𝑘 of one Gaussian primitive G𝑘 are modeled relative to the SMPL-X template as follows:

<!-- formula-not-decoded -->

where ˆ 𝜇𝑘 , ˆs𝑘 and ˆr𝑘 are initial values based on SMPL-X parametric model. 𝛿𝜇𝑘 , 𝛿s𝑘 , 𝛿r𝑘 represent the predicted values queried from attribute maps. The weak constraints to the surface of SMPL-X model ensure reasonable generation results without undermining the representation ability of the original 3D Gaussian.

Gaussian Primitives Initialization. The positions ˆ 𝜇 of Gaussian primitives are initialized by sampling the center points of faces on a densified SMPL-X model. Similar to TADA [33], we subdivide the SMPL-X model to enhance the generation quality. We do not sample points uniformly on the UV space as the UV space shows a sparsity of points on the body surface which might undermine the representation ability. All Gaussian primitives are assigned an initial scale ˆs according to the distances between them and their neighbors. The initial scale is calculated in the targeted pose space to enable stretching ability for reasonable animation results. Different from origin 3D Gaussian, we set the orientation of each Gaussian primitive as the local tangent frame of the 3D surface point, similar to Lombardi et al. [35]. This initialization introduces human prior which alleviates the spiking artifacts during the animation process. We also adopt a different rotation representation: axis-angles compared to the quaternion utilized in previous work. Because the elements in this representation have the same value range which stabilizes the optimization of the neural network.

## 4.2 Part-aware DeformationModule

To achieve expressive full-body pose control, we propose a partaware deformation module to transform the extracted digital avatar G into targeted pose space with accurate control over hands and face. The deformation module follows a forward skinning scheme based on linear blend skinning technique. Specifically, we apply the following transformation to the position 𝜇𝑘 of each Gaussian primitive G𝑘:

<!-- formula-not-decoded -->

where 𝑛𝑏 is the number of joints, and B𝑖 denotes the transformation matrix of the 𝑖th joint from the canonical pose space to the targeted pose space. 𝑤𝑖 represents skinning weights, which determine the influence of the motion of each joint on position 𝜇𝑘 quantitatively.

To compute the skinning weights field for our deformation module, we leverage the predefined skinning weights on the SMPL-X mesh due to the complexity of diffusion training scheme compared to reconstruction tasks. Additionally, since the hands and face regions are relatively small but exhibit intricate deformation, fullbody pose control has long been challenging. With the alignment to SMPL-X and the observation that the face and hands regions undergo minimal topology changes, we can compute the skinning weights directly based on their barycentric coordinates. This approach ensures accurate and robust animation for these specific regions. For body parts where large topology changes may occur, applying the same technique directly would result in artifacts. Drawing inspiration from AG3D [15], we represent the skinning weights field for body parts using a low-resolution volume. Each voxel in the volume is assigned skinning weights by calculating the values through the accumulation of skinning weights from each of the K nearest vertices on the densified SMPL-X surface, weighted by inverse distance. Gaussian primitives associated with the body part can then retrieve their skinning weights through trilinear interpolation from the volume. This approach enables accurate and smooth deformation while handling large topology changes effectively. Different from [15], our deformation method enables full-body animation, whereas AG3D focuses on specific body parts. Additionally, we avoid the costly root-finding process associated with the implicit representation adopted by AG3D, achieve more efficient deformation.

3D Gaussian Attribute Deformation. During the deformation process, the opacity 𝛼 and color 𝑐 remain unchanged, while the scale 𝑠, rotation 𝑟 and position 𝜇 change. We have already discussed the deformation of position 𝜇 in the previous paragraphs. In this part, we try to tackle the deformation in rotation 𝑟 and scale 𝑠. The rotation 𝑟 is updated using the following formula:

<!-- formula-not-decoded -->

where R is the rotation matrix derived from the axis angle representation 𝑟 , and T is the transformation matrix computed as the weighted sum of the bone transformations B𝑖 . 𝑤𝑖 (𝜇) corresponds to the skinning weights associated with the position 𝜇. To account for the change in scale 𝑠, we define the initial value of 𝑠 in the targeted pose space. Specifically, using the deformed Gaussian primitives, we determine the initial value of 𝑠 based on the distance to its neighboring primitives.

By updating the rotation 𝑟 and scale 𝑠, we ensure that the deformations applied to the Gaussian primitives encompass changes in rotation, scale, and position, , allowing for robust and accurate pose control.

Adaptation to Multi-subjects. To be compatible with the generation task, which are trained with multiple subjects in various body shapes, we disentangle the body shape factor. We model the generated avatar in canonical space with a neutral body shape. And a warping process is employed to map the neutral body avatar into the targeted body shape space before transforming to the targeted pose space. The warping process can be formulated as follows:

<!-- formula-not-decoded -->

where, 𝜇 is the positions of 3D Gaussian primitives in canonical space. 𝐵𝑆 (𝛽; S, 𝜇) is the offset derived from the displacements on each SMPL-X vertex calculated by shape parameters 𝛽 with corresponding bases S. To enable accurate expressive control and alleviate artifacts in joints, we further add expression offsets 𝐵𝐸 (𝜓; 𝜖) and pose correction term 𝐵𝑃 (𝜃; P) to the warped 3D Gaussian, following SMPL-X deformation process.

<!-- page-break:6 -->

## 4.3 Training

We follow the single-stage diffusion training process where UV features plane fitting and denoising processes are conducted simultaneously. The total training objective can be formulated as:

<!-- formula-not-decoded -->

where 𝑥𝑖 denotes the UV feature plane, 𝜙 and𝜓 are parameters of the denoising U-Net and shared decoder, respectively. Lfit and Ldenois represent training objective for the fitting and denoising process. 𝜆∗ are loss weights. Previous diffusion-based methods adopt a twostage training scheme, where a fitting process is trained first to obtain per-subject latent feature planes. The obtained latent feature planes are utilized as ground truth for the second-step denoising process training. Different these methods, our UV features plane 𝑥𝑖 is constrained by both terms in the loss function. The introducing of denoising process constrain for UV feature plane optimization is beneficial for learning unseen regions in the training data as demonstrated by previous work [6].

Fitting Process. During the fitting process, we optimize the UV features plane as well as the geometry and appearance decoder via the following loss function:

<!-- formula-not-decoded -->

The color loss Lc computes the L2 distance between the ground truth images and the rendered results of our generated avatars. We randomly sample several images instead of one from all available observations for each scene in one training step to prevent the model from getting stuck in a local minimum. Different from SSDNeRF [6] which is limited by implicit representation that can only be optimized via per-pixel objectives, the efficiency of generative UV features representation enables us to render the whole images and apply perceptual loss [24] on them. Specifically, the perceptual loss is calculated based on the features maps of ground truth images and our rendered outputs which are extracted from a pre-trained VGG [52] network. Lreg = ∥𝛿𝜇𝑘 ∥ 2 2 constrains the predicted offset values from being unreasonably large.

Denoising Process. For digital avatar generation, we utilize the diffusion model to learn a mapping from Gaussian noise to generative UV features plane. With Gaussian noise as input, the diffusion model can denoise it and output a reasonable UV features plane. During this process, we optimize both the generative UV features plane 𝑥𝑖 and the denoising UNet' s parameters 𝜙. Specifically, we first add Gaussian noise 𝜖 ∼ N(0, 𝐼 ) into the given generative UV features plane 𝑥𝑖 via a noise schedule comprising differentiable functions 𝛼(𝑡) and 𝜎(𝑡), obtaining a noisy feature plane 𝑥𝑖 (𝑡) := 𝛼(𝑡)𝑥𝑖 +𝜎(𝑡)𝜖 at diffusion time step 𝑡 . Then, we utilize the denoising UNet to obtain the denoised output ˆ 𝑥𝑖 via:

<!-- formula-not-decoded -->

where ˆ 𝑣 ≡ 𝛼(𝑡)𝜖 - 𝜎(𝑡)𝑥𝑖 according to the 𝑣-parameterization method proposed in [47]. The denoising loss is formulated as following:

<!-- formula-not-decoded -->

where 𝑡 ∼ U(0,𝑇 ), 𝜔 is a hyperparameter which we empirically set to 0.5.

## 4.4 Editing

Our novel representation, generative UV features plane, facilitates various customization applications, including local region editing and attribute transfer between subjects. By disentangling geometry and appearance, we expand the capabilities for editing, allowing for editing on either geometry or appearance individually. We provide visual examples in our experiment results(Fig 6).

Local Region Editing. In contrast to previous methods that employ monolithic representations, where generated avatars are treated as a unified entity with entangled attributes, the generative UV features plane represents 3D digital avatars as a collection of 3D Gaussian primitives that are loosely connected to the SMPL-X parametric model. This representation allows for enhanced flexibility and freedom in editing the avatars. Each Gaussian primitive can be independently modified by optimizing its associated geometry or appearance UV features, or by directly manipulating its attribute values.

Attribute Transfer. Benefiting from the shared structure among subjects, we can easily transfer geometry and appearance attributes from one generated subject to another by swapping the corresponding features. The utilization of UV space, which ensures semantic consistency, further facilitates the transfer of specific regions such as the nose, clothing, and face.

## 5 EXPERIMENTS

Dataset. In our experiments, we utilized the THUman2.0 Dataset [58] as our training data. This dataset comprises 526 textured 3D scans captured using a dense DSLR rig, covering a wide range of poses. Each scan is accompanied by its corresponding SMPLX parameters. Our data pre-processing involves rendering 500 identities from the THUman2.0 dataset using 54 camera views for each identity. Notably, our training did not rely on explicit 3D supervision, such as normals or 3D meshes, ensuring future extension to multi-view video datasets.

Metrics. Consistent with previous works, we evaluate the quality and diversity of the generated avatars using the Fre ' chet Inception Distance (FID) metrics. To compute metrics, we employ a set of 50,000 rendered multi-view images. Additionally, we extend our analysis by calculating the FIDnorm metrics between the generated and ground truth normal maps, providing insights into the quality of generated geometry. We estimate the normal values based on the axis directions of each 3D Gaussian Primitives.

## 5.1 Evaluation of Generated Avatars

5.1.1 Generation and Animation Capacities. We demonstrate the capability of our method to generate diverse and detailed avatars

<!-- page-break:7 -->

𝐸 3 Gen: Efficient, Expressive and Editable Avatars Generation MM'24, October 28 - November 1, 2024, Melbourne, Australia.

Figure 3: We demonstrate the effectiveness of our method in achieving precise and robust control over facial expressions and gestures. Our approach enables clear and distinct control over each individual finger, ensuring their visibility and accurate positioning. Additionally, our method exhibits strong robustness when faced with novel poses, producing reasonable and plausible results for facial expressions.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_to3dmodel_orfullbody/e3%20gen%20avatars/images/image_000003_2b490bffc9f902ecb88bf1c903884135833472aee694ea92cb4325bf882be803.png)

with controllable poses and camera views in Fig 1 and Fig 3. Our method ensures view consistency and produces high-resolution rendering results for the generated avatars. Additionally, we achieve accurate and robust control over hand and facial expressions. Notably, despite being trained on the THUman2.0 dataset, which has limited facial expressions, our method is able to generate reasonable open mouth expressions that are not explicitly present in the dataset. These results highlight the effectiveness and versatility of our approach.

5.1.2 Comparisons. We compare our method with representative approaches in both implicit representation and explicit representation, including those utilizing 3D Aware GANs or diffusion models. The comparison results are presented in Table 1. The visual results in shown in Fig 4. Our method exhibits superior visual quality and diversity and achieves a rendering speed of over 100 FPS for highresolution rendering, as measured by FID and FPS. This demonstrates the power of our proposed generative UV features plane. Furthermore, even without the supervision of normal maps and 3D models, our method achieves high performance in capturing detailed geometry surfaces, as indicated by the FIDnorm metrics. Constrained by implicit representations, GNARF [3], ENARF-GAN [39], and EVA3D [22] exhibit limited frames per second. GETAvatar [61] achieves faster rendering benefiting from its explicit Mesh representation. PrimDiffusion, which adopts a mixture of primitive representation, achieves an extraordinary FPS of 88. However, obtaining geometry surfaces from this representation is challenging. The results tend to be blurry, we guess that might result from artifacts caused by the overlapping patches.

Table 1: Quantitative comparison on the THUman2.0 [58] dataset, and the ∗ results are adopted from GETAvatar [61].

| Methods            |   Res |   FID↓ | FIDnorm ↓   |   FPS↑ |
|--------------------|-------|--------|-------------|--------|
| ENARF ∗ [39]       |   128 | 124.61 | 223.72      |      8 |
| GNARF ∗ [3]        |   256 |  68.31 | 166.62      |      8 |
| EVA3D [22]         |   512 | 188.95 | 60.37       |      6 |
| GETAvatar ∗ [61]   |  1024 |  17.91 | 55.02       |     17 |
| PrimDiffusion [10] |   512 |  68.60 | NA          |     88 |
| Ours               |  1024 |  15.78 | 25.63       |    110 |

## 5.2 Ablation Study

Generative UV features vs Generative UV attributes. Instead of intuitively producing the Gaussian Attribute maps, we encode them as latent feature code planes, which enhance the training stability and lead to significant generation quality improvements according to Tab 2. This is due to the complexity of the attributes of 3D Gaussians, which makes them challenging to optimize directly. To address this issue, we employ a lightweight decoder to preserve the efficiency of the original 3D Gaussians. As shown in the Tab 1, our model' s speed is not significantly affected by the additional latent decoding process.

The initialization of 3D Gaussian Primitives. We initialize our generative UV features plane based on SMPL-X prior. Experimental results presented in Tab 2 demonstrate that this initialization approach yields improved generation quality and better surface Ours Figure 4: Qualitative Comparison. Our method demonstrates superior performance in rendering quality and geometry quality compared to other methods. Due to the challenge of obtaining normals directly from PrimDiffusion, we visualize its mixture primitives as a rough representation of the geometric structure.

<!-- page-break:8 -->

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_to3dmodel_orfullbody/e3%20gen%20avatars/images/image_000004_cd7d27875a80782d8a0a4df6d3101f30de2642c7e224ab90716b005ddada19b0.png)

Forward Skinning KNN

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_to3dmodel_orfullbody/e3%20gen%20avatars/images/image_000005_14f94d8e69f5734c9ed95e449e5dc81947eab3009019da0e285880401d5f6325.png)

Ours

Figure 5: Ablation on deformation method. Our method achieves more accurate results for a given facial expression compared to the K-nearest neighbors (KNN) based forward skinning method.

geometry. The weak constraint to SMPL-X does not compromise the expressive power of our representation. As evident from Fig 4, our method is capable of generating avatars in loose clothes (the girl in red dress) that are not constrained by the SMPL-X topology.

Table 2: Ablation study on the initialization of Gaussian primitives and the utilization of UV features.

| Methods       |   FID↓ |   FIDnorm ↓ |   KID ↓ |
|---------------|--------|-------------|---------|
| w.o init      |  16.74 |       51.42 |   13.75 |
| UV Attributes |  17.92 |       43.23 |   15.41 |
| Full Pipeline |  15.78 |       25.63 |   13.30 |

The part-aware deformation methods vs KNN-based forward skinning. Due to the small facial area and its rich range of motion, using a K-nearest neighbors (KNN) based forward skinning method leads to errors, as depicted in Fig 5. The KNN-based forward skinning method fails to accurately open the mouth of the avatar according to the driving pose. In contrast, our part-aware deformation module takes advantage of the minimal topology changes in the face and hands, employing different approaches to obtain skinning weights for the body, face, and hands, resulting in robust and accurate full-body expressive pose control. Fig 3 demonstrates the precise control achieved over the fingers and facial expressions using our method. It is important to note that since our method does not model the interior of the mouth, the color display after opening the mouth represents the colors from the hair and neck region, and not artifacts caused by the driving method.

## 5.3 Applications

The Generative UV features plane supports editing of generated avatars, including local editing and attribute transfer between subjects. Fig 6 provides examples of these two editing methods. For local editing, the UV features plane represents a person as a collection of 3D Gaussian primitives. By modifying the attributes of local regions ' 3D Gaussian primitives, local editing can be achieved.

Fig 6 shows an example of editing the nose of the person is shown,

resulting in a change in nose length. The edited avatar can still be

controlled in a similar manner.

Local editing

Attribute Transfer

Figure 6: Our method enables local editing and attribute transfer. In row one, we demonstrate the capability to modify only the nose of the avatar. The shared structure of UV featuers plane allows us to transfer attributes between different subjects, as showcased in row two.

![Image](/home/hp/btp_avatar/Literature-review-rag/parsed/text_to3dmodel_orfullbody/e3%20gen%20avatars/images/image_000006_6cdd46960df6b258a986e1b6a92ac96a9f83158d6ab44005209b315eece5cea2.png)

Due to the shared UV structure provided by the UV features plane among subjects, attribute transfer between subjects can be easily performed. The second row of Fig 6 demonstrates the effect of exchanging facial attributes. The first two columns show the original generated avatars, while the last two columns show the results after attribute exchange. This support for editing enhances the practical applicability of our method in industrial settings. For more editing results, please refer to the Sup. Mat.

<!-- page-break:9 -->

## 6 CONCLUSION

In conclusion, this paper introduces a novel method, called 𝐸 3 Gen, for efficient, expressive, and editable digital avatar generation using 3D Gaussians. The paper addresses two major challenges in this task: the unstructured nature of 3D Gaussians and the animation of 3D Gaussians in a generative setting involving multiple subjects. To overcome these challenges, the proposed method introduces a generative UV features representation that encodes unstructured 3D Gaussians onto a structured 2D UV space defined by the SMPLX parametric model. This representation preserves the expressive power of 3D Gaussians while introducing a shared structure among subjects, enabling generative learning of the diffusion model. To achieve robust and accurate full-body expressive pose control, a part-aware deformation module is proposed. This module enables precise control of avatar poses. Extensive experiments demonstrate the superior performance of the proposed method in avatar generation, as well as its ability to enable expressive full-body pose control and editing.

## REFERENCES

- [1] Rameen Abdal, Wang Yifan, Zifan Shi, Yinghao Xu, Ryan Po, Zhengfei Kuang, Qifeng Chen, Dit-Yan Yeung, and Gordon Wetzstein. 2023. Gaussian Shell Maps for Efficient 3DHuman Generation. arXiv:2311.17857 [cs.CV]
- [2] Timur Bagautdinov, Chenglei Wu, Tomas Simon, Fabián Prada, Takaaki Shiratori, Shih-En Wei, Weipeng Xu, Yaser Sheikh, and Jason Saragih. 2021. Driving-signal aware full-body avatars. ACM Trans. Graph. 40, 4, Article 143 (jul 2021), 17 pages. https://doi.org/10.1145/3450626.3459850
- [3] Alexander W. Bergman, Petr Kellnhofer, Wang Yifan, Eric R. Chan, David B. Lindell, and Gordon Wetzstein. 2022. Generative Neural Articulated Radiance Fields. In NeurIPS.
- [4] Eric Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and Gordon Wetzstein. 2021. pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis. In Proc. CVPR.
- [5] Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J. Guibas, Jonathan Tremblay, Sameh Khamis, Tero Karras, and Gordon Wetzstein. 2022. Efficient Geometry-Aware 3D Generative Adversarial Networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 16123-16133.
- [6] Hansheng Chen, Jiatao Gu, Anpei Chen, Wei Tian, Zhuowen Tu, Lingjie Liu, and Hao Su. 2023. Single-stage diffusion nerf: A unified approach to 3d generation and reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2416-2425.
- [7] Xinya Chen, Jiaxin Huang, Yanrui Bin, Lu Yu, and Yiyi Liao. 2023. VeRi3D: Generative Vertex-based Radiance Fields for 3D Controllable Human Image Synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 8986-8997.
- [8] Xu Chen, Tianjian Jiang, Jie Song, Jinlong Yang, Michael J Black, Andreas Geiger, and Otmar Hilliges. 2022. gdna: Towards generative detailed neural avatars. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20427-20437.
- [9] Xu Chen, Yufeng Zheng, Michael J Black, Otmar Hilliges, and Andreas Geiger. 2021. Snarf: Differentiable forward skinning for animating non-rigid neural implicit shapes. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 11594-11604.
- [10] Zhaoxi Chen, Fangzhou Hong, Haiyi Mei, Guangcong Wang, Lei Yang, and Ziwei Liu. 2023. PrimDiffusion: Volumetric Primitives Diffusion for 3D Human Generation. In Thirty-seventh Conference on Neural Information Processing Systems.
- [11] Hang Chu, Shugao Ma, Fernando De la Torre, Sanja Fidler, and Yaser Sheikh. 2020. Expressive telepresence via modular codec avatars. In Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XII 16. Springer, 330-345.
- [12] Alvaro Collet, Ming Chuang, Pat Sweeney, Don Gillett, Dennis Evseev, David Calabrese, Hugues Hoppe, Adam Kirk, and Steve Sullivan. 2015. High-quality streamable free-viewpoint video. ACM Transactions on Graphics (ToG) 34, 4 (2015), 1-13.
- [13] Enric Corona, Albert Pumarola, Guillem Alenya, Gerard Pons-Moll, and Francesc Moreno-Noguer. 2021. Smplicit: Topology-aware generative model for clothed people. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 11875-11885.
- [14] Prafulla Dhariwal and Alexander Nichol. 2021. Diffusion models beat gans on image synthesis. Advances in neural information processing systems 34 (2021), 8780-8794.
- [15] Zijian Dong, Xu Chen, Jinlong Yang, Michael J Black, Otmar Hilliges, and Andreas Geiger. 2023. AG3D: Learning to Generate 3D Avatars from 2D Image Collections. In International Conference on Computer Vision (ICCV).
- [16] Ziya Erkoç, Fangchang Ma, Qi Shan, Matthias Nießner, and Angela Dai. 2023. HyperDiffusion: Generating Implicit Neural Fields with Weight-Space Diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 14300-14310.
- [17] Jun Gao, Tianchang Shen, ZianWang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. 2022. Get3d: A generative model of high quality 3d textured shapes learned from images. Advances In Neural Information Processing Systems 35 (2022), 31841-31854.
- [18] Artur Grigorev, Karim Iskakov, Anastasia Ianina, Renat Bashirov, Ilya Zakharkin, Alexander Vakhitov, and Victor Lempitsky. 2021. Stylepeople: A generative model of fullbody human avatars. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5151-5160.
- [19] Jiatao Gu, Lingjie Liu, Peng Wang, and Christian Theobalt. 2022. StyleNeRF: A Style-based 3D Aware Generator for High-resolution Image Synthesis. In International Conference on Learning Representations. https://openreview.net/ forum?id=iUuzzTMUw9K
- [20] Kaiwen Guo, Peter Lincoln, Philip Davidson, Jay Busch, Xueming Yu, Matt Whalen, Geoff Harvey, Sergio Orts-Escolano, Rohit Pandey, Jason Dourgarian, et al. 2019. The relightables: Volumetric performance capture of humans with realistic relighting. ACM Transactions on Graphics (ToG) 38, 6 (2019), 1-19.
- [21] JonathanHo, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840-6851.
- [22] FangzhouHong, Zhaoxi Chen, Yushi LAN, LiangPan, andZiwei Liu. 2023. EVA3D: Compositional 3D Human Generation from 2D Image Collections. In International Conference on Learning Representations. https://openreview.net/forum?id= g7U9jD \_ 2CUr
- [23] Shoukang Hu, Fangzhou Hong, Tao Hu, Liang Pan, Haiyi Mei, Weiye Xiao, Lei Yang, and Ziwei Liu. 2023. HumanLiff: Layer-wise 3D Human Generation with Diffusion Model. arXiv preprint (2023).
- [24] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. 2016. Perceptual Losses for Real-Time Style Transfer and Super-Resolution. In Computer Vision - ECCV 2016, Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling (Eds.). Springer International Publishing, Cham, 694-711.
- [25] Animesh Karnewar, Andrea Vedaldi, David Novotny, and Niloy Mitra. 2023. HoloDiffusion: Training a 3D Diffusion Model using 2D Images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.
- [26] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. 2020. Training Generative Adversarial Networks with Limited Data. In Proc. NeurIPS.
- [27] Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. 2021. Alias-Free Generative Adversarial Networks. In Proc. NeurIPS.
- [28] Tero Karras, Samuli Laine, and Timo Aila. 2019. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 4401-4410.
- [29] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42, 4 (2023), 1-14.
- [30] Byungjun Kim, Patrick Kwon, Kwangho Lee, Myunggi Lee, Sookwan Han, Daesik Kim, and Hanbyul Joo. 2023. Chupa: Carving 3D Clothed Humans from Skinned Shape Priors using 2D Diffusion Probabilistic Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 15965-15976.
- [31] Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila. 2020. Modular primitives for high-performance differentiable rendering. ACM Transactions on Graphics (ToG) 39, 6 (2020), 1-14.
- [32] Zhe Li, Zerong Zheng, Lizhen Wang, and Yebin Liu. 2024. Animatable Gaussians: Learning Pose-dependent Gaussian Maps for High-fidelity Human Avatar Modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
- [33] Tingting Liao, Hongwei Yi, Yuliang Xiu, Jiaxiang Tang, Yangyi Huang, Justus Thies, and Michael J. Black. 2024. TADA! Text to Animatable Digital Avatars. In International Conference on 3D Vision (3DV).
- [34] Zhen Liu, Yao Feng, Michael J. Black, Derek Nowrouzezahrai, Liam Paull, and Weiyang Liu. 2023. MeshDiffusion: Score-based Generative 3D Mesh Modeling.

<!-- page-break:10 -->

- In International Conference on Learning Representations. https://openreview.net/ forum?id=0cpM2ApF9p6
- [35] Stephen Lombardi, Tomas Simon, Gabriel Schwartz, Michael Zollhoefer, Yaser Sheikh, and Jason Saragih. 2021. Mixture of volumetric primitives for efficient neural rendering. ACM Transactions on Graphics (ToG) 40, 4 (2021), 1-13.
- [36] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. 2015. SMPL: A Skinned Multi-Person Linear Model. ACM Trans. Graphics (Proc. SIGGRAPH Asia) 34, 6 (Oct. 2015), 248:1-248:16.
- [37] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In ECCV.
- [38] Michael Niemeyer and Andreas Geiger. 2021. GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR).
- [39] Atsuhiro Noguchi, Xiao Sun, Stephen Lin, and Tatsuya Harada. 2022. Unsupervised Learning of Efficient Geometry-Aware Neural Articulated Representations. In European Conference on Computer Vision.
- [40] Evangelos Ntavelis, Aliaksandr Siarohin, Kyle Olszewski, ChaoyangWang, Luc V Gool, and Sergey Tulyakov. 2023. Autodecoding latent 3d diffusion models. Advances in Neural Information Processing Systems 36 (2023), 67021-67047.
- [41] Roy Or-El, Xuan Luo, Mengyi Shan, Eli Shechtman, Jeong Joon Park, and Ira Kemelmacher-Shlizerman. 2022. StyleSDF: High-Resolution 3D-Consistent Image and Geometry Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 13503-13513.
- [42] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. 2019. Expressive Body Capture: 3D Hands, Face, and Body from a Single Image. In Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR).
- [43] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. 2023. DreamFusion: Text-to-3D using 2D Diffusion. In The Eleventh International Conference on Learning Representations. https://openreview.net/forum?id=FjNys5c7VyY
- [44] Serban D. Porumbescu, Brian Budge, Louis Feng, and Kenneth I. Joy. 2005. Shell maps. ACM Trans. Graph. 24, 3 (jul 2005), 626-633. https://doi.org/10.1145/ 1073204.1073239
- [45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684-10695.
- [46] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention-MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer, 234-241.
- [47] Tim Salimans and Jonathan Ho. 2022. Progressive Distillation for Fast Sampling of Diffusion Models. In International Conference on Learning Representations. https://openreview.net/forum?id=TIdIXIpzhoI
- [48] Soubhik Sanyal, Partha Ghosh, Jinlong Yang, Michael J Black, Justus Thies, and Timo Bolkart. 2023. SCULPT: Shape-conditioned unpaired learning of posedependent clothed and textured human meshes. arXiv preprint arXiv:2308.10638 (2023).
- [49] Katja Schwarz, Yiyi Liao, Michael Niemeyer, and Andreas Geiger. 2020. Graf: Generative radiance fields for 3d-aware image synthesis. Advances in Neural Information Processing Systems 33 (2020), 20154-20166.
- [50] Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. 2021. Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis. In Advances in Neural Information Processing Systems (NeurIPS).
- [51] J Ryan Shue, Eric Ryan Chan, Ryan Po, Zachary Ankner, Jiajun Wu, and Gordon Wetzstein. 2023. 3d neural field generation using triplane diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2087520886.
- [52] K Simonyan and A Zisserman. 2015. Very deep convolutional networks for largescale image recognition. 3rd International Conference on Learning Representations (ICLR 2015), 1-14.
- [53] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2021. Score-Based Generative Modeling through Stochastic Differential Equations. In International Conference on Learning Representations. https://openreview.net/forum?id=PxTIG12RRHS
- [54] Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltruaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, and Baining Guo. 2022. RODIN: A Generative Model for Sculpting 3D Digital Avatars Using Diffusion. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022), 4563-4573.
- [55] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. 2023. ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation. In Advances in Neural Information Processing Systems (NeurIPS).
- [56] Donglai Xiang, Fabian Prada, Timur Bagautdinov, Weipeng Xu, Yuan Dong, He Wen, Jessica Hodgins, and Chenglei Wu. 2021. Modeling clothing as a separate layer for an animatable human avatar. ACM Transactions on Graphics (TOG) 40,

6 (2021), 1-15.

- [57] Zhangyang Xiong, Di Kang, Derong Jin, Weikai Chen, Linchao Bao, Shuguang Cui, and Xiaoguang Han. 2023. Get3dhuman: Lifting stylegan-human into a 3d generative model using pixel-aligned reconstruction priors. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 9287-9297.
- [58] Tao Yu, Zerong Zheng, Kaiwen Guo, Pengpeng Liu, Qionghai Dai, and Yebin Liu. 2021. Function4D: Real-time Human Volumetric Capture from Very Sparse Consumer RGBD Sensors. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR2021).
- [59] Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, and Karsten Kreis. 2022. LION: Latent Point Diffusion Models for 3D Shape Generation. In Advances in Neural Information Processing Systems (NeurIPS).
- [60] Jianfeng Zhang, Zihang Jiang, Dingdong Yang, Hongyi Xu, Yichun Shi, Guoxian Song, Zhongcong Xu, Xinchao Wang, and Jiashi Feng. 2022. AvatarGen: A 3D Generative Model for Animatable Human Avatars. In Arxiv.
- [61] Xuanmeng Zhang, Jianfeng Zhang, Chacko Rohan, Hongyi Xu, Guoxian Song, Yi Yang, and Jiashi Feng. 2023. GETAvatar: Generative Textured Meshes for Animatable Human Avatars. In ICCV.