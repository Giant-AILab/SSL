# Self-Supervised Learning of Deviation in Latent Representation for Co-speech Gesture Video Generation

_Huan Yang*<sup>1</sup>, Jiahui Chen*<sup>1, 2</sup>, Chaofan Ding<sup>1</sup>, Runhua Shi<sup>1</sup>, Siyu Xiong<sup>1</sup>, Qingqi Hong<sup>2</sup>, Xiaoqi Mo<sup>1</sup>, Xinhan Di<sup>1</sup>_

1 Giant Interactive Group Inc & AI Lab, 2 Xiamen University

*Denotes Equal Contribution

in _ECCV 2024 workshop_

![image](https://github.com/user-attachments/assets/c49ae05a-b3f2-4ef8-8524-b43410e7fc69)

## Abstract

Gestures are pivotal in enhancing co-speech communication, while recent works have mostly focused on pointlevel motion transformation or fully supervised motion representations through data-driven approaches, we explore the representation of gestures in co-speech, with a focus on self-supervised representation and pixel-level motion deviation, utilizing aa diffusion model which incorporates latent motion features. Our approach leverages self-supervised deviation in latent representation to facilitate hand gestures generation, which are crucial for generating realistic gesture videos. Results of our first experiment demonstrate that our method enhances the quality of generated videos, with an improvement from 2.7 to 4.5% for FGD, DIV and FVD, and 8.1% for PSNR, 2.5% for SSIM over the current stateof-the-art methods.

## Pipeline

 Co-speech gesture video generation pipeline of our proposed method consists of three main components: 1) the latent deviation extractor (orange) 2) the latent deviation decoder (blue) 3) the latent motion diffusion (green).

We propose a novel method for generating co-speech gesture videos, utilizing a self-supervised full scene deviation, produces co-speech gesture video $V$ (i.e., image sequence) that exhibit natural poses and synchronized movements. The generation process takes as input the speaker’s speech audio a and a source image $I_S$.

We structured the training process into two stages. In the first stage, a driving image $I_D$ and a source image $I_S$ are used to train the base model. In one aspect, the proposed latent deviation module consisting of latent deviation extractor, warping calculator and latent deviation decoder is trained under self-supervision. In another aspect, other modules in the base model is trained under full supervision. In the second stage, the self-supervised motion features, consisting of $MF_i$, $\hat{MF}_{[i − 4, i − 1]}$, and the noiseadded motion feature sequence ${MF_j}$, are used to train the latent motion diffusion model.

![image](https://github.com/user-attachments/assets/5723b685-2fb8-4ecf-ab7c-309f83bb07b7)

## Generated video




## Visual comparison

![image](https://github.com/user-attachments/assets/9a7cfca4-d46b-4fc4-9df2-fd9d9e2aceed)

![image](https://github.com/user-attachments/assets/db1292fc-55db-4be7-ae59-74d8cdd86dfd)

## Generated video comparison

[![](https://res.cloudinary.com/marcomontalbano/image/upload/v1726822479/video_to_markdown/images/youtube--HPRfwyL4vMc-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=HPRfwyL4vMc&ab_channel=CaffeyChen "")

[![](https://res.cloudinary.com/marcomontalbano/image/upload/v1726822553/video_to_markdown/images/youtube--U8i7QRGOQGo-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=U8i7QRGOQGo&ab_channel=CaffeyChen "")

## Code
Coming soon...



