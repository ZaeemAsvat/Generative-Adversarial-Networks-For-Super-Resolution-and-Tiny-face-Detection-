# Generative Adversarial Networks For Super-Resolution and Tiny face Detection
 
An Honours Research Project being conducted at The University of Witwatersrand, South Africa that explores the use of Generative Adversarial Networks (GANs) for the task of single-image super resolution (SISR) and Tiny Face Detection in The wild. 

Our generator network largely follows that of Residual Channel Attention Network (RCAN), as proposed in [1], which incorporates Residual-In-Residual architecture and Channel Attention mechanisms. 

Our discriminatior network will be trained to distinguish whether images produced by the generator are real or fake, and whether they are faces or non-faces, as was similarly achieved in [2]

This project is being implemented in PyTorch and the WIDER-FACE dataset [3] will be used for training and evaluation.

------------------------------------------------------------------------------------------------
[1] [Zhang et al. 2018a] Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong,
and Yun Fu. Image super-resolution using very deep residual channel attention
networks. In The European Conference on Computer Vision (ECCV), September
2018.

[2] [Bai et al. 2018a] Yancheng Bai, Yongqiang Zhang, Mingli Ding, and Bernard Ghanem.
Finding tiny faces in the wild with generative adversarial network. In The IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

[3] [Yang et al. 2015] Shuo Yang, Ping Luo, Chen Change Loy, and Xiaoou Tang. WIDER
FACE: A face detection benchmark. CoRR, abs/1511.06523, 2015.
