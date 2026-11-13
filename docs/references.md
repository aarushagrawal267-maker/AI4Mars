# References

## Dataset

1. R. Michael Swan, Deegan Atha, Henry A. Leopold, Matthew Gildner, Stephanie Oij, Cindy Chiu and
   Masahiro Ono. "AI4Mars: A Dataset for Terrain-Aware Autonomous Driving on Mars." In *Proceedings of
   the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops*, 2021,
   pp. 1982-1991.
2. Robert Swan, Masahiro Ono and Deegan Atha. *AI4Mars: A Dataset for Terrain-Aware Autonomous Driving
   on Mars*, version 0.6 [data set]. Zenodo, record 15995036. Licence CC-BY-4.0.
   https://doi.org/10.5281/zenodo.15995036
3. AI4Mars project on Zooniverse (source of the crowdsourced labels and of the labelling tutorials
   that define the classes). https://www.zooniverse.org/projects/hiro-ono/ai4mars
4. Mars Science Laboratory camera Software Interface Specification (MSL Camera SIS), referenced by the
   dataset's `info.md` for the meaning of MSL image file names.
   https://planetarydata.jpl.nasa.gov/img/data/msl/MSLNAV_1XXX/DOCUMENT/MSL_CAMERA_SIS.PDF

## Models

5. Olaf Ronneberger, Philipp Fischer and Thomas Brox. "U-Net: Convolutional Networks for Biomedical
   Image Segmentation." In *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*,
   2015. arXiv:1505.04597.
6. Liang-Chieh Chen, George Papandreou, Florian Schroff and Hartwig Adam. "Rethinking Atrous
   Convolution for Semantic Image Segmentation." arXiv:1706.05587, 2017.
7. Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang,
   Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le and Hartwig Adam. "Searching for
   MobileNetV3." In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*,
   2019. arXiv:1905.02244.
8. Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. "Deep Residual Learning for Image
   Recognition." In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
   (CVPR)*, 2016, pp. 770-778. arXiv:1512.03385.
9. Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar
   and C. Lawrence Zitnick. "Microsoft COCO: Common Objects in Context." In *European Conference on
   Computer Vision (ECCV)*, 2014. arXiv:1405.0312. (Source data of the pretrained DeepLabV3 weights.)
10. Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn and Andrew Zisserman. "The
    Pascal Visual Object Classes (VOC) Challenge." *International Journal of Computer Vision*, 88(2),
    2010, pp. 303-338. (Category set of the pretrained DeepLabV3 weights.)

## Training and evaluation

11. Diederik P. Kingma and Jimmy Ba. "Adam: A Method for Stochastic Optimization." In *International
    Conference on Learning Representations (ICLR)*, 2015. arXiv:1412.6980.
12. Ilya Loshchilov and Frank Hutter. "Decoupled Weight Decay Regularization." In *International
    Conference on Learning Representations (ICLR)*, 2019. arXiv:1711.05101. (AdamW.)
13. Fausto Milletari, Nassir Navab and Seyed-Ahmad Ahmadi. "V-Net: Fully Convolutional Neural Networks
    for Volumetric Medical Image Segmentation." In *Fourth International Conference on 3D Vision
    (3DV)*, 2016. arXiv:1606.04797. (Dice loss.)
14. Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger. "On Calibration of Modern Neural
    Networks." In *Proceedings of the 34th International Conference on Machine Learning (ICML)*,
    PMLR 70, 2017, pp. 1321-1330. arXiv:1706.04599. (Basis of the calibration caveat.)

## Route planning

15. Peter E. Hart, Nils J. Nilsson and Bertram Raphael. "A Formal Basis for the Heuristic
    Determination of Minimum Cost Paths." *IEEE Transactions on Systems Science and Cybernetics*,
    4(2), 1968, pp. 100-107. doi:10.1109/TSSC.1968.300136. (A* search.)

## Software

16. Adam Paszke et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." In
    *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019. https://pytorch.org
17. torchvision (models and pretrained segmentation weights): https://pytorch.org/vision/
18. FastAPI: https://fastapi.tiangolo.com - React: https://react.dev - Vite: https://vite.dev
