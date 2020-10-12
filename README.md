# Inside-Outside-Guidance (IOG) -- Demo
This project hosts the demo for the IOG algorithms for interactive segmentation.


> [Interactive Object Segmentation with Inside-Outside Guidance](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Interactive_Object_Segmentation_With_Inside-Outside_Guidance_CVPR_2020_paper.pdf)  
> Shiyin Zhang, Jun Hao Liew, Yunchao Wei, Shikui Wei, Yao Zhao  

### Installation
1. Install requirement  
  - PyTorch = 0.4
  - python >= 3.5
  - torchvision = 0.2
  - pycocotools

2. Run
```
python demo.py <input_img_path> <output_mask_path> 
```
- Click two dots (Upper Left and Lower Right corners)
- Click one dot on the object 
- Press "Enter"

### Pretrained models
| Dataset | Backbone |      Download Link        |
|---------|-------------|:-------------------------:|
|PASCAL + SBD  |  ResNet-101 |  [IOG_PASCAL_SBD.pth](https://drive.google.com/file/d/1Lm1hhMhhjjnNwO4Pf7SC6tXLayH2iH0l/view?usp=sharing)     |
|PASCAL |  ResNet-101   |  [IOG_PASCAL.pth](https://drive.google.com/file/d/1GLZIQlQ-3KUWaGTQ1g_InVcqesGfGcpW/view?usp=sharing)   |

### Dataset
With the annotated bounding boxes (∼0.615M) of ILSVRCLOC, we apply our IOG to collect their pixel-level annotations, named Pixel-ImageNet, which are publicly available at https://github.com/shiyinzhang/Pixel-ImageNet.

### Reference
https://github.com/shiyinzhang/Inside-Outside-Guidance
