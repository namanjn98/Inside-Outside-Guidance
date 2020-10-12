# Automatic Segmentation Tool
This project hosts the demonstration of Inside-Outside Guidance algorithm for interactive segmentation.

### Usage
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

### Reference
> https://github.com/shiyinzhang/Inside-Outside-Guidance
