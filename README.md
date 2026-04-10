# CoreKD: Adaptive Core-aware Knowledge Distillation for Tiny Object Detection in Remote Sensing Imagery

## 📢 Important Notice

The complete source code will be fully open-sourced immediately upon the paper's acceptance.

## Introduction
**Abstract**: Minimizing satellite on-orbit interpretation latency is paramount in remote sensing. Due to the limited computational resources of edge devices, Knowledge Distillation (KD) serves as a natural solution for reducing inference costs through distilling powerful models into lightweight ones. However, traditional rigid feature imitation is suboptimal for tiny-sized object distillation in remote sensing. The difficulty arises from two aspects: successive downsampling attenuates the already weak signals of tiny objects; the structural gap between powerful and lightweight frameworks causes an intrinsic activation shift, making rigid feature matching ineffective for supervision. Consequently, we propose the adaptive core-aware knowledge distillation for tiny-sized object detection in remote sensing scene (CoreKD). Our method integrates three pivotal submodules: (i) Adaptive Core Feature Imitation (ACFI), which introduces a size-aware Gaussian prior to decouple sparse instance features from the background noise, guiding the student framework to focus on tiny-sized instance cores; (ii) Non-Rigid Distillation (NRD), which learns sampling offsets to geometrically rectify the student's receptive field, resolving the spatial feature misalignment induced by structural asymmetry; (iii) Foreground-Guided Relation Distillation (FGRD), which captures the global contextual dependencies to strengthen the student's robustness in complex scenarios. Extensive experiments confirm that CoreKD delivers consistent performance improvements for student models on the challenging DIOR, AI-TOD, SODA-D, and SSDD datasets. For the deployment-friendly lightweight backbones, our CoreKD boosts the student's performance by $\sim$ 7.0 AP while introducing zero extra inference cost.

![fig](https://i.postimg.cc/rFzbmRzx/overview.jpg)

## Installation
### Required environments:
-  Linux
-  Python **3.8** or higher
-   PyTorch **1.8**  or higher
-   CUDA **11.1** or higher
-   GCC(G++)  **5.4**  or higher
-   [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
-   [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)==**12.0.3**

### Install:
This project is implemented based on the [MMDetection](https://github.com/open-mmlab/mmdetection) toolkit. Once your environment has met the above requirements, follow the steps below to install.
```c
git clone 
cd corekd
pip install -r requirements/build.txt
python setup.py develop
```

## Get Started

### Prepare datasets
Please refer to [AI-TOD](https://github.com/jwwangchn/AI-TOD) for the AI-TOD dataset.
If your folder structure is different, you may need to change the corresponding paths in config files.
```
├── mmdet-kldet
├── tools
├── configs
├── data
│   ├── AI-TOD
│   │   ├── annotations
│   │   │    │─── aitod_training_v1.json
│   │   │    │─── aitod_validation_v1.json
│   │   ├── trainval
│   │   │    │─── ***.png
│   │   │    │─── ***.png
│   │   ├── test
│   │   │    │─── ***.png
│   │   │    │─── ***.png
```


## Visualization
Some representative detection results of CoreKD:
![](https://i.postimg.cc/3wrfsyMj/results.jpg)
