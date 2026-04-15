# LiSTMNet
Lightweight Spatial-Temporal Mamba Network for Medical Video Segmentation, historical version.

## DATA
* Please download datasets from
[CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/), 
[EchoNet-Dynamic](https://echonet.github.io/dynamic/), 
[VPS](https://drive.google.com/file/d/1TyaRy4c4nHFDa3o2bOl4dP5Z7wes7HV2/view).

* Please follow [MemSAM](https://github.com/dengxl0520/MemSAM) for cardiac ultrasound video segmentation data (CAMUS, EchoNet-Dynamic).
* Please follow the data format and storage requirements from [SALI](https://github.com/Scatteredrain/SALI) for video polyp segmentation (VPS).


## ENVIRONMENT
* NVIDIA RTX 3090 + Ubuntu 22.04 + CUDA 12.4
* Python 3.12 + PyTorch 2.5 + TorchVision 0.20.0


## INSTALLATION
* Please follow [Vivim](https://github.com/scott-yjyang/Vivim) to install casual-conv1d and mamba.
```bash
git clone https://github.com/scott-yjyang/Vivim.git
pip install ninja
cd causal-conv1d python setup.py install
cd mamba python setup.py install
pip uninstall transformers
pip install transformers==4.35.2
```

* Please follow [MemSAM](https://github.com/dengxl0520/MemSAM) and [SALI](https://github.com/Scatteredrain/SALI) to install other necessary requirements.
```bash
git clone https://github.com/dengxl0520/MemSAM.git
pip install requirements.txt
git clone https://github.com/Scatteredrain/SALI.git
pip install thop medpy ...
```

* Some packages may not be mentioned, please install them according to the error message. XD


## CHECKPOINTS
* If you want to use TransUNet in this repository, please follow [TransUNet](https://github.com/Beckschen/TransUNet) to get the ViT checkpoints.
```
Checkpoint Path: ../configs/transunet/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

* If you want to use SegFormer or Vivim in this repository, please download pretrianed SegFormer model from [huggingface](https://huggingface.co/nvidia/segformer-b3-finetuned-ade-512-512/tree/main).
```
Checkpoint Path: ../segformer_b3
```


## ACKNOWLEDGEMENT
The work is based on 
[MemSAM](https://github.com/dengxl0520/MemSAM),
[SALI](https://github.com/Scatteredrain/SALI),
[Vivim](https://github.com/scott-yjyang/Vivim).
Thanks for their open source contributions.
