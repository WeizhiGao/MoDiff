# Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

This repository provides the official implementation of our ICML 2025 paper, [Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization](https://icml.cc/virtual/2025/poster/43551), along with the pretrained checkpoints and the calibrated dataset used for Q-Diffusion. MoDiff is designed to be compatible with any post-training quantization (PTQ) method, enabling lower activation precision without compromising generation quality.

![Example output on LSUN Church dataset](assets/example_church.png]

## Overview
Diffusion models have emerged as powerful generative models, but their high computation cost in iterative sampling remains a significant bottleneck. In this work, we present an in-depth and insightful study of state-of-the-art acceleration techniques for diffusion models, including caching and quantization, revealing their limitations in computation error and generation quality. To break these limits, this work introduces Modulated Diffusion (MoDiff), an innovative, rigorous, and principled framework that accelerates generative modeling through modulated quantization and error compensation. MoDiff not only inherents the advantages of existing caching and quantization methods but also serves as a general framework to accelerate all diffusion models. The advantages of MoDiff are supported by solid theoretical insight and analysis. In addition, extensive experiments on CIFAR-10 and LSUN demonstrate that MoDiff significant reduces activation quantization from 8 bits to 3 bits without performance degradation in post-training quantization (PTQ).

## Checkpoints Downloading
- Download the corresponding pretrained models in https://ommer-lab.com/files/latent-diffusion, and put them into the corresponding folder in ```models/ldm/*```
- Download the quantized checkpoints in Q-Diffusion in https://drive.google.com/drive/folders/1ImRbmAvzCsU6AOaXbIeI7-4Gu2_Scc-X to ```quantized_models```

## Reproduce
To run the experiments with dynamic quantization, please use the following command:
```
python scripts/sample_diffusion_ddim.py --config configs/cifar10.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq --weight_bit 4 --quant_mode qdiff --quant_act --act_bit 8 --a_sym --split --resume -l log/data/w4a8 --cali_ckpt quantized_models/cifar_w4a8_ckpt.pth --rt --ds --warm 
```

To run the experiments with Q-diff, please first generate calibration dataset with the following command:
```
python scripts/sample_diffusion_ddim.py --config configs/cifar10.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad --generate --residual -l logs/cifar10/test --cali_n 2048 --cali_st 20
```

Then run the code to calibarte the model and generate images with the following command:
```
CUDA_VISIBLE_DEVICES=1 python scripts/sample_diffusion_ddim.py --config configs/cifar10.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq --weight_bit 4 --quant_mode qdiff --cali_st 20 --cali_batch_size 32 --cali_n 256 --quant_act --act_bit 8 --a_sym --split --cali_data_path logs/cifar10/test/samples/cali_data.pt  -l logs/qdiff/w4a8  --cali_ckpt quantized_models/cifar_w4a8_ckpt.pth --resume_w --ds --warm
```