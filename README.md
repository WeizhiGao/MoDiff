# Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

This repository provides the official implementation of our ICML 2025 paper, [Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization](https://icml.cc/virtual/2025/poster/43551), along with the pretrained checkpoints and the calibrated dataset used for Q-Diffusion. MoDiff is designed to be compatible with any post-training quantization (PTQ) method, enabling lower activation precision (up to 3 bits) without compromising generation quality. 

![Example output on LSUN Church dataset](assets/example_church.png)

Our MoDiff scales to Stable Diffusion on text-guided image generation. Here is an example on COCO-MS-2014.

![Example output on MS-COCO dataset](assets/example_coco.png)

## Overview
Diffusion models have emerged as powerful generative models, but their high computation cost in iterative sampling remains a significant bottleneck. In this work, we present an in-depth and insightful study of state-of-the-art acceleration techniques for diffusion models, including caching and quantization, revealing their limitations in computation error and generation quality. To break these limits, this work introduces Modulated Diffusion (MoDiff), an innovative, rigorous, and principled framework that accelerates generative modeling through modulated quantization and error compensation. MoDiff not only inherents the advantages of existing caching and quantization methods but also serves as a general framework to accelerate all diffusion models. The advantages of MoDiff are supported by solid theoretical insight and analysis. In addition, extensive experiments on CIFAR-10 and LSUN demonstrate that MoDiff significant reduces activation quantization from 8 bits to 3 bits without performance degradation in post-training quantization (PTQ).

![Method illustration](assets/modiff.png)

## Get Started

### Installation
Clone the MoDiff repository and create a conda environment ` modiff ` with the following commands:

```
git clone https://github.com/WeizhiGao/MoDiff.git
cd MoDiff
conda env create -f environment.yml
conda activate modiff
```

If you encounter errors while installing the environment, please manually install the packages with compatible versions.

### Pretrained Model Preparation
Before the quantization, you need to prepare the pre-trained models with the following instructions:

1. Specify the model directory and move ` models ` directory there.

2. For DDIM on CIFAR10, the pre-trained model will be automatically downloaded. You are able to change the saving path in the input args.

3. For Latent Diffusion and Stable Diffusion experiments, we follow the checkpoints in [latent-diffusion](https://github.com/CompVis/latent-diffusion#model-zoo) and [stable-diffusion](https://github.com/CompVis/stable-diffusion#weights) . We use ` LDM-8 ` for LSUN-Churches, ` LDM-4 ` for LSUN-Bedrooms, and ` sd-v1-1.4 ` for MS-COCO. We provide a script for automatically download:
    ```
    cd <model_path>/models
    sh download.sh
    ```

4. Our work targets the activation quantization, and you can use the quantized models weight for any quantization. You should download your quantized model weights to ` quant_models ` directory. We use the checkpoints in [Q-Diffusion](https://github.com/Xiuyu-Li/q-diffusion) and you can download those models from [Google Drive](https://drive.google.com/drive/folders/1ImRbmAvzCsU6AOaXbIeI7-4Gu2_Scc-X).

### Calibration Dataset Generation
If you use dynamic quantization, you can skip this step. For some quantization methods, calibration dataset is required. We provide scipts to generate calibration dataset of CIFAR-10 and LSUN-Churches for MoDiff as follows:

```
# CIFAR10
python scripts/sample_diffusion_ddim.py --config configs/cifar10.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad \
 --logdir <logdir> --generate residual --cali_n 256 --cali_st 20 --cali_data_path cali_data/cifar10.pt 

# LSUN-Churches
python scripts/sample_diffusion_ldm.py -r <path>/models/ldm/lsun_churches256/model.ckpt --batch_size 32 -c 400 -e 0.0 --seed 42 \
 -l <logdir> --generate residual --cali_n 256 --cali_st 20 --cali_data_path cali_data/church.pt 
```

You can apply the script to other datasets. In practice, we only generate 256 data for each timestep, which cost several minutes on one H100 GPU. We also provide well-generated calibration data in [Hugging Face](https://huggingface.co/datasets/Weizhi98/MoDiff/tree/main/cali_data).

### Post-Training Quantization
1. For dynamic quantization, you do not need the calibration dataset. Reproduce the results of our paper with the following code. You can control the usage of modulated quantization by ` --modulate `, which can allow 3-bit activation as the limit:
    ```
    # CIFAR10
    python scripts/sample_diffusion_ddim.py --config configs/cifar10.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq \
     --weight_bit 4 --quant_act --act_bit 4 --a_sym --split --resume -l <logdir> --cali_ckpt $path/quant_models/cifar_w4a8_ckpt.pth \
     --modulate --quant_mode dynamic --max_images 50000

    # LSUN-Churches
    python scripts/sample_diffusion_ldm.py -r $path/models/ldm/lsun_churches256/model.ckpt --batch_size 8 -c 400 -e 0.0 --seed 41 --ptq \
     --weight_bit 4 --quant_act --act_bit 4 --resume -l <logdir> --cali_ckpt $path/quant_models/church_w4a8_ckpt.pth \
     --modulate --quant_mode dynamic -n 50000

    # LSUN-Bedrooms
    python scripts/sample_diffusion_ldm.py -r $path/models/ldm/lsun_beds256/model.ckpt --batch_size 8 -c 200 -e 0.0 --seed 41 --ptq \
     --weight_bit 4 --quant_act --act_bit 4 --resume -l <logdir> --cali_ckpt $path/quant_models/bedroom_w4a8_ckpt.pth \
     --modulate --quant_mode dynamic -n 50000

    # Text-Guided with Stable Diffusion
    python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms --cond --n_iter 1 --n_samples 5 --skip_grid --ptq \
     --weight_bit 8 --quant_act --act_bit 4 --no_grad_ckpt --split --running_stat --outdir <logdir> \
     --ckpt $path/models/stable-diffusion-v1/model.ckpt --resume --cali_ckpt $path/quantized_models/sd_w8a8_ckpt.pth \
     --modulate --quant_mode dynamic --act_tensor

    ```
    **Note:** You can enable the tensor wise activation quantization with ` --act_tensor `. For complete text-guided generation results on Stable Diffusion, please download the annotation of [MSCOCO-2014](https://cocodataset.org/#download) and specify `--from-file` augments. You can also download the annotation from [Huggging Face](https://huggingface.co/datasets/Weizhi98/MoDiff/tree/main/annotation/MS-COCO).

2. For Q-Diffusion, please first prepare the calibration dataset following the last step. We recommend to use the min-max initalization, which is both data-efficient and computation-efficient, resulting in comparable results compared to MSE calibration. You can use only 32 calibrated data for each time steps as follows:
    ```
    # CIFAR10
    python scripts/sample_diffusion_ddim.py --config configs/cifar10.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq \
     --weight_bit 4 --cali_st 20 --cali_batch_size 32 --cali_n 32 --quant_act --act_bit 4 --a_sym --split \
     --cali_data_path $path/cali_data/cifar10.pt -l <logdir> --cali_ckpt $path/quant_models/cifar_w4a8_ckpt.pth --resume_w \
     --modulate --quant_mode qdiff --cali_min_max --max_image 50000

    # LSUN-Churches
    python scripts/sample_diffusion_ldm.py -r $path/models/ldm/lsun_churches256/model.ckpt --batch_size 8 -c 400 -e 0.0 --seed 42 \
     --ptq --weight_bit 4 --cali_st 20 --cali_batch_size 32 --cali_n 32 --quant_act --act_bit 4 \
     --cali_data_path $path/cali_data/church.pt -l <logdir> --cali_ckpt $path/quantized_models/church_w4a8_ckpt.pth --resume_w \
     --modulate --quant_mode qdiff --cali_min_max -n 50000
    ```

3. For MSE calibration of Q-Diffusion, you can reproduce with the following code:
    ```
    # CIFAR10
    python scripts/sample_diffusion_ddim.py --config configs/cifar10.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq \
     --weight_bit 4 --cali_st 20 --cali_batch_size 32 --cali_n 256 --quant_act --act_bit 4 --a_sym --split \
     --cali_data_path $path/cali_data/cifar10.pt  -l <logdir> --cali_ckpt $path/quant_models/cifar_w4a8_ckpt.pth --resume_w \
     --modulate --quant_mode qdiff --max_image 50000

    # LSUN-Churches
    python scripts/sample_diffusion_ldm.py -r $path/models/ldm/lsun_churches256/model.ckpt --batch_size 8 -c 400 -e 0.0 --seed 42 \
     --ptq --weight_bit 4 --cali_st 20 --cali_batch_size 32 --cali_n 256 --quant_act --act_bit 4 \
     --cali_data_path $path/cali_data/church.pt -l logdir> --cali_ckpt $path/quantized_models/church_w4a8_ckpt.pth --resume_w \
     --modulate --quant_mode qdiff -n 50000
    ```
    **Note:** You can tune the calibration learning rate and the outlier penalty with `--cali_lr` and `--out_penalty` for better generation quality.

### Image Generation
You can generate images from the quantized checkpoints. For dynamic quantization, please refer the [Dynamic Quantization](#post-training-quantization). For Q-Diffusion checkpoints, use the following commands to generate images:
```
# CIFAR10
python scripts/sample_diffusion_ddim.py --config configs/cifar10.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq \
 --weight_bit 8 --quant_act --act_bit 4 --a_sym --split --resume -l <logdir> --cali_ckpt <ckpt_dir> \
 --modulate --quant_mode qdiff --max_images 50000

# LSUN-Churches
python scripts/sample_diffusion_ldm.py -r $path/models/ldm/lsun_churches256/model.ckpt --batch_size 64 -c 400 -e 0.0 --seed 41 \
 --ptq --weight_bit 4 --quant_act --act_bit 4 --resume -l <logdir> --cali_ckpt <ckpt_dir> \
 --modulate --quant_mode qdiff -n 50000 
```

### Evaluation
For the evaluation of IS, FID, sFID, we follow [torch-fidelity](https://github.com/toshas/torch-fidelity) and [guided-diffusion](https://github.com/openai/guided-diffusion). We generate 50,000 images for CIFAR-10 and LSUN datasets. 

1. Evaluate IS and FID scores with torch-fidelity. Before evlautaion, you need to download the reference datasets. The cifar-10 reference data will be downladed automatically, and you can obtain the LSUN datasets [here](https://github.com/fyu/lsun). Then run the following command to gain the IS score and FID score:
    
    ```
    # CIFAR10
    fidelity --gpu 0 --isc --fid --input1 <generated_image_path> --input2 cifar10-train

    # LSUN
    fidelity --gpu 0 --fid --input1 <generated_image_path> --input2 <reference_image_path>
    ```
2. Evaluate FID and sFID scores with guided-diffusion. Please follow the [requirements](https://github.com/openai/guided-diffusion/blob/main/evaluations/requirements.txt) to install a TensorFlow-based environment. After installing the environment, you need to compress your reference dataset as `.npz` files. We provided the files of CIFAR10 and LSUN datasets in [Hugging Face](https://huggingface.co/datasets/Weizhi98/MoDiff/tree/main/reference_data). For LSUN generation, it defaultly saves the '.npz' files. For the generated'.npz' file of CIFAR-10, you can collect it during sampling or post process the generated '.png' files. Then run the following command to gain the IS score and FID score:
    ```
    python scripts/evaluate.py <reference_npz_file> <generated_npz_file>
    ```

## Citation
If you find this work helpful in your usage, please consider citing our paper:
```
TBD
```

## Acknowledgements
We appreciate the availability of well-maintained public codebases. Our diffusion model code is developed based on [q-diffusion](https://github.com/Xiuyu-Li/q-diffusion), [ddim](https://github.com/ermongroup/ddim), [latent-diffusion](https://github.com/CompVis/latent-diffusion), and [stable-diffusion](https://github.com/CompVis/latent-diffusion). If you find any bugs in this repo, feel free to contact my through email or put them in [Issues](https://github.com/WeizhiGao/MoDiff/issues).

We thank [DeepSpeed](https://github.com/microsoft/DeepSpeed) for model sizes and BOPS measurements, [torch-fidelity](https://github.com/toshas/torch-fidelity) and [guided-diffusion](https://github.com/openai/guided-diffusion) for IS, FID and sFID evaluation, and [LSUN]((https://github.com/fyu/lsun)) for providing datasets. 
