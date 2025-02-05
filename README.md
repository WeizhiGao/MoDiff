# MoDiff
Implementation of MoDiff as an anoynous repo for ICML 2025 submission.

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