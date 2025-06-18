# latentdiffusion models
wget -O models/ldm/lsun_churches256/lsun_churches-256.zip https://ommer-lab.com/files/latent-diffusion/lsun_churches.zip
wget -O models/ldm/lsun_beds256/lsun_beds-256.zip https://ommer-lab.com/files/latent-diffusion/lsun_bedrooms.zip

cd ldm/lsun_churches256
unzip -o lsun_churches-256.zip

cd ../lsun_beds256
unzip -o lsun_beds-256.zip

cd ../..

# stable diffusion models
mkdir stable-diffusion-v1-4
cd stable-diffusion-v1-4
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt