export CUDA_VISIBLE_DEVICES=0

deepspeed --master_port=24999 train_ds.py \
  --version="liuhaotian/llava-llama-2-13b-chat-lightning-preview" \
  --dataset_dir='./dataset' \
  --vision_pretrained="downloads/sam.pth" \
  --dataset="sem_seg" \
  --sample_rates="9" \
  --exp_name="lisa-7b"



python -m debugpy --wait-for-client --listen 5678 -m deepspeed.launcher.runner --master_port=24999 train_ds.py \
  --version="liuhaotian/llava-llama-2-13b-chat-lightning-preview" \
  --dataset_dir='./dataset' \
  --vision_pretrained="downloads/sam.pth" \
  --dataset="sem_seg" \
  --sample_rates="9,3" \
  --exp_name="lisa-7b"

# for debugging, without deepspeed
python train.py  --version="liuhaotian/llava-llama-2-13b-chat-lightning-preview"   --dataset_dir="/scratch/pramish_paudel/LISA/dataset"   --vision_pretrained="downloads/sam.pth"   --dataset="sem_seg"   --sample_rates="9"   --exp_name="lisa-7b" --precision bf16


