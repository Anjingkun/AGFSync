export MODEL_NAME="models/stable-diffusion-xl-base-1.0"
export DATASET_NAME="dpo_dataset/post_sdxl_clip_aes_dataset_v1"

accelerate launch --main_process_port=29502 train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --enable_xformers_memory_efficient_attention \
  --resolution=1024\
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=1000 \
  --use_8bit_adam \
  --learning_rate=1e-06 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --checkpointing_steps=250 \
  --output_dir="dpo_clip_aes_sdxl_v3" \
  --good_image_column="good_jpg" \
  --bad_image_column="bad_jpg" \
  --caption_column="caption" \
  --max_grad_norm=1 \
  --resume_from_checkpoint="latest" \
  --sd_version="xl"
