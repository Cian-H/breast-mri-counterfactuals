#!/bin/bash

# Set experiment name and timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="mri_diffusion_tanh_corrected_${TIMESTAMP}"

# Create output directory if it doesn't exist
mkdir -p out/models

# Run the training script with optimized dataset
python train_mri.py \
    --model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --hdf5_file "data/mama_mia_processed_tanh_corrected.h5" \
    --output_dir "out/models/tanh_corrected" \
    --resolution 512 \
    --train_batch_size 4 \
    --max_train_steps 10000000 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_weight_decay 1e-2 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 1.0 \
    --num_workers 6 \
    --save_steps 20000 \
    --log_images_every 1000 \
    --wandb \
    --wandb_project "mri-diffusion" \
    --wandb_run_name "${EXPERIMENT_NAME}" \
    --verbose \
    #Uncomment and set the path to resume from a checkpoint
    #--resume_from_checkpoint "out/models/checkpoint-90000"

# Print completion message
echo "Training completed for experiment: ${EXPERIMENT_NAME}"
echo "Model saved to: out/models/tanh_corrected"
echo "Used corrected tanh-normalized dataset with proper [-1,1] range" 
