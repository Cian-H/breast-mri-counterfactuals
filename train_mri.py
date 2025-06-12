import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import torch
from accelerate import Accelerator
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL
)
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
import argparse
import logging
from datetime import datetime
from dataset.optimized_mri_dataset import get_optimized_dataloaders
import wandb
import numpy as np

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
timestamp = datetime.now().strftime("%Y%m%d-%H%M")
file_handler = logging.FileHandler(f'out/finetune_{timestamp}.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def train(
    accelerator,
    unet,
    vae,
    text_encoder,
    noise_scheduler,
    dataloader,
    optimizer,
    args,
    global_step
):
    unet.train()
    total_loss = 0
    progress_bar = tqdm(range(args.max_train_steps), desc="Training")
    
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Get the image and text inputs
                images = batch["image"]
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                
                # Debug input images
                if accelerator.is_main_process and global_step % args.log_images_every == 0:
                    logger.info(f"Input image stats - min: {images.min().item():.3f}, max: {images.max().item():.3f}, mean: {images.mean().item():.3f}")
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(input_ids, attention_mask=attention_mask)[0]
                
                # Convert images to latent space
                vae_output = vae.encode(images)
                latents = vae_output.latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Debug latent space
                if accelerator.is_main_process and global_step % args.log_images_every == 0:
                    logger.info(f"Latent stats - min: {latents.min().item():.3f}, max: {latents.max().item():.3f}, mean: {latents.mean().item():.3f}")

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none").mean()
                
                # Debug loss and predictions
                if accelerator.is_main_process and global_step % args.log_images_every == 0:
                    logger.info(f"Loss: {loss.item():.6f}")
                    logger.info(f"Noise pred stats - min: {noise_pred.min().item():.3f}, max: {noise_pred.max().item():.3f}, mean: {noise_pred.mean().item():.3f}")
                    logger.info(f"Target noise stats - min: {noise.min().item():.3f}, max: {noise.max().item():.3f}, mean: {noise.mean().item():.3f}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # Debug print to verify this block is entered
                    logger.info("Gradient sync block entered")
                    
                    # Clip gradients
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    
                    # Calculate gradient norm after clipping
                    grad_norm = 0.0
                    for param in unet.parameters():
                        if param.grad is not None:
                            grad_norm += param.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    
                    # Debug gradients after clipping
                    if accelerator.is_main_process:
                        logger.info(f"Total gradient norm: {grad_norm:.6f}")
                    
                    # Log metrics to wandb
                    if accelerator.is_main_process:
                        wandb.log({
                            "train/loss": loss.detach().item(),
                            "train/grad_norm": grad_norm,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/step": global_step
                        })
                
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.detach().item()

            if global_step % args.save_steps == 0:
                save_progress(accelerator, unet, args.output_dir, global_step)

            progress_bar.update(1)
            global_step += 1
            
            # Log images periodically (after step increment for correct timing)
            if accelerator.is_main_process and global_step % args.log_images_every == 0:
                # Get efficient reconstruction for quality monitoring
                with torch.no_grad():
                    # Use a faster reconstruction method: single denoising step from mid-timestep
                    mid_timestep = noise_scheduler.config.num_train_timesteps // 2  # e.g., 500
                    mid_timestep_tensor = torch.tensor([mid_timestep], device=latents.device).expand(1)
                    
                    # Add noise to clean latents at mid-timestep
                    sample_noise = torch.randn_like(latents[:1])  # Just first image
                    noisy_sample = noise_scheduler.add_noise(latents[:1], sample_noise, mid_timestep_tensor)
                    
                    # Predict noise and reconstruct
                    noise_pred = unet(noisy_sample, mid_timestep_tensor, encoder_hidden_states[:1]).sample
                    
                    # Simple single-step reconstruction (not perfect but good for monitoring)
                    alpha_prod_t = noise_scheduler.alphas_cumprod[mid_timestep]
                    beta_prod_t = 1 - alpha_prod_t
                    reconstructed_latents = (noisy_sample - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
                    
                    # Decode to image space
                    reconstructed_images = vae.decode(reconstructed_latents / vae.config.scaling_factor).sample
                    
                    # Clamp to valid range
                    reconstructed_images = torch.clamp(reconstructed_images, -1, 1)
                    
                    wandb.log({
                        "images/original": wandb.Image(images[0].cpu()),
                        "images/reconstructed": wandb.Image(reconstructed_images[0].cpu()),
                        "images/step": global_step
                    })

            if global_step >= args.max_train_steps:
                break

    return total_loss / args.max_train_steps

def save_progress(accelerator, unet, output_dir, step):
    if accelerator.is_main_process:
        save_path = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)
        # Save only UNet weights
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(os.path.join(save_path, "unet"))

def main(args):
    # Initialize wandb
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
       # mixed_precision="fp16",
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Load models
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_name_or_path, subfolder="unet")

    # Load checkpoint if specified
    global_step = 0
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        unet = UNet2DConditionModel.from_pretrained(
            os.path.join(args.resume_from_checkpoint, "unet"),
            local_files_only=True
        )
        # Extract step number from checkpoint path
        try:
            global_step = int(os.path.basename(args.resume_from_checkpoint).split("-")[-1])
            logger.info(f"Resuming from step {global_step}")
        except:
            logger.warning("Could not determine step number from checkpoint path")

    # Move models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = unet.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Setup training data
    train_loader, test_loader = get_optimized_dataloaders(
        hdf5_file_path=args.hdf5_file,
        tokenizer=tokenizer,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        size=args.resolution,
        center_crop=args.center_crop,
        verbose=args.verbose
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Prepare for training
    unet, optimizer, train_loader = accelerator.prepare(
        unet, optimizer, train_loader
    )

    # Training info
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_loader.dataset)}")
    logger.info(f"  Max train steps = {args.max_train_steps}")
    logger.info(f"  Instantaneous batch size = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    if args.resume_from_checkpoint:
        logger.info(f"  Resuming from step {global_step}")

    # Training loop
    train_loss = train(
        accelerator,
        unet,
        vae,
        text_encoder,
        noise_scheduler,
        train_loader,
        optimizer,
        args,
        global_step
    )
    
    logger.info(f"Training completed. Average loss = {train_loss}")
    
    # Save final model (only UNet)
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(os.path.join(final_path, "unet"))
        logger.info(f"Final UNet weights saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion on MRI data")
    parser.add_argument("--model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--hdf5_file", type=str, required=True, help="Path to the HDF5 file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the model")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int, default=10000, help="Total number of training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop images")
    parser.add_argument("--verbose", action="store_true", help="Whether to print debug information")
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb logging")
    parser.add_argument("--wandb_project", type=str, default="mri-diffusion", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--log_images_every", type=int, default=100, help="Log images to wandb every N steps")

    args = parser.parse_args()
    main(args)