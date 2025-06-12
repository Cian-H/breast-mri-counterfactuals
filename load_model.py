import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import os

def load_trained_model(checkpoint_path, base_model_path="runwayml/stable-diffusion-v1-5"):
    # Load base models
    noise_scheduler = DDPMScheduler.from_pretrained(base_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder")
    
    # Load the trained UNet
    unet = UNet2DConditionModel.from_pretrained(os.path.join(checkpoint_path, "unet"))
    
    # Create the pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
    )
    
    # Move to GPU if available
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    pipeline = pipeline.to(device)
    
    return pipeline

def load_base_model(base_model_path="runwayml/stable-diffusion-v1-5"):
    # Create the pipeline with base model
    pipeline = StableDiffusionPipeline.from_pretrained(base_model_path)
    
    # Move to GPU if available
   # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    pipeline = pipeline.to(device)
    
    return pipeline

if __name__ == "__main__":
    # Example usage
    checkpoint_path = "/home/luab/MRI-diffusion/out/models/checkpoint-240000"
    prompt = "MRI showing normal tissue"
    
    # Generate with trained model
    print("Loading trained model...")
    trained_pipeline = load_trained_model(checkpoint_path)
    trained_image = trained_pipeline(prompt).images[0]
    trained_image.save("trained_generation.png")
    print(f"Generated image with trained model saved as trained_generation.png")
    
    # Generate with base model
    print("\nLoading base model...")
    base_pipeline = load_base_model()
    base_image = base_pipeline(prompt).images[0]
    base_image.save("base_generation.png")
    print(f"Generated image with base model saved as base_generation.png") 
