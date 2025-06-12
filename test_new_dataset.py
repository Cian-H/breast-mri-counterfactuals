#!/usr/bin/env python3
"""
Quick test script to verify the new optimized dataset works correctly.
Run this before full preprocessing to catch any issues early.
"""

import torch
from transformers import CLIPTokenizer
from dataset.optimized_mri_dataset import get_optimized_dataloaders
import numpy as np

def test_dataset():
    print("Testing optimized MRI dataset...")
    
    # Test paths
    hdf5_paths = [
        "data/mama_mia_processed_tanh_corrected.h5",  # New corrected version
        "data/mama_mia_processed_tanh_optimized.h5",  # Old version (for comparison)
        "data/mama_mia_processed.h5"  # Original version (for comparison)
    ]
    
    # Find available HDF5 file
    hdf5_path = None
    for path in hdf5_paths:
        import os
        if os.path.exists(path):
            hdf5_path = path
            print(f"Found HDF5 file: {path}")
            break
    
    if hdf5_path is None:
        print("❌ No HDF5 file found! Please run preprocessing first:")
        print("python dataset/preprocess_mama_mia_optimized.py")
        return False
    
    try:
        # Initialize tokenizer
        print("Loading tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Create data loaders
        print("Creating data loaders...")
        train_loader, test_loader = get_optimized_dataloaders(
            hdf5_file_path=hdf5_path,
            tokenizer=tokenizer,
            batch_size=2,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            size=512,
            center_crop=False,
            verbose=True
        )
        
        print(f"✅ Dataset loaded successfully!")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Test first batch
        print("\nTesting first batch...")
        for batch in train_loader:
            images = batch['image']
            masks = batch['mask']
            input_ids = batch['input_ids']
            has_seg = batch['has_segmentation']
            
            print(f"✅ Batch loaded successfully!")
            print(f"Image shape: {images.shape}")
            print(f"Image range: [{images.min().item():.6f}, {images.max().item():.6f}]")
            print(f"Mask shape: {masks.shape}")
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Has segmentation: {has_seg}")
            
            # Check if values are in expected range
            if "tanh_corrected" in hdf5_path:
                if images.min().item() >= -1.0 and images.max().item() <= 1.0:
                    print("✅ Image values are in correct tanh range (-1, 1)")
                else:
                    print(f"❌ Image values outside expected range: [{images.min().item():.6f}, {images.max().item():.6f}]")
                    return False
            
            # Test a few more batches
            break
        
        # Test access to patient info
        print("\nTesting patient info access...")
        train_dataset_base = train_loader.dataset.base_dataset
        first_patient = train_dataset_base.index[0][0]
        patient_info = train_dataset_base.get_patient_info(first_patient)
        print(f"✅ Patient info keys: {list(patient_info.keys())}")
        
        if 'valid_slices' in patient_info:
            print(f"✅ Validation info available: {len(patient_info['valid_slices'])} valid slices")
        else:
            print("⚠️  No validation info (using fallback mode)")
        
        print("\n🎉 All tests passed! Dataset is ready for training.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if 'train_loader' in locals():
            train_loader.dataset.base_dataset.close()
        if 'test_loader' in locals():
            test_loader.dataset.base_dataset.close()

if __name__ == "__main__":
    test_dataset() 