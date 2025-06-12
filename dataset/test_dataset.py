import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from mama_mia import MRIDiffusionDataset, get_dataloaders

def test_dataset():
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    # Setup paths relative to project root
    data_dir = project_root / "data/images"
    segmentation_dir = project_root / "data/segmentations"
    split_file = project_root / "data/train_test_splits.csv"
    patient_info_dir = project_root / "data/patient_info_files"
    
    print("Initializing dataset...")
    print(f"Data directory: {data_dir}")
    print(f"Segmentation directory: {segmentation_dir}")
    print(f"Split file: {split_file}")
    print(f"Patient info directory: {patient_info_dir}")
    
    # Test dataloader with focus on problematic region
    print("\nTesting dataloader around batch 11...")
    batch_size = 8
    train_loader, _ = get_dataloaders(
        data_dir=str(data_dir),
        segmentation_dir=str(segmentation_dir),
        split_file=str(split_file),
        patient_info_dir=str(patient_info_dir),
        batch_size=batch_size,
        num_workers=0,  # Use 0 for debugging
        segmentation_type='expert',
        size=512,
        center_crop=False,
        verbose=True
    )
    
    # Test individual samples in the problematic region
    start_idx = (11 - 1) * batch_size  # Start from items in batch 10
    end_idx = (11 + 2) * batch_size    # Go until items in batch 12
    
    dataset = train_loader.dataset
    print(f"\nTesting individual samples from index {start_idx} to {end_idx}...")
    
    for idx in range(start_idx, end_idx):
        try:
            sample = dataset[idx]
            print(f"\nSample {idx}:")
            print(f"Patient ID: {sample['patient_id']}")
            print(f"Image shape: {sample['image'].shape}")
            print(f"Mask shape: {sample['mask'].shape}")
            print(f"Caption: {sample['caption']}")
            
            # Additional debugging info
            img_path = dataset.image_paths[idx]
            seg_path = dataset.segmentation_paths[idx]
            slice_idx = dataset.slice_indices[idx]
            print(f"Image path: {img_path}")
            print(f"Segmentation path: {seg_path}")
            print(f"Slice index: {slice_idx}")
            
        except Exception as e:
            print(f"\nError in sample {idx}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            
            # Print relevant paths even if there's an error
            try:
                print(f"Image path: {dataset.image_paths[idx]}")
                print(f"Segmentation path: {dataset.segmentation_paths[idx]}")
                print(f"Slice index: {dataset.slice_indices[idx]}")
            except Exception as path_error:
                print(f"Could not access paths: {str(path_error)}")
            
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
    
    print("\nTesting batches 10-12...")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx < 10:  # Skip until batch 10
            continue
        if batch_idx > 12:  # Stop after batch 12
            break
            
        print(f"\nBatch {batch_idx}:")
        try:
            print(f"Image batch shape: {batch['image'].shape}")
            print(f"Mask batch shape: {batch['mask'].shape}")
            print(f"Patient IDs: {batch['patient_id']}")
            print(f"Captions: {batch['caption']}")
        except Exception as e:
            print(f"Error in batch {batch_idx}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())

if __name__ == "__main__":
    test_dataset() 