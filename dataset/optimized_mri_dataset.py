import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import json
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import torchvision.transforms as transforms
from transformers import CLIPTokenizer
import psutil

class OptimizedMRIDataset(Dataset):
    """
    Memory-efficient HDF5-based MRI dataset with pre-validated slices.
    
    Key features:
    - Uses pre-processed HDF5 files for fast loading
    - Pre-validated slices (no empty/artifact slices)
    - Minimal memory footprint (only indices in RAM)
    - Supports slices with and without segmentation
    - Fast initialization (seconds instead of minutes)
    """
    
    def __init__(
        self,
        hdf5_file_path: str,
        split: str = 'train',
        size: int = 512,
        center_crop: bool = False,
        verbose: bool = False
    ):
        """
        Initialize dataset from pre-processed HDF5 file.
        
        Args:
            hdf5_file_path (str): Path to the HDF5 file created by preprocessing
            split (str): Either 'train' or 'test'
            size (int): Size to resize images to
            center_crop (bool): Whether to center crop images
            verbose (bool): Whether to print debug information
        """
        self.hdf5_file_path = hdf5_file_path
        self.split = split
        self.size = size
        self.center_crop = center_crop
        self.verbose = verbose
        
        # Open HDF5 file in read-only mode
        self.h5_file = h5py.File(hdf5_file_path, 'r')
        
        if split not in self.h5_file:
            raise ValueError(f"Split '{split}' not found in HDF5 file. Available: {list(self.h5_file.keys())}")
        
        # Build index of valid slices
        self.index, self.dataset_stats = self._build_validated_index()
        
        if len(self.index) == 0:
            raise ValueError(f"No valid slices found for split '{split}'")
        
        # Setup transforms
        self._setup_transforms()
        
        if verbose:
            self._print_dataset_info()
    
    def _build_validated_index(self) -> Tuple[List[Tuple[str, int]], Dict[str, Any]]:
        """Build index using only pre-validated slices."""
        index = []
        stats = {
            'total_patients': 0,
            'total_slices': 0,
            'valid_slices': 0,
            'slices_with_segmentation': 0,
            'slices_without_segmentation': 0,
            'patients_with_no_valid_slices': 0
        }
        
        split_group = self.h5_file[self.split]
        
        for patient_id in split_group.keys():
            patient_group = split_group[patient_id]
            stats['total_patients'] += 1
            
            # Get pre-validated slice indices
            if 'valid_slices' in patient_group:
                valid_slices = patient_group['valid_slices'][:]
                
                if len(valid_slices) == 0:
                    stats['patients_with_no_valid_slices'] += 1
                    continue
                
                # Add to index
                for slice_idx in valid_slices:
                    index.append((patient_id, int(slice_idx)))
                
                # Update statistics from stored attributes
                stats['valid_slices'] += len(valid_slices)
                stats['total_slices'] += patient_group.attrs.get('total_slices', len(valid_slices))
                stats['slices_with_segmentation'] += patient_group.attrs.get('slices_with_segmentation', 0)
                stats['slices_without_segmentation'] += patient_group.attrs.get('slices_without_segmentation', 0)
            else:
                # Fallback: if no validation was done, use all slices with basic check
                if self.verbose:
                    print(f"Warning: No validation data for patient {patient_id}, using all slices")
                
                volume_shape = patient_group['volume'].shape
                num_slices = volume_shape[-1] if len(volume_shape) == 3 else volume_shape[0]
                
                # Basic validation: skip constant slices
                for slice_idx in range(num_slices):
                    if len(volume_shape) == 3:
                        slice_data = patient_group['volume'][:, :, slice_idx]
                    else:
                        slice_data = patient_group['volume'][slice_idx, :, :]
                    
                    # Skip constant value slices
                    if not np.allclose(slice_data.min(), slice_data.max()):
                        index.append((patient_id, slice_idx))
                
                stats['valid_slices'] += len([idx for idx in range(num_slices) if (patient_id, idx) in index])
                stats['total_slices'] += num_slices
        
        return index, stats
    
    def _setup_transforms(self):
        """Setup image transforms."""
        if self.center_crop:
            self.transforms = transforms.Compose([
                transforms.Resize(self.size),
                transforms.CenterCrop(self.size)
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((self.size, self.size))
            ])
    
    def _print_dataset_info(self):
        """Print dataset statistics."""
        print(f"\n=== {self.split.upper()} Dataset Statistics ===")
        print(f"Patients: {self.dataset_stats['total_patients']}")
        print(f"Total slices: {self.dataset_stats['total_slices']}")
        print(f"Valid slices: {self.dataset_stats['valid_slices']}")
        print(f"Slices with segmentation: {self.dataset_stats['slices_with_segmentation']}")
        print(f"Slices without segmentation: {self.dataset_stats['slices_without_segmentation']}")
        print(f"Validation ratio: {self.dataset_stats['valid_slices'] / max(self.dataset_stats['total_slices'], 1):.3f}")
        if self.dataset_stats['patients_with_no_valid_slices'] > 0:
            print(f"Patients with no valid slices: {self.dataset_stats['patients_with_no_valid_slices']}")
        print(f"Memory usage: {get_memory_usage():.1f} MB")
        print("=" * 40)
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single slice with guaranteed validity."""
        patient_id, slice_idx = self.index[idx]
        
        # Direct HDF5 access - OS handles caching
        patient_group = self.h5_file[f'{self.split}/{patient_id}']
        
        # Get slice data (handle different volume orientations)
        volume = patient_group['volume']
        segmentation = patient_group['segmentation']
        
        if len(volume.shape) == 3:  # (H, W, D)
            img_data = volume[:, :, slice_idx]
            seg_slice = segmentation[:, :, slice_idx]
        else:  # (D, H, W)
            img_data = volume[slice_idx, :, :]
            seg_slice = segmentation[slice_idx, :, :]
        
        # Convert to tensor with RGB channels (required for RGB networks)
        img_tensor = torch.from_numpy(img_data.copy()).float()
        img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)  # Create RGB channels
        
        # Apply transforms
        img_tensor = self.transforms(img_tensor)
        
        # Convert segmentation to tensor
        seg_tensor = torch.from_numpy(seg_slice.copy()).float()
        seg_tensor = seg_tensor.unsqueeze(0)
        seg_tensor = self.transforms(seg_tensor)
        
        # Get patient metadata (cached access)
        metadata = {}
        for key in patient_group.attrs.keys():
            if not key.startswith('metadata_'):
                metadata[key] = patient_group.attrs[key]
        
        # Load additional metadata if present
        for key in patient_group.keys():
            if key.startswith('metadata_'):
                try:
                    metadata[key[9:]] = json.loads(patient_group[key][()].decode('utf-8'))
                except:
                    pass
        
        # Generate caption based on segmentation presence
        has_segmentation = torch.any(seg_tensor > 0).item()
        if has_segmentation:
            # Try to get tumor type from metadata
            tumor_type = "tumor"
            if 'primary_lesion' in metadata and 'tumor_subtype' in metadata['primary_lesion']:
                tumor_type = metadata['primary_lesion']['tumor_subtype']
            caption = f"MRI showing {tumor_type}"
        else:
            caption = "MRI showing normal tissue"
        
        return {
            'image': img_tensor,
            'mask': seg_tensor,
            'patient_id': patient_id,
            'slice_idx': slice_idx,
            'has_segmentation': has_segmentation,
            'caption': caption,
            'metadata': metadata
        }
    
    def get_patient_info(self, patient_id: str) -> Dict[str, Any]:
        """Get complete information for a specific patient."""
        if patient_id not in self.h5_file[self.split]:
            raise ValueError(f"Patient {patient_id} not found in {self.split} split")
        
        patient_group = self.h5_file[f'{self.split}/{patient_id}']
        
        info = {
            'patient_id': patient_id,
            'volume_shape': patient_group['volume'].shape,
            'segmentation_shape': patient_group['segmentation'].shape,
        }
        
        # Get validation info if available
        if 'valid_slices' in patient_group:
            info['valid_slices'] = patient_group['valid_slices'][:].tolist()
            if 'slice_stats' in patient_group:
                info['slice_stats'] = json.loads(patient_group['slice_stats'][()].decode('utf-8'))
        
        # Get attributes
        for key in patient_group.attrs.keys():
            info[key] = patient_group.attrs[key]
        
        return info
    
    def close(self):
        """Close HDF5 file."""
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class TokenizedMRIDataset(Dataset):
    """Wrapper dataset that handles tokenization for diffusion models."""
    
    def __init__(
        self,
        base_dataset: OptimizedMRIDataset,
        tokenizer: CLIPTokenizer,
        max_length: int = 77
    ):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get base sample
        sample = self.base_dataset[idx]
        
        # Tokenize caption
        text_inputs = self.tokenizer(
            sample['caption'],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Return only tensors for training
        return {
            'image': sample['image'],
            'mask': sample['mask'],
            'input_ids': text_inputs.input_ids.squeeze(0),
            'attention_mask': text_inputs.attention_mask.squeeze(0),
            'has_segmentation': torch.tensor(sample['has_segmentation'], dtype=torch.bool)
        }


def get_optimized_dataloaders(
    hdf5_file_path: str,
    tokenizer: CLIPTokenizer,
    batch_size: int = 1,
    num_workers: int = 4,
    size: int = 512,
    center_crop: bool = False,
    verbose: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create optimized data loaders from HDF5 file.
    
    Args:
        hdf5_file_path (str): Path to preprocessed HDF5 file
        tokenizer: CLIP tokenizer for text encoding
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        size (int): Image size
        center_crop (bool): Whether to center crop
        verbose (bool): Print dataset information
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    
    # Create base datasets
    train_dataset = OptimizedMRIDataset(
        hdf5_file_path=hdf5_file_path,
        split='train',
        size=size,
        center_crop=center_crop,
        verbose=verbose
    )
    
    test_dataset = OptimizedMRIDataset(
        hdf5_file_path=hdf5_file_path,
        split='test',
        size=size,
        center_crop=center_crop,
        verbose=verbose
    )
    
    # Wrap with tokenized datasets
    train_dataset = TokenizedMRIDataset(train_dataset, tokenizer)
    test_dataset = TokenizedMRIDataset(test_dataset, tokenizer)
    
    print(f"\nDataset loaded successfully:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    return train_loader, test_loader


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


# Example usage and testing
if __name__ == "__main__":
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    
    # Test with your HDF5 file
    hdf5_path = "data/mama_mia_processed_tanh_corrected.h5"
    
    if os.path.exists(hdf5_path):
        print(f"\nTesting optimized dataset with {hdf5_path}")
        
        # Initialize tokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Create data loaders
        train_loader, test_loader = get_optimized_dataloaders(
            hdf5_file_path=hdf5_path,
            tokenizer=tokenizer,
            batch_size=2,
            num_workers=2,
            size=512,
            center_crop=False,
            verbose=True
        )
        
        print(f"\nAfter dataset creation: {get_memory_usage():.1f} MB")
        
        # Test first batch
        print("\nTesting first batch...")
        for batch in train_loader:
            print(f"Image batch shape: {batch['image'].shape}")
            print(f"Mask batch shape: {batch['mask'].shape}")
            print(f"Input IDs shape: {batch['input_ids'].shape}")
            print(f"Has segmentation: {batch['has_segmentation']}")
            print(f"Image range: [{batch['image'].min().item():.3f}, {batch['image'].max().item():.3f}]")
            break
        
        print(f"\nFinal memory usage: {get_memory_usage():.1f} MB")
        
        # Test patient info access
        train_dataset_base = train_loader.dataset.base_dataset
        first_patient = train_dataset_base.index[0][0]
        patient_info = train_dataset_base.get_patient_info(first_patient)
        print(f"\nSample patient info keys: {list(patient_info.keys())}")
        
    else:
        print(f"HDF5 file not found: {hdf5_path}")
        print("Please run the preprocessing script first:")
        print("python dataset/preprocess_mama_mia_optimized.py") 