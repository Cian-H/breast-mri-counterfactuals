import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torchvision.transforms as transforms
from transformers import CLIPTokenizer
import json

class MRIDiffusionHDF5Dataset(Dataset):
    def __init__(
        self,
        hdf5_file: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        size: int = 512,
        center_crop: bool = False,
        verbose: bool = False
    ):
        """
        Dataset for MRI diffusion data loaded from HDF5 file.
        
        Args:
            hdf5_file (str): Path to the HDF5 file
            split (str): Either 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on the images
            size (int): Size to resize images to
            center_crop (bool): Whether to center crop images
            verbose (bool): Whether to print debug information
        """
        self.hdf5_file = hdf5_file
        self.split = split
        self.transform = transform
        self.size = size
        self.center_crop = center_crop
        self.verbose = verbose
        
        # Create index mapping
        self.index = []
        skipped_constant = 0
        
        with h5py.File(hdf5_file, 'r') as f:
            group = f[split]
            for patient_id in group.keys():
                patient_group = group[patient_id]
                volume = patient_group['volume']
                num_slices = volume.shape[-1]
                for slice_idx in range(num_slices):
                    # Check if slice has constant values
                    img_slice = volume[:, :, slice_idx]
                    if np.allclose(img_slice.min(), img_slice.max()):
                        skipped_constant += 1
                        if verbose:
                            print(f"Skipping constant value slice: {patient_id} slice {slice_idx}")
                        continue
                    
                    self.index.append((patient_id, slice_idx))
        
        if len(self.index) == 0:
            raise ValueError(f"No valid samples found for split '{split}'")
        
        print(f"Loaded {len(self.index)} valid slices for {split} split")
        print(f"Skipped {skipped_constant} constant value slices")
        
        # Setup transforms
        if center_crop:
            self.transforms = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).float().permute(2, 0, 1))
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).float().permute(2, 0, 1))
            ])
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patient_id, slice_idx = self.index[idx]
        
        with h5py.File(self.hdf5_file, 'r') as f:
            patient_group = f[self.split][patient_id]
            
            # Get volume and segmentation slices
            img_data = patient_group['volume'][:, :, slice_idx]
            seg_slice = patient_group['segmentation'][:, :, slice_idx]
            
            # Get metadata
            metadata = {}
            for key, value in patient_group.attrs.items():
                metadata[key] = value
            
            # Get additional metadata from datasets
            for key in patient_group.keys():
                if key.startswith('metadata_'):
                    metadata[key[9:]] = json.loads(patient_group[key][()])
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_data).float()
        img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)  # Add channel dimension and repeat for RGB
        
        # Resize if needed
        if self.center_crop:
            img_tensor = transforms.Resize(self.size)(img_tensor)
            img_tensor = transforms.CenterCrop(self.size)(img_tensor)
        else:
            img_tensor = transforms.Resize((self.size, self.size))(img_tensor)
        
        # Normalize to [-1, 1] range
        img_min = img_tensor.min()
        img_max = img_tensor.max()
        img_tensor = 2 * (img_tensor - img_min) / (img_max - img_min) - 1
        
        # Convert segmentation to tensor
        seg_tensor = torch.from_numpy(seg_slice).float()
        seg_tensor = seg_tensor.unsqueeze(0)
        if self.center_crop:
            seg_tensor = transforms.Resize(self.size)(seg_tensor)
            seg_tensor = transforms.CenterCrop(self.size)(seg_tensor)
        else:
            seg_tensor = transforms.Resize((self.size, self.size))(seg_tensor)
        
        # Generate caption
        caption = "MRI showing normal tissue"
        if 'primary_lesion' in metadata:
            lesion = metadata['primary_lesion']
            if 'breast_coordinates' in lesion and 'tumor_subtype' in lesion:
                coords = lesion['breast_coordinates']
                if coords['z_min'] <= slice_idx <= coords['z_max'] and torch.any(seg_tensor > 0):
                    tumor_type = lesion['tumor_subtype']
                    caption = f"MRI showing {tumor_type} tumor"
        
        return {
            'image': img_tensor,
            'mask': seg_tensor,
            'patient_id': patient_id,
            'clinical_data': metadata.get('clinical_data', {}),
            'imaging_data': metadata.get('imaging_data', {}),
            'primary_lesion': metadata.get('primary_lesion', {}),
            'caption': caption,
            'slice_idx': slice_idx
        }

class TokenizedMRIDataset(Dataset):
    """Wrapper dataset that handles tokenization and ensures tensor outputs."""
    def __init__(
        self,
        base_dataset: MRIDiffusionHDF5Dataset,
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
        
        # Create new sample with only tensors
        return {
            'image': sample['image'],
            'mask': sample['mask'],
            'input_ids': text_inputs.input_ids.squeeze(0),
            'attention_mask': text_inputs.attention_mask.squeeze(0)
        }

def get_dataloaders(
    hdf5_file: str,
    tokenizer: CLIPTokenizer,
    batch_size: int = 1,
    num_workers: int = 4,
    transform: Optional[callable] = None,
    size: int = 512,
    center_crop: bool = False,
    verbose: bool = False
) -> Tuple[DataLoader, DataLoader]:
    # Create base datasets
    train_dataset = MRIDiffusionHDF5Dataset(
        hdf5_file=hdf5_file,
        split='train',
        transform=transform,
        size=size,
        center_crop=center_crop,
        verbose=verbose
    )
    
    test_dataset = MRIDiffusionHDF5Dataset(
        hdf5_file=hdf5_file,
        split='test',
        transform=transform,
        size=size,
        center_crop=center_crop,
        verbose=verbose
    )
    
    # Wrap with tokenized datasets
    train_dataset = TokenizedMRIDataset(train_dataset, tokenizer)
    test_dataset = TokenizedMRIDataset(test_dataset, tokenizer)
    
    print(f"Found {len(train_dataset)} training samples")
    print(f"Found {len(test_dataset)} test samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Initialize CLIP tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    print("\nLoading dataset...")
    # Create base datasets first to access metadata
    base_train_dataset = MRIDiffusionHDF5Dataset(
        hdf5_file="data/mama_mia_processed.h5",
        split='train',
        size=512,
        center_crop=False,
        verbose=True
    )
    
    # Print metadata from first sample
    first_sample = base_train_dataset[0]
    print("\nMetadata from first sample:")
    print(f"Patient ID: {first_sample['patient_id']}")
    print(f"Slice Index: {first_sample['slice_idx']}")
    print(f"Caption: {first_sample['caption']}")
    print("\nClinical Data:")
    for key, value in first_sample['clinical_data'].items():
        print(f"  {key}: {value}")
    print("\nImaging Data:")
    for key, value in first_sample['imaging_data'].items():
        print(f"  {key}: {value}")
    print("\nPrimary Lesion Data:")
    for key, value in first_sample['primary_lesion'].items():
        print(f"  {key}: {value}")
    
    # Now create the tokenized datasets and dataloaders
    train_loader, test_loader = get_dataloaders(
        hdf5_file="data/mama_mia_processed.h5",
        tokenizer=tokenizer,
        batch_size=2,  # Increased batch size for testing
        num_workers=4,
        size=512,
        center_crop=False,
        verbose=True
    )
    
    print("\nTesting multiple batches...")
    num_batches_to_test = 5
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches_to_test:
            break
            
        print(f"\nBatch {batch_idx + 1}:")
        print(f"Image batch shape: {batch['image'].shape}")
        print(f"Mask batch shape: {batch['mask'].shape}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Image batch min: {batch['image'].min().item():.3f}, max: {batch['image'].max().item():.3f}")
        print(f"Mask batch min: {batch['mask'].min().item():.3f}, max: {batch['mask'].max().item():.3f}")
        print(f"Input IDs min: {batch['input_ids'].min().item()}, max: {batch['input_ids'].max().item()}")
        print(f"Attention mask min: {batch['attention_mask'].min().item()}, max: {batch['attention_mask'].max().item()}") 