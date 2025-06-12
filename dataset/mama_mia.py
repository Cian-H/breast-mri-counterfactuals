import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPTokenizer
import SimpleITK as sitk
from tqdm import tqdm
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

def preprocess_dce_mri(
    nifti_files: List[str],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),  # Isotropic spacing
    target_size: int = 512,
    verbose: bool = False
) -> np.ndarray:
    """
    Preprocess DCE-MRI sequence using SimpleITK.
    Following MAMA-MIA approach with z-score normalization and isotropic resampling.
    Uses only the first phase of the DCE-MRI sequence.
    Returns a single 3D volume (height, width, depth).
    """
    # Load first phase only
    file_path = nifti_files[0]  # Use first phase
    sitk_img = sitk.ReadImage(file_path)
    volume = sitk.GetArrayFromImage(sitk_img)
    original_spacing = sitk_img.GetSpacing()
    
    if verbose:
        print(f"\nSimpleITK Image Info for {file_path}:")
        print(f"Size: {sitk_img.GetSize()}")
        print(f"Spacing: {sitk_img.GetSpacing()}")
        print(f"Origin: {sitk_img.GetOrigin()}")
        print(f"Direction: {sitk_img.GetDirection()}")
        print(f"Pixel Type: {sitk_img.GetPixelIDTypeAsString()}")
        print(f"Data range before normalization: [{volume.min()}, {volume.max()}]")
    
    # Calculate z-score normalization parameters
    mean = np.mean(volume)
    std = np.std(volume)
    
    if verbose:
        print(f"Original spacing: {original_spacing}")
        print(f"Mean: {mean:.2f}, Std: {std:.2f}")
    
    # Z-score normalization
    normalized_volume = (volume - mean) / std
    
    # Convert to SimpleITK image
    sitk_volume = sitk.GetImageFromArray(normalized_volume)
    sitk_volume.SetSpacing(original_spacing)
    
    # Create resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(sitk_volume.GetSize())
    resample.SetInterpolator(sitk.sitkLinear)
    
    # Resample
    resampled = resample.Execute(sitk_volume)
    resampled_volume = sitk.GetArrayFromImage(resampled)
    
    if verbose:
        print(f"Shape after resampling: {resampled_volume.shape}")
        print(f"Data range after normalization: [{resampled_volume.min():.2f}, {resampled_volume.max():.2f}]")
    
    return resampled_volume

class MRIDiffusionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        segmentation_dir: str,
        split_file: str,
        patient_info_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        segmentation_type: str = 'expert',
        size: int = 512,
        center_crop: bool = False,
        verbose: bool = False,
        num_workers: int = 10
    ):
        """
        Dataset for MRI diffusion data with segmentation masks and patient information.
        
        Args:
            data_dir (str): Path to the data directory containing images
            segmentation_dir (str): Path to the segmentation directory
            split_file (str): Path to the CSV file containing train/test splits
            patient_info_dir (str): Path to the directory containing patient JSON files
            split (str): Either 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on the images
            segmentation_type (str): Either 'expert' or 'automatic' segmentations
            size (int): Size to resize images to
            center_crop (bool): Whether to center crop images
            verbose (bool): Whether to print debug information
            num_workers (int): Number of workers for parallel processing
        """
        self.data_dir = Path(data_dir)
        self.segmentation_dir = Path(segmentation_dir) / segmentation_type
        self.patient_info_dir = Path(patient_info_dir)
        self.transform = transform
        self.size = size
        self.center_crop = center_crop
        self.verbose = verbose
        
        # Load train-test split
        splits_df = pd.read_csv(split_file, dtype=str)
        splits_df = splits_df.fillna('')
        
        if split == 'train':
            self.patient_ids = [id for id in splits_df['train_split'].tolist() if id != '']
        else:
            self.patient_ids = [id for id in splits_df['test_split'].tolist() if id != '']
        
        # Process patients and create index mapping
        self.processed_data = {}
        self.index = []
        
        print(f"Processing {len(self.patient_ids)} patients...")
        for patient_id in tqdm(self.patient_ids, desc="Processing patients"):
            patient_img_dir = self.data_dir / patient_id
            seg_file = self.segmentation_dir / f"{patient_id}.nii.gz"
            
            if not (patient_img_dir.exists() and seg_file.exists()):
                continue
                
            # Process first phase
            first_phase = next(patient_img_dir.glob('*.nii.gz'))
            processed_volume = preprocess_dce_mri(
                [str(first_phase)],
                target_size=self.size,
                verbose=self.verbose
            )
            
            # Process segmentation
            nifti_seg = sitk.ReadImage(str(seg_file))
            seg_spacing = nifti_seg.GetSpacing()
            seg_size = nifti_seg.GetSize()
            new_seg_size = [int(round(sz * (spc / tgt))) for sz, spc, tgt in zip(seg_size, seg_spacing, (1.0, 1.0, 1.0))]
            resample = sitk.ResampleImageFilter()
            resample.SetOutputSpacing((1.0, 1.0, 1.0))
            resample.SetSize(new_seg_size)
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
            resample.SetOutputOrigin(nifti_seg.GetOrigin())
            resample.SetOutputDirection(nifti_seg.GetDirection())
            resample.SetDefaultPixelValue(0)
            seg_resampled = sitk.GetArrayFromImage(resample.Execute(nifti_seg))
            
            # Store processed data
            self.processed_data[patient_id] = {
                'volume': processed_volume,
                'segmentation': seg_resampled
            }
            
            # Add to index
            num_slices = processed_volume.shape[-1]
            for slice_idx in range(num_slices):
                self.index.append((patient_id, slice_idx))
        
        if len(self.index) == 0:
            raise ValueError(f"No valid image-segmentation pairs found for split '{split}'")
        
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
        
        # Get preprocessed data
        patient_data = self.processed_data[patient_id]
        img_data = patient_data['volume'][:, :, slice_idx]
        seg_slice = patient_data['segmentation'][:, :, slice_idx]
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_data).float()
        img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)  # Add channel dimension and repeat for RGB
        
        # Resize if needed
        if self.center_crop:
            img_tensor = transforms.Resize(self.size)(img_tensor)
            img_tensor = transforms.CenterCrop(self.size)(img_tensor)
        else:
            img_tensor = transforms.Resize((self.size, self.size))(img_tensor)
        
        # Convert segmentation to tensor
        seg_tensor = torch.from_numpy(seg_slice).float()
        seg_tensor = seg_tensor.unsqueeze(0)
        if self.center_crop:
            seg_tensor = transforms.Resize(self.size)(seg_tensor)
            seg_tensor = transforms.CenterCrop(self.size)(seg_tensor)
        else:
            seg_tensor = transforms.Resize((self.size, self.size))(seg_tensor)
        
        # Get patient info
        info_file = self.patient_info_dir / f"{patient_id}.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                patient_data = json.load(f)
        else:
            patient_data = {}
        
        # Generate caption
        caption = "MRI showing normal tissue"
        if 'primary_lesion' in patient_data:
            lesion = patient_data['primary_lesion']
            if 'breast_coordinates' in lesion and 'tumor_subtype' in lesion:
                coords = lesion['breast_coordinates']
                if coords['z_min'] <= slice_idx <= coords['z_max'] and torch.any(seg_tensor > 0):
                    tumor_type = lesion['tumor_subtype']
                    caption = f"MRI showing {tumor_type} tumor"
        
        return {
            'image': img_tensor,
            'mask': seg_tensor,
            'patient_id': patient_id,
            'clinical_data': patient_data.get('clinical_data', {}),
            'imaging_data': patient_data.get('imaging_data', {}),
            'primary_lesion': patient_data.get('primary_lesion', {}),
            'caption': caption,
            'slice_idx': slice_idx
        }

class TokenizedMRIDataset(Dataset):
    """Wrapper dataset that handles tokenization and ensures tensor outputs."""
    def __init__(
        self,
        base_dataset: MRIDiffusionDataset,
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
    data_dir: str,
    segmentation_dir: str,
    split_file: str,
    patient_info_dir: str,
    tokenizer: CLIPTokenizer,
    batch_size: int = 1,
    num_workers: int = 4,
    transform: Optional[callable] = None,
    segmentation_type: str = 'expert',
    size: int = 512,
    center_crop: bool = False,
    verbose: bool = False
) -> Tuple[DataLoader, DataLoader]:
    # Create base datasets
    train_dataset = MRIDiffusionDataset(
        data_dir=data_dir,
        segmentation_dir=segmentation_dir,
        split_file=split_file,
        patient_info_dir=patient_info_dir,
        split='train',
        transform=transform,
        segmentation_type=segmentation_type,
        size=size,
        center_crop=center_crop,
        verbose=verbose
    )
    
    test_dataset = MRIDiffusionDataset(
        data_dir=data_dir,
        segmentation_dir=segmentation_dir,
        split_file=split_file,
        patient_info_dir=patient_info_dir,
        split='test',
        transform=transform,
        segmentation_type=segmentation_type,
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

def inspect_nifti_file(file_path: str) -> dict:
    """
    Inspect a NIfTI file and return relevant information about its structure.
    
    Args:
        file_path (str): Path to the NIfTI file
        
    Returns:
        dict: Dictionary containing file information
    """
    nifti_img = sitk.ReadImage(file_path)
    header = nifti_img.GetMetaData()
    
    # Get pixel spacing from header
    pixel_spacing = nifti_img.GetSpacing()
    
    # Get shape and number of phases
    shape = nifti_img.GetSize()
    
    return {
        'shape': shape,
        'pixel_spacing': pixel_spacing,
        'affine': nifti_img.GetDirection(),
        'data_type': nifti_img.GetPixelIDValueAsString(nifti_img.GetPixelID()),
        'data_range': (nifti_img.GetPixelIDValueAsDouble(nifti_img.GetPixelID()), nifti_img.GetPixelIDValueAsDouble(nifti_img.GetPixelID()))
    }

# Example usage:
if __name__ == "__main__":
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # in MB
    
    print("Initial memory usage:", get_memory_usage(), "MB")
    
    # Initialize CLIP tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    print("\nLoading dataset...")
    train_loader, test_loader = get_dataloaders(
        data_dir="data/images",
        segmentation_dir="data/segmentations",
        split_file="data/train_test_splits.csv",
        patient_info_dir="data/patient_info_files",
        tokenizer=tokenizer,
        batch_size=1,
        num_workers=4,
        segmentation_type='expert',
        size=512,
        center_crop=False,
        verbose=True
    )
    
    print("\nAfter dataset loading memory usage:", get_memory_usage(), "MB")
    
    print("\nTesting first batch...")
    for batch in train_loader:
        print(f"Image batch shape: {batch['image'].shape}")
        print(f"Mask batch shape: {batch['mask'].shape}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Slice index: {batch.get('slice_idx', 'N/A')}")
        print(f"Image batch min: {batch['image'].min().item()}, max: {batch['image'].max().item()}")
        break  # Just print the first batch
    
    print("\nFinal memory usage:", get_memory_usage(), "MB")
