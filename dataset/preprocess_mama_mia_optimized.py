import os
import numpy as np
import h5py
import SimpleITK as sitk
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import time

# Optimized version with significant performance improvements:
# 1. ProcessPoolExecutor instead of ThreadPoolExecutor (bypasses GIL)
# 2. Pre-validation and caching
# 3. Optimized I/O operations
# 4. Better memory management
# 5. Reduced redundant operations
# 6. Slice validation during preprocessing

# Global resampler setup to avoid recreation
RESAMPLER_CONFIG = {
    'target_spacing': (1.0, 1.0, 1.0),
    'interpolator_volume': sitk.sitkBSpline,
    'interpolator_seg': sitk.sitkNearestNeighbor,
    'default_direction': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
}

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def is_valid_slice(slice_data: np.ndarray, seg_slice: np.ndarray) -> bool:
    """
    Validate slice quality, filtering out problematic slices but keeping 
    valid slices regardless of segmentation content.
    """
    # 1. Check for completely empty/zero slices
    if np.all(slice_data == 0) or np.std(slice_data) < 1e-6:
        return False
        
    # 2. Check for acquisition artifacts (extreme values)
    if np.any(np.isnan(slice_data)) or np.any(np.isinf(slice_data)):
        return False
        
    # 3. Check for minimal anatomical content (very low contrast)
    p99, p1 = np.percentile(slice_data, [99, 1])
    if (p99 - p1) < 0.05:  # Minimal dynamic range after tanh normalization
        return False
        
    # 4. Check for mostly background (outside body boundaries)
    # After tanh normalization, meaningful tissue should have some variation
    non_background_pixels = np.sum(np.abs(slice_data) > 0.1)  # Threshold for tanh normalized data
    total_pixels = slice_data.size
    if non_background_pixels / total_pixels < 0.05:  # >95% background
        return False
    
    # 5. Segmentation presence is NOT a criterion - we want slices without segmentation too!
    
    return True

def validate_slices(volume: np.ndarray, segmentation: np.ndarray, verbose: bool = False) -> Tuple[List[int], List[Dict]]:
    """
    Validate all slices in a volume and return valid slice indices and statistics.
    """
    valid_slices = []
    slice_stats = []
    
    num_slices = volume.shape[-1] if len(volume.shape) == 3 else volume.shape[0]
    
    for slice_idx in range(num_slices):
        # Handle different volume orientations
        if len(volume.shape) == 3:
            slice_data = volume[:, :, slice_idx]
            seg_slice = segmentation[:, :, slice_idx]
        else:
            slice_data = volume[slice_idx, :, :]
            seg_slice = segmentation[slice_idx, :, :]
        
        if is_valid_slice(slice_data, seg_slice):
            valid_slices.append(slice_idx)
            
            # Store slice characteristics for analysis
            slice_stats.append({
                'slice_idx': slice_idx,
                'has_segmentation': bool(np.any(seg_slice > 0)),
                'std': float(np.std(slice_data)),
                'dynamic_range': float(np.percentile(slice_data, 99) - np.percentile(slice_data, 1)),
                'non_background_ratio': float(np.sum(np.abs(slice_data) > 0.1) / slice_data.size)
            })
    
    if verbose:
        total_slices = num_slices
        valid_count = len(valid_slices)
        with_seg = sum(1 for stats in slice_stats if stats['has_segmentation'])
        print(f"  Slice validation: {valid_count}/{total_slices} valid ({with_seg} with segmentation)")
    
    return valid_slices, slice_stats

def create_resampler(original_spacing, original_size, interpolator, default_pixel_value=0):
    """Create optimized resampler with cached configuration."""
    target_spacing = RESAMPLER_CONFIG['target_spacing']
    
    # Calculate new size (MAMA-MIA approach)
    tol = 0.00001
    new_size = [
        round(original_size[i] * (original_spacing[i] + tol) / target_spacing[i])
        for i in range(3)
    ]
    
    # Create resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(interpolator)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(default_pixel_value)
    
    return resample

def fast_preprocess_dce_mri(file_path: str, verbose: bool = False) -> np.ndarray:
    """Optimized DCE-MRI preprocessing with minimal operations."""
    # Read image once
    sitk_img = sitk.ReadImage(file_path)
    volume = sitk.GetArrayFromImage(sitk_img)
    original_spacing = sitk_img.GetSpacing()
    original_size = sitk_img.GetSize()
    
    if verbose:
        print(f"Processing {Path(file_path).name}: {original_size}, range: [{volume.min():.1f}, {volume.max():.1f}]")
    
    # First normalize with Z-score for stable resampling
    volume_flat = volume.ravel()
    mean = np.mean(volume_flat)
    std = np.std(volume_flat)
    zscore_volume = (volume - mean) / std
    
    # Create resampler
    resampler = create_resampler(
        original_spacing, 
        original_size, 
        RESAMPLER_CONFIG['interpolator_volume']
    )
    
    # Setup resampler with original image properties
    sitk_volume = sitk.GetImageFromArray(zscore_volume)
    sitk_volume.SetSpacing(original_spacing)
    sitk_volume.SetOrigin(sitk_img.GetOrigin())
    sitk_volume.SetDirection(sitk_img.GetDirection())
    
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetOutputPixelType(sitk_volume.GetPixelID())
    
    # Resample with error handling
    try:
        resampled = resampler.Execute(sitk_volume)
    except RuntimeError:
        # Handle direction determinant error
        resampler.SetOutputDirection(RESAMPLER_CONFIG['default_direction'])
        sitk_volume.SetDirection(RESAMPLER_CONFIG['default_direction'])
        resampled = resampler.Execute(sitk_volume)
    
    resampled_volume = sitk.GetArrayFromImage(resampled)
    
    # Apply tanh normalization AFTER resampling to ensure (-1, 1) range
    normalized_volume = np.tanh(resampled_volume)
    
    if verbose:
        print(f"  Final range after tanh: [{normalized_volume.min():.6f}, {normalized_volume.max():.6f}]")
    
    return normalized_volume

def fast_process_segmentation(seg_file: str) -> np.ndarray:
    """Optimized segmentation processing."""
    nifti_seg = sitk.ReadImage(seg_file)
    original_spacing = nifti_seg.GetSpacing()
    original_size = nifti_seg.GetSize()
    
    # Create resampler
    resampler = create_resampler(
        original_spacing, 
        original_size, 
        RESAMPLER_CONFIG['interpolator_seg']
    )
    
    # Setup resampler
    resampler.SetOutputDirection(nifti_seg.GetDirection())
    resampler.SetOutputOrigin(nifti_seg.GetOrigin())
    resampler.SetOutputPixelType(nifti_seg.GetPixelID())
    
    # Resample with error handling
    try:
        resampled = resampler.Execute(nifti_seg)
    except RuntimeError:
        resampler.SetOutputDirection(RESAMPLER_CONFIG['default_direction'])
        nifti_seg.SetDirection(RESAMPLER_CONFIG['default_direction'])
        resampled = resampler.Execute(nifti_seg)
    
    return sitk.GetArrayFromImage(resampled)

def validate_and_prepare_patient_files(patient_id: str, base_paths: Dict[str, Path]) -> Optional[Dict[str, Path]]:
    """Pre-validate all patient files to avoid runtime errors."""
    patient_img_dir = base_paths['data_dir'] / patient_id
    seg_file = base_paths['segmentation_dir'] / f"{patient_id}.nii.gz"
    info_file = base_paths['patient_info_dir'] / f"{patient_id}.json"
    
    # Quick existence checks
    if not patient_img_dir.exists() or not seg_file.exists():
        return None
    
    # Find and validate phase files
    phase_files = list(patient_img_dir.glob('*.nii.gz'))
    if not phase_files:
        return None
    
    # Sort by phase number
    def get_phase_number(file_path):
        filename = file_path.name.replace('.nii.gz', '')
        parts = filename.split('_')
        return int(parts[-1])
    
    phase_files.sort(key=get_phase_number)
    
    return {
        'first_phase': phase_files[0],
        'seg_file': seg_file,
        'info_file': info_file if info_file.exists() else None
    }

def process_single_patient(args: Tuple[str, Dict[str, Path], bool]) -> Optional[Dict[str, Any]]:
    """Optimized single patient processing function for multiprocessing with slice validation."""
    patient_id, file_paths, verbose = args
    
    try:
        # Process volume
        processed_volume = fast_preprocess_dce_mri(str(file_paths['first_phase']), verbose)
        
        # Process segmentation
        seg_resampled = fast_process_segmentation(str(file_paths['seg_file']))
        
        # Validate slices
        valid_slices, slice_stats = validate_slices(processed_volume, seg_resampled, verbose)
        
        if len(valid_slices) == 0:
            print(f"No valid slices found for patient {patient_id}")
            return None
        
        # Load patient info efficiently
        patient_data = {}
        if file_paths['info_file']:
            with open(file_paths['info_file'], 'r') as f:
                patient_data = json.load(f)
        
        return {
            'patient_id': patient_id,
            'volume': processed_volume,
            'segmentation': seg_resampled,
            'valid_slices': valid_slices,
            'slice_stats': slice_stats,
            'info': patient_data
        }
        
    except Exception as e:
        print(f"Error processing patient {patient_id}: {str(e)}")
        return None

def create_optimized_hdf5_dataset(
    data_dir: str,
    segmentation_dir: str,
    split_file: str,
    patient_info_dir: str,
    output_file: str,
    target_size: int = 512,
    num_workers: Optional[int] = None,
    batch_size: int = 50,  # Larger batches for better efficiency
    verbose: bool = False
):
    """Optimized HDF5 dataset creation with significant performance improvements."""
    
    # Auto-detect optimal worker count
    if num_workers is None:
        num_workers = min(mp.cpu_count() - 1, 20)  # Leave one core free
    
    print(f"Using {num_workers} workers with batch size {batch_size}")
    
    # Setup paths
    base_paths = {
        'data_dir': Path(data_dir),
        'segmentation_dir': Path(segmentation_dir) / "expert",
        'patient_info_dir': Path(patient_info_dir)
    }
    
    # Load and process splits
    splits_df = pd.read_csv(split_file, dtype=str).fillna('')
    
    all_patients = []
    for split in ['train_split', 'test_split']:
        patients = [id for id in splits_df[split].tolist() if id != '']
        all_patients.extend([(patient_id, split) for patient_id in patients])
    
    print(f"Pre-validating {len(all_patients)} patients...")
    
    # Pre-validate all patients (fast I/O check)
    start_time = time.time()
    valid_patients = []
    
    for patient_id, split in tqdm(all_patients, desc="Validating"):
        file_paths = validate_and_prepare_patient_files(patient_id, base_paths)
        if file_paths:
            valid_patients.append((patient_id, split, file_paths))
        else:
            print(f"Skipping {patient_id} - missing files")
    
    validation_time = time.time() - start_time
    print(f"Validation completed in {validation_time:.1f}s. {len(valid_patients)} valid patients.")
    
    # Create HDF5 file with optimized settings
    with h5py.File(output_file, 'w') as f:
        # Create groups
        train_group = f.create_group('train')
        test_group = f.create_group('test')
        
        # Process in batches with multiprocessing
        for i in range(0, len(valid_patients), batch_size):
            batch = valid_patients[i:i + batch_size]
            batch_start = time.time()
            
            print(f"\nBatch {i//batch_size + 1}/{(len(valid_patients) + batch_size - 1)//batch_size}")
            print(f"Memory usage: {get_memory_usage():.1f} MB")
            
            # Prepare arguments for multiprocessing
            process_args = [
                (patient_id, file_paths, verbose) 
                for patient_id, split, file_paths in batch
            ]
            
            # Process batch with multiprocessing
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(
                    executor.map(process_single_patient, process_args),
                    total=len(process_args),
                    desc=f"Processing batch {i//batch_size + 1}"
                ))
            
            # Write results to HDF5 efficiently
            write_start = time.time()
            for idx, result in enumerate(results):
                if result is None:
                    continue
                
                patient_id, split = batch[idx][0], batch[idx][1]
                group = train_group if split == 'train_split' else test_group
                patient_group = group.create_group(patient_id)
                
                # Optimized dataset creation with better compression
                patient_group.create_dataset(
                    'volume', 
                    data=result['volume'], 
                    compression='gzip',
                    compression_opts=6,  # Good balance of speed/compression
                    shuffle=True,
                    fletcher32=True
                )
                patient_group.create_dataset(
                    'segmentation', 
                    data=result['segmentation'], 
                    compression='gzip',
                    compression_opts=6,
                    shuffle=True
                )
                
                # Store slice validation results
                patient_group.create_dataset(
                    'valid_slices',
                    data=np.array(result['valid_slices'], dtype=np.int32),
                    compression='gzip'
                )
                
                # Store slice statistics as JSON string for flexibility
                slice_stats_json = json.dumps(result['slice_stats'])
                patient_group.create_dataset(
                    'slice_stats',
                    data=slice_stats_json,
                    dtype=h5py.string_dtype(encoding='utf-8')
                )
                
                # Store validation summary in attributes
                total_slices = result['volume'].shape[-1] if len(result['volume'].shape) == 3 else result['volume'].shape[0]
                valid_count = len(result['valid_slices'])
                with_seg_count = sum(1 for stats in result['slice_stats'] if stats['has_segmentation'])
                
                patient_group.attrs['total_slices'] = total_slices
                patient_group.attrs['valid_slices_count'] = valid_count
                patient_group.attrs['slices_with_segmentation'] = with_seg_count
                patient_group.attrs['slices_without_segmentation'] = valid_count - with_seg_count
                patient_group.attrs['validation_ratio'] = valid_count / total_slices
                
                # Save metadata
                for key, value in result['info'].items():
                    if isinstance(value, (str, int, float, bool)):
                        patient_group.attrs[key] = value
                    else:
                        patient_group.create_dataset(f'metadata_{key}', data=json.dumps(value))
            
            write_time = time.time() - write_start
            batch_time = time.time() - batch_start
            
            print(f"Batch {i//batch_size + 1} completed in {batch_time:.1f}s (write: {write_time:.1f}s)")
            print(f"Memory usage: {get_memory_usage():.1f} MB")

if __name__ == "__main__":
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    print(f"Available CPU cores: {mp.cpu_count()}")
    
    start_time = time.time()
    
    create_optimized_hdf5_dataset(
        data_dir="data/images",
        segmentation_dir="data/segmentations",
        split_file="data/train_test_splits.csv",
        patient_info_dir="data/patient_info_files",
        output_file="data/mama_mia_processed_tanh_corrected.h5",
        target_size=512,
        num_workers=None,  # Auto-detect
        batch_size=50,
        verbose=False  # Less verbose for speed
    )
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.1f}s")
    print(f"Final memory usage: {get_memory_usage():.1f} MB")
    
    # Print final summary
    print(f"\n{'='*50}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Output file: data/mama_mia_processed_tanh_corrected.h5")
    print(f"Processing time: {total_time:.1f} seconds")
    print(f"Ready for use with dataset/optimized_mri_dataset.py")
    print(f"{'='*50}") 