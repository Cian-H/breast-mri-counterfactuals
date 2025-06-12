import os
import numpy as np
import h5py
import SimpleITK as sitk
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# This version uses tanh normalization for better numerical stability
# in two-stage normalization scenarios (volume-level + per-slice normalization)
# Output range: (-1, 1) with smooth handling of outliers

def preprocess_dce_mri(
    nifti_files: List[str],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    target_size: int = 512,
    verbose: bool = False
) -> np.ndarray:
    """Preprocess DCE-MRI sequence using SimpleITK with tanh normalization."""
    file_path = nifti_files[0]  # Use first phase
    sitk_img = sitk.ReadImage(file_path)
    volume = sitk.GetArrayFromImage(sitk_img)
    original_spacing = sitk_img.GetSpacing()
    original_size = sitk_img.GetSize()
    
    if verbose:
        print(f"\nSimpleITK Image Info for {file_path}:")
        print(f"Size: {original_size}")
        print(f"Spacing: {original_spacing}")
        print(f"Data range before normalization: [{volume.min()}, {volume.max()}]")
    
    # Tanh normalization for stable two-stage processing
    mean = np.mean(volume)
    std = np.std(volume)
    
    # Z-score first
    zscore_volume = (volume - mean) / std
    
    # Apply tanh to map to (-1, 1) range
    normalized_volume = np.tanh(zscore_volume)
    
    if verbose:
        print(f"\nTanh normalization stats:")
        print(f"Original mean: {mean:.6f}, std: {std:.6f}")
        print(f"Z-score range: [{zscore_volume.min():.6f}, {zscore_volume.max():.6f}]")
        print(f"Tanh normalized min: {normalized_volume.min():.6f}, max: {normalized_volume.max():.6f}")
        print(f"Tanh normalized mean: {normalized_volume.mean():.6f}, std: {normalized_volume.std():.6f}")
        
        # Check slice-level statistics for numerical stability
        slice_stds = [np.std(normalized_volume[i]) for i in range(min(5, normalized_volume.shape[0]))]
        print(f"Sample slice standard deviations: {slice_stds}")
    
    # Convert to SimpleITK image
    sitk_volume = sitk.GetImageFromArray(normalized_volume)
    sitk_volume.SetSpacing(original_spacing)
    
    # Calculate new size based on spacing ratio (MAMA-MIA approach)
    tol = 0.00001
    new_size = [
        round(original_size[0] * (original_spacing[0] + tol) / target_spacing[0]),
        round(original_size[1] * (original_spacing[1] + tol) / target_spacing[1]),
        round(original_size[2] * (original_spacing[2] + tol) / target_spacing[2])
    ]
    
    # Create resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(sitk.sitkBSpline)  # Use BSpline as in MAMA-MIA
    resample.SetOutputDirection(sitk_volume.GetDirection())
    resample.SetOutputOrigin(sitk_volume.GetOrigin())
    resample.SetOutputPixelType(sitk_volume.GetPixelID())
    resample.SetTransform(sitk.Transform())
    
    # Resample
    try:
        resampled = resample.Execute(sitk_volume)
    except RuntimeError:
        # Handle direction determinant error as in MAMA-MIA
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        resample.SetOutputDirection(direction)
        sitk_volume.SetDirection(direction)
        resampled = resample.Execute(sitk_volume)
    
    resampled_volume = sitk.GetArrayFromImage(resampled)
    
    if verbose:
        print(f"Shape after resampling: {resampled_volume.shape}")
        print(f"Data range after normalization: [{resampled_volume.min():.2f}, {resampled_volume.max():.2f}]")
    
    return resampled_volume

def process_segmentation(
    seg_file: str,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """Process segmentation mask."""
    nifti_seg = sitk.ReadImage(seg_file)
    original_spacing = nifti_seg.GetSpacing()
    original_size = nifti_seg.GetSize()
    
    # Calculate new size based on spacing ratio (MAMA-MIA approach)
    tol = 0.00001
    new_size = [
        round(original_size[0] * (original_spacing[0] + tol) / target_spacing[0]),
        round(original_size[1] * (original_spacing[1] + tol) / target_spacing[1]),
        round(original_size[2] * (original_spacing[2] + tol) / target_spacing[2])
    ]
    
    # Create resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)  # Keep nearest neighbor for segmentation
    resample.SetOutputDirection(nifti_seg.GetDirection())
    resample.SetOutputOrigin(nifti_seg.GetOrigin())
    resample.SetOutputPixelType(nifti_seg.GetPixelID())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    
    # Resample
    try:
        resampled = resample.Execute(nifti_seg)
    except RuntimeError:
        # Handle direction determinant error as in MAMA-MIA
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        resample.SetOutputDirection(direction)
        nifti_seg.SetDirection(direction)
        resampled = resample.Execute(nifti_seg)
    
    seg_resampled = sitk.GetArrayFromImage(resampled)
    return seg_resampled

def process_patient(
    patient_id: str,
    data_dir: Path,
    segmentation_dir: Path,
    patient_info_dir: Path,
    target_size: int = 512,
    verbose: bool = False
) -> Dict[str, Any]:
    """Process a single patient's data."""
    patient_img_dir = data_dir / patient_id
    seg_file = segmentation_dir / f"{patient_id}.nii.gz"
    info_file = patient_info_dir / f"{patient_id}.json"
    
    print(f"\nProcessing patient {patient_id}:")
    print(f"Image dir exists: {patient_img_dir.exists()}")
    print(f"Seg file exists: {seg_file.exists()}")
    print(f"Info file exists: {info_file.exists()}")
    
    if not (patient_img_dir.exists() and seg_file.exists()):
        print(f"Skipping patient {patient_id} - missing files")
        return None
    
    # Find all phase files and sort them
    phase_files = sorted(list(patient_img_dir.glob('*.nii.gz')))
    if not phase_files:
        print(f"No phase files found for patient {patient_id}")
        return None
    
    # Sort files by phase number in filename (e.g., DUKE_010_0000.nii.gz -> 0)
    def get_phase_number(file_path):
        # Get filename without any extension
        filename = file_path.name.replace('.nii.gz', '')
        # Split by underscore and get the last part
        parts = filename.split('_')
        # Get the last part and convert to int
        return int(parts[-1])
    
    phase_files.sort(key=get_phase_number)
    
    # Use the first phase (pre-contrast)
    first_phase = phase_files[0]
    print(f"Selected first phase: {first_phase}")
    
    processed_volume = preprocess_dce_mri(
        [str(first_phase)],
        target_size=target_size,
        verbose=verbose
    )
    
    # Process segmentation
    seg_resampled = process_segmentation(str(seg_file))
    
    # Load patient info
    if info_file.exists():
        with open(info_file, 'r') as f:
            patient_data = json.load(f)
        print(f"Loaded patient info: {list(patient_data.keys())}")
    else:
        patient_data = {}
        print("No patient info file found")
    
    return {
        'patient_id': patient_id,
        'volume': processed_volume,
        'segmentation': seg_resampled,
        'info': patient_data
    }

def create_hdf5_dataset(
    data_dir: str,
    segmentation_dir: str,
    split_file: str,
    patient_info_dir: str,
    output_file: str,
    target_size: int = 512,
    num_workers: int = 10,
    batch_size: int = 10,  # Process 5 patients at a time
    verbose: bool = False
):
    """Create HDF5 dataset from raw data."""
    data_dir = Path(data_dir)
    segmentation_dir = Path(segmentation_dir) / "expert"  # Use expert segmentations
    patient_info_dir = Path(patient_info_dir)
    
    print(f"Using segmentation directory: {segmentation_dir}")
    
    # Load train-test split
    splits_df = pd.read_csv(split_file, dtype=str)
    splits_df = splits_df.fillna('')
    
    # Process all patients
    all_patients = []
    for split in ['train_split', 'test_split']:
        patients = [id for id in splits_df[split].tolist() if id != '']
        all_patients.extend([(patient_id, split) for patient_id in patients])
    
    print(f"Processing {len(all_patients)} patients using {num_workers} workers in batches of {batch_size}...")
    
    # Create HDF5 file
    with h5py.File(output_file, 'w') as f:
        # Create groups for train and test
        train_group = f.create_group('train')
        test_group = f.create_group('test')
        
        # Process patients in batches
        for i in range(0, len(all_patients), batch_size):
            batch = all_patients[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(all_patients) + batch_size - 1)//batch_size}")
            print(f"Current memory usage: {get_memory_usage():.1f} MB")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit batch tasks
                future_to_patient = {
                    executor.submit(
                        process_patient,
                        patient_id,
                        data_dir,
                        segmentation_dir,
                        patient_info_dir,
                        target_size,
                        verbose
                    ): (patient_id, split)
                    for patient_id, split in batch
                }
                
                # Process results as they complete
                for future in tqdm(as_completed(future_to_patient), total=len(batch), desc=f"Batch {i//batch_size + 1}"):
                    patient_id, split = future_to_patient[future]
                    try:
                        result = future.result()
                        if result is None:
                            print(f"Skipping patient {patient_id} - processing failed")
                            continue
                        
                        # Create patient group
                        group = train_group if split == 'train_split' else test_group
                        patient_group = group.create_group(patient_id)
                        
                        # Save volume and segmentation
                        patient_group.create_dataset('volume', data=result['volume'], compression='gzip')
                        patient_group.create_dataset('segmentation', data=result['segmentation'], compression='gzip')
                        
                        # Save metadata
                        for key, value in result['info'].items():
                            if isinstance(value, (str, int, float, bool)):
                                patient_group.attrs[key] = value
                            else:
                                patient_group.create_dataset(f'metadata_{key}', data=json.dumps(value))
                                
                    except Exception as e:
                        print(f"Error processing patient {patient_id}: {str(e)}")
                        continue
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
            
            print(f"Batch completed. Memory usage: {get_memory_usage():.1f} MB")

if __name__ == "__main__":
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # in MB
    
    print("Initial memory usage:", get_memory_usage(), "MB")
    
    create_hdf5_dataset(
        data_dir="data/images",
        segmentation_dir="data/segmentations",
        split_file="data/train_test_splits.csv",
        patient_info_dir="data/patient_info_files",
        output_file="data/mama_mia_processed_tanh.h5",
        target_size=512,
        num_workers=20,
        batch_size=20,  # Process 5 patients at a time
        verbose=True
    )
    
    print("\nFinal memory usage:", get_memory_usage(), "MB") 