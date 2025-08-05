#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import gc
from third_parts import VideoReader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime


model_path = "ByteDance/Sa2VA-8B"

def load_model():
    """Load/reload model - use for aggressive memory reset"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Force FP16 to save memory
        device_map='cuda:0',
        trust_remote_code=True,
        low_cpu_mem_usage=True  # Reduce CPU-GPU transfer overhead
    ).eval()
    return model

def clear_model_memory(model):
    """Complete model memory reset"""
    del model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

def read_video(video_path, video_interval):
    vid_frames = VideoReader(video_path)[::video_interval]
    print(f"Number of frames: {len(vid_frames)}")
    for frame_idx in range(len(vid_frames)):
        frame_image = vid_frames[frame_idx]
        frame_image = frame_image[..., ::-1]  # BGR (opencv system) to RGB (numpy system)
        frame_image = Image.fromarray(frame_image)
        vid_frames[frame_idx] = frame_image
    return vid_frames
    
def _parse_labels(labels_string):
    """Parse pipe-separated labels into list of clean, unique labels"""
    if pd.isna(labels_string):
        return []
    
    labels = [label.strip() for label in str(labels_string).split('|')]
    # Remove empty strings and duplicates while preserving order
    seen = set()
    unique_labels = []
    for label in labels:
        if label and label not in seen:
            unique_labels.append(label)
            seen.add(label)
    
    return unique_labels

def process_single_row(row_index, mode, csv_type="long"):
    """Process a single row by index with specified segmentation mode
    
    Args:
        row_index: Index of the row to process
        mode: "default" or "csv" - which labels to use for segmentation
        csv_type: "long" or "short" - which CSV file to use
    """
    DEFAULT_LABELS = ["Road", "Sidewalk", "Building", "Vegetation", "Terrain", "Sky", "Person", "Road transport"]
    csv_file = f'./assets/labels_{csv_type}.csv'
    df = pd.read_csv(csv_file)
    
    # Initialize results tracking
    results = []
    row_start_time = datetime.now()
    
    if row_index >= len(df):
        error_msg = f"ERROR: Row index {row_index} out of range (max: {len(df)-1})"
        print(error_msg)
        # Log the error
        results.append({
            'row_index': row_index,
            'video_path': 'N/A',
            'mode': mode,
            'label': 'N/A',
            'status': 'ERROR',
            'message': error_msg,
            'output_path': 'N/A',
            'timestamp': row_start_time.isoformat(),
            'has_segmentation': False
        })
        _save_results_to_csv(results, mode)
        return
    
    row = df.iloc[row_index]
    print(f"Processing row {row_index + 1}/{len(df)}: {row['file_path']} (mode: {mode})")
    
    # Load model fresh for this row
    try:
        model = load_model()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        error_msg = f"ERROR: Failed to load model: {str(e)}"
        print(error_msg)
        results.append({
            'row_index': row_index,
            'video_path': row['file_path'],
            'mode': mode,
            'label': 'N/A',
            'status': 'ERROR',
            'message': error_msg,
            'output_path': 'N/A',
            'timestamp': datetime.now().isoformat(),
            'has_segmentation': False
        })
        _save_results_to_csv(results, mode)
        return
        
    vid_path = row['file_path']
    labels = row['final_labels']
    parsed_labels = _parse_labels(labels)
    
    # Select labels based on mode
    if mode == "default":
        LABELS = DEFAULT_LABELS
        print(f"Using DEFAULT labels only: {len(LABELS)} labels")
    elif mode == "csv":
        LABELS = parsed_labels
        print(f"Using CSV labels only: {len(LABELS)} labels")
    else:
        error_msg = f"Invalid mode '{mode}'. Must be 'default' or 'csv'"
        print(f"ERROR: {error_msg}")
        results.append({
            'row_index': row_index,
            'video_path': vid_path,
            'mode': mode,
            'label': 'N/A',
            'status': 'ERROR',
            'message': error_msg,
            'output_path': 'N/A',
            'timestamp': datetime.now().isoformat(),
            'has_segmentation': False
        })
        _save_results_to_csv(results, mode)
        return
    
    # Load video frames once per row with memory consideration
    try:
        vid_frames = read_video(vid_path, video_interval=120)
        
        # If video has more than 1 frame, use a higher interval to reduce memory usage
        if len(vid_frames) > 1:
            print(f"Video has {len(vid_frames)} frames, using higher interval to reduce memory usage")
            vid_frames = VideoReader(vid_path)[::360]  # Use every 360th frame instead of 120th
            print(f"Reduced to {len(vid_frames)} frames")
            for frame_idx in range(len(vid_frames)):
                frame_image = vid_frames[frame_idx]
                frame_image = frame_image[..., ::-1]  # BGR to RGB
                frame_image = Image.fromarray(frame_image)
                vid_frames[frame_idx] = frame_image
                
    except Exception as e:
        error_msg = f"ERROR: Failed to load video: {str(e)}"
        print(error_msg)
        results.append({
            'row_index': row_index,
            'video_path': vid_path,
            'mode': mode,
            'label': 'N/A',
            'status': 'ERROR',
            'message': error_msg,
            'output_path': 'N/A',
            'timestamp': datetime.now().isoformat(),
            'has_segmentation': False
        })
        _save_results_to_csv(results, mode)
        clear_model_memory(model)
        return
    
    for label in tqdm(LABELS, desc=f"Labels for row {row_index}"):
        label_start_time = datetime.now()
        
        try:
            # Clear GPU cache before each inference
            torch.cuda.empty_cache()
            gc.collect()
            
            # create a question (<image> is a placeholder for the video frames)
            question = f"<image>Please segment all the {label} in the scene." 
            return_dict = model.predict_forward(
                video=vid_frames,
                text=question,
                mask_prompts=None,
                tokenizer=tokenizer,
            )
            answer = return_dict["prediction"]
            
            # Handle segmentation if present
            if '[SEG]' in answer:
                pred_masks = return_dict["prediction_masks"][0]
                
                mask_confidence_score = 0.7
                # Ensure mask is in the correct format
                if isinstance(pred_masks, np.ndarray):
                    binary_mask = (pred_masks > mask_confidence_score).astype('uint8') * 255
                else:
                    binary_mask = (pred_masks.cpu().numpy() > mask_confidence_score).astype('uint8') * 255
                
                # Ensure mask has valid dimensions
                if binary_mask.ndim == 2:
                    height, width = binary_mask.shape
                elif binary_mask.ndim == 3:
                    # If we have a 3D array, take the first channel
                    binary_mask = binary_mask[0] if binary_mask.shape[0] == 1 else binary_mask[:, :, 0]
                    height, width = binary_mask.shape
                else:
                    msg = f"Invalid mask dimensions, skipping segmentation output"
                    print(msg)
                    print(f"Model response: {answer}")
                    results.append({
                        'row_index': row_index,
                        'video_path': vid_path,
                        'mode': mode,
                        'label': label,
                        'status': 'WARNING',
                        'message': msg,
                        'output_path': 'N/A',
                        'timestamp': label_start_time.isoformat(),
                        'has_segmentation': True
                    })
                    continue
                
                # Check if dimensions are valid and mask is not empty
                if binary_mask.ndim == 2:  # Only proceed if we have valid 2D mask
                    height, width = binary_mask.shape
                    if width > 0 and height > 0 and np.any(binary_mask):
                        # Create output directory structure
                        output_dir = f"./results/{mode}/{csv_type}"
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Save the binary mask with new directory structure
                        filename = vid_path.split('/')[-1].split('.')[0]
                        output_path = f"{output_dir}/{filename}_{label}.png"
                        if cv2.imwrite(output_path, binary_mask):
                            msg = f"SUCCESS: Segmentation mask saved to: {output_path}"
                            print(msg)
                            print(f"Model response: {answer}")
                            results.append({
                                'row_index': row_index,
                                'video_path': vid_path,
                                'mode': mode,
                                'label': label,
                                'status': 'SUCCESS',
                                'message': msg,
                                'output_path': output_path,
                                'timestamp': label_start_time.isoformat(),
                                'has_segmentation': True
                            })
                        else:
                            msg = f"ERROR: Failed to save mask to {output_path}"
                            print(msg)
                            print(f"Model response: {answer}")
                            results.append({
                                'row_index': row_index,
                                'video_path': vid_path,
                                'mode': mode,
                                'label': label,
                                'status': 'ERROR',
                                'message': msg,
                                'output_path': 'N/A',
                                'timestamp': label_start_time.isoformat(),
                                'has_segmentation': True
                            })
                    else:
                        msg = f"WARNING: Invalid or empty mask, skipping segmentation output"
                        print(msg)
                        print(f"Model response: {answer}")
                        results.append({
                            'row_index': row_index,
                            'video_path': vid_path,
                            'mode': mode,
                            'label': label,
                            'status': 'WARNING',
                            'message': msg,
                            'output_path': 'N/A',
                            'timestamp': label_start_time.isoformat(),
                            'has_segmentation': True
                        })
            else:
                print(f"Model response: {answer}")
                results.append({
                    'row_index': row_index,
                    'video_path': vid_path,
                    'mode': mode,
                    'label': label,
                    'status': 'SUCCESS',
                    'message': f"No segmentation in response: {answer}",
                    'output_path': 'N/A',
                    'timestamp': label_start_time.isoformat(),
                    'has_segmentation': False
                })
            
            # Clear prediction data from GPU after processing
            if "prediction_masks" in return_dict:
                del return_dict["prediction_masks"]
            del return_dict
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            error_msg = f"ERROR: Exception during segmentation for label '{label}': {str(e)}"
            print(error_msg)
            results.append({
                'row_index': row_index,
                'video_path': vid_path,
                'mode': mode,
                'label': label,
                'status': 'ERROR',
                'message': error_msg,
                'output_path': 'N/A',
                'timestamp': label_start_time.isoformat(),
                'has_segmentation': False
            })
            # Continue with next label
    
    # Clear memory at the end of this row processing
    clear_model_memory(model)
    print(f"GPU memory cleared for row {row_index}")
    
    # Save results to CSV
    _save_results_to_csv(results, mode)

def _save_results_to_csv(results, mode):
    """Save processing results to CSV file"""
    if not results:
        return
        
    # Create logs directory if it doesn't exist
    os.makedirs("./logs", exist_ok=True)
    
    # Create filename with timestamp and mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"./logs/processing_results_{mode}_{timestamp}.csv"
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(csv_filename)
    
    # Append to existing file or create new one
    df.to_csv(csv_filename, mode='a', header=not file_exists, index=False)
    
    print(f"Results logged to: {csv_filename}")

if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Usage: python process_single_row.py <row_index> <mode> [csv_type]")
        print("  mode: 'default' or 'csv'")
        print("  csv_type: 'long' or 'short' (default: 'long')")
        sys.exit(1)
    
    try:
        row_index = int(sys.argv[1])
        mode = sys.argv[2]
        csv_type = sys.argv[3] if len(sys.argv) == 4 else "long"
        
        if mode not in ["default", "csv"]:
            print("ERROR: Mode must be 'default' or 'csv'")
            sys.exit(1)
            
        if csv_type not in ["long", "short"]:
            print("ERROR: CSV type must be 'long' or 'short'")
            sys.exit(1)
            
        process_single_row(row_index, mode, csv_type)
    except ValueError:
        print("ERROR: Row index must be an integer")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Error processing row {row_index}: {e}")
        sys.exit(1)
