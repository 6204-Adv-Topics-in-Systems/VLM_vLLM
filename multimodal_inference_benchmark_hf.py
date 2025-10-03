#!/usr/bin/env python3
"""
Multimodal Model Inference Benchmark Script using Hugging Face Transformers

This script runs multimodal model inference using Hugging Face Transformers directly,
with configurable batch sizes and model names. It includes comprehensive performance
metrics tracking and can download sample images from Hugging Face datasets.
"""

import argparse
import json
import time
import os
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics
import warnings

# Handle optional imports gracefully
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("PIL not available. Image processing will be limited.")

try:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaNextProcessor, LlavaNextForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. Please install transformers package.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. System monitoring will be limited.")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    warnings.warn("GPUtil not available. GPU monitoring will be limited.")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    warnings.warn("datasets package not available. HF dataset downloading will be limited.")


def get_system_info():
    """Get system information for benchmarking context."""
    info = {
        "cpu_count": "unknown",
        "memory_total_gb": "unknown",
        "gpus": []
    }
    
    if PSUTIL_AVAILABLE:
        info.update({
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3)
        })
    
    if GPUTIL_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "memory_util": gpu.memoryUtil,
                    "load": gpu.load
                })
            info["gpus"] = gpu_info
        except Exception as e:
            warnings.warn(f"Error getting GPU info: {e}")
    
    return info


def download_sample_images_from_hf(output_dir: str, max_images: int = 20) -> List[Path]:
    """Download sample images from Hugging Face datasets."""
    if not DATASETS_AVAILABLE:
        print("Error: datasets package not available. Please install with: pip install datasets")
        return []
    
    if not PIL_AVAILABLE:
        print("Error: PIL not available. Please install with: pip install Pillow")
        return []
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Downloading sample images to {output_dir}...")
    
    # Try different datasets that contain images
    datasets_to_try = [
        ("food101", "image"),
        ("cifar10", "img"),
        ("imagenet-1k", "image"),
    ]
    
    downloaded_files = []
    
    for dataset_name, image_column in datasets_to_try:
        try:
            print(f"Trying to load dataset: {dataset_name}")
            # Try to load a small subset
            if dataset_name == "food101":
                dataset = load_dataset(dataset_name, split="train", streaming=True)
            elif dataset_name == "cifar10":
                dataset = load_dataset(dataset_name, split="train", streaming=True)
            else:
                continue  # Skip problematic datasets
            
            count = 0
            for i, item in enumerate(dataset):
                if count >= max_images:
                    break
                
                try:
                    if image_column in item and item[image_column] is not None:
                        image = item[image_column]
                        if hasattr(image, 'convert'):  # PIL Image
                            image_path = output_path / f"sample_{count:03d}_from_{dataset_name.replace('/', '_')}.jpg"
                            image.convert('RGB').save(image_path)
                            downloaded_files.append(image_path)
                            count += 1
                            print(f"Downloaded {count}/{max_images}: {image_path.name}")
                except Exception as e:
                    print(f"Error processing image {i} from {dataset_name}: {e}")
                    continue
            
            if downloaded_files:
                print(f"Successfully downloaded {len(downloaded_files)} images from {dataset_name}")
                break
                
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            continue
    
    if not downloaded_files:
        # Fallback: download some common test images from URLs
        print("Fallback: downloading test images from URLs...")
        test_urls = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/800px-Camponotus_flavomarginatus_ant.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Bucephala-albeola-010.jpg/800px-Bucephala-albeola-010.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
        ]
        
        for i, url in enumerate(test_urls[:max_images]):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    image_path = output_path / f"test_image_{i:03d}.jpg"
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Verify it's a valid image
                    if PIL_AVAILABLE:
                        img = Image.open(image_path)
                        img.verify()
                        # Reopen and convert to RGB
                        img = Image.open(image_path).convert('RGB')
                        img.save(image_path)
                        downloaded_files.append(image_path)
                        print(f"Downloaded: {image_path.name}")
                
            except Exception as e:
                print(f"Error downloading image from {url}: {e}")
                continue
    
    return downloaded_files


def load_images_from_directory(image_dir: str, max_images: int = None) -> List[Path]:
    """Load images from a directory, with option to download if directory doesn't exist."""
    image_dir_path = Path(image_dir)
    
    if not image_dir_path.exists():
        print(f"Image directory {image_dir} not found.")
        choice = input("Would you like to download sample images? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            download_count = max_images if max_images else 10
            downloaded_files = download_sample_images_from_hf(image_dir, download_count)
            if not downloaded_files:
                raise FileNotFoundError(f"Could not create or populate image directory: {image_dir}")
            return downloaded_files
        else:
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Support common image formats
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(image_dir_path.glob(ext))
        image_files.extend(image_dir_path.glob(ext.upper()))
    
    image_files = sorted(image_files)
    
    if max_images:
        image_files = image_files[:max_images]
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        choice = input("Would you like to download sample images to this directory? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            download_count = max_images if max_images else 10
            downloaded_files = download_sample_images_from_hf(image_dir, download_count)
            if not downloaded_files:
                raise ValueError(f"Could not find or download images for {image_dir}")
            return downloaded_files
        else:
            raise ValueError(f"No image files found in {image_dir}")
    
    print(f"Found {len(image_files)} images in {image_dir}")
    return image_files


def create_batches(items: List, batch_size: int) -> List[List]:
    """Split items into batches of specified size."""
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def load_model_and_processor(model_name: str, device: str = "auto"):
    """Load the multimodal model and processor."""
    print(f"Loading model: {model_name}")
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            print("Using CPU device")
    
    try:
        # Try to load with AutoProcessor first
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            model = model.to(device)
            
        print(f"Successfully loaded {model_name} with AutoProcessor")
        return model, processor, device
        
    except Exception as e1:
        print(f"AutoProcessor failed: {e1}")
        
        # Fallback: Try LLaVA specific processors
        try:
            if "llava" in model_name.lower():
                processor = LlavaNextProcessor.from_pretrained(model_name)
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device if device == "cuda" else None
                )
                
                if device == "cpu":
                    model = model.to(device)
                    
                print(f"Successfully loaded {model_name} with LlavaNextProcessor")
                return model, processor, device
                
        except Exception as e2:
            print(f"LlavaNextProcessor also failed: {e2}")
            raise Exception(f"Could not load model {model_name}. Errors: {e1}, {e2}")


def run_multimodal_inference(
    model_name: str,
    image_files: List[Path],
    batch_size: int,
    prompt: str = None,
    max_tokens: int = 100,
    temperature: float = 0.8,
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Run multimodal inference on a set of images with specified batch size using Hugging Face Transformers.
    """
    
    if not TRANSFORMERS_AVAILABLE:
        return {"error": "Transformers not available. Please install transformers package."}
    
    if not PIL_AVAILABLE:
        return {"error": "PIL not available. Please install Pillow package."}
    
    if prompt is None:
        prompt = "What is shown in this image? Describe it in detail."
    
    print(f"Initializing model: {model_name}")
    model_init_start = time.perf_counter()
    
    try:
        model, processor, actual_device = load_model_and_processor(model_name, device)
    except Exception as e:
        print(f"Error initializing model {model_name}: {e}")
        return {"error": str(e)}
    
    model_init_time = time.perf_counter() - model_init_start
    print(f"Model initialization time: {model_init_time:.4f} seconds")
    
    # Create batches
    batches = create_batches(image_files, batch_size)
    
    # Track metrics
    metrics = {
        "model_name": model_name,
        "device": actual_device,
        "total_images": len(image_files),
        "batch_size": batch_size,
        "num_batches": len(batches),
        "model_init_time": model_init_time,
        "batch_times": [],
        "total_inference_time": 0,
        "average_time_per_image": 0,
        "images_per_second": 0,
        "tokens_generated": 0,
        "average_tokens_per_response": 0,
        "system_info": get_system_info()
    }
    
    print(f"\nRunning inference on {len(image_files)} images in {len(batches)} batches of size {batch_size}")
    print("=" * 60)
    
    all_outputs = []
    total_inference_start = time.perf_counter()
    
    for batch_idx, batch_files in enumerate(batches):
        print(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_files)} images)")
        
        # Prepare batch
        batch_load_start = time.perf_counter()
        batch_images = []
        batch_prompts = []
        valid_files = []
        
        for img_path in batch_files:
            try:
                image = Image.open(img_path).convert("RGB")
                batch_images.append(image)
                batch_prompts.append(prompt)
                valid_files.append(img_path)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
        
        batch_load_time = time.perf_counter() - batch_load_start
        
        if not batch_images:
            print(f"No valid images in batch {batch_idx + 1}, skipping...")
            continue
        
        # Run batch inference
        batch_inference_start = time.perf_counter()
        
        try:
            # Process images and text
            if len(batch_images) == 1:
                # Single image processing
                inputs = processor(text=batch_prompts[0], images=batch_images[0], return_tensors="pt")
                if actual_device == "cuda":
                    inputs = {k: v.to(actual_device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True if temperature > 0 else False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                
                # Decode the response
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                # Remove the input prompt from the response
                if prompt in generated_text:
                    generated_text = generated_text.replace(prompt, "").strip()
                
                batch_outputs = [generated_text]
                
            else:
                # Batch processing
                batch_outputs = []
                for i, (image, text) in enumerate(zip(batch_images, batch_prompts)):
                    inputs = processor(text=text, images=image, return_tensors="pt")
                    if actual_device == "cuda":
                        inputs = {k: v.to(actual_device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            do_sample=True if temperature > 0 else False,
                            pad_token_id=processor.tokenizer.eos_token_id
                        )
                    
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    if prompt in generated_text:
                        generated_text = generated_text.replace(prompt, "").strip()
                    
                    batch_outputs.append(generated_text)
            
            batch_inference_time = time.perf_counter() - batch_inference_start
            
            # Process outputs
            batch_tokens = 0
            for i, (img_path, output_text) in enumerate(zip(valid_files, batch_outputs)):
                # Rough token count estimation
                token_count = len(output_text.split())
                batch_tokens += token_count
                
                all_outputs.append({
                    "image_path": str(img_path),
                    "generated_text": output_text,
                    "batch_idx": batch_idx,
                    "token_count": token_count
                })
            
            metrics["batch_times"].append({
                "batch_idx": batch_idx,
                "load_time": batch_load_time,
                "inference_time": batch_inference_time,
                "total_time": batch_load_time + batch_inference_time,
                "images_in_batch": len(batch_images),
                "tokens_generated": batch_tokens
            })
            
            metrics["tokens_generated"] += batch_tokens
            
            print(f"  Batch {batch_idx + 1} completed in {batch_inference_time:.4f}s "
                  f"(+{batch_load_time:.4f}s loading)")
            
        except Exception as e:
            print(f"Error during batch {batch_idx + 1} inference: {e}")
            metrics["batch_times"].append({
                "batch_idx": batch_idx,
                "error": str(e)
            })
    
    total_inference_time = time.perf_counter() - total_inference_start
    
    # Calculate final metrics
    metrics["total_inference_time"] = total_inference_time
    metrics["average_time_per_image"] = total_inference_time / len(image_files) if len(image_files) > 0 else 0
    metrics["images_per_second"] = len(image_files) / total_inference_time if total_inference_time > 0 else 0
    metrics["average_tokens_per_response"] = metrics["tokens_generated"] / len(image_files) if len(image_files) > 0 else 0
    
    # Calculate batch statistics
    valid_batch_times = [b["inference_time"] for b in metrics["batch_times"] if "inference_time" in b]
    if valid_batch_times:
        metrics["batch_stats"] = {
            "min_batch_time": min(valid_batch_times),
            "max_batch_time": max(valid_batch_times),
            "avg_batch_time": statistics.mean(valid_batch_times),
            "median_batch_time": statistics.median(valid_batch_times),
            "std_batch_time": statistics.stdev(valid_batch_times) if len(valid_batch_times) > 1 else 0
        }
    
    metrics["outputs"] = all_outputs
    
    return metrics


def print_metrics(metrics: Dict[str, Any]):
    """Print performance metrics in a formatted way."""
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    
    if "error" in metrics:
        print(f"ERROR: {metrics['error']}")
        return
    
    print(f"Model: {metrics['model_name']}")
    print(f"Device: {metrics['device']}")
    print(f"Total Images: {metrics['total_images']}")
    print(f"Batch Size: {metrics['batch_size']}")
    print(f"Number of Batches: {metrics['num_batches']}")
    print()
    
    print("TIMING METRICS:")
    print(f"  Model Initialization: {metrics['model_init_time']:.4f} seconds")
    print(f"  Total Inference Time: {metrics['total_inference_time']:.4f} seconds")
    print(f"  Average Time per Image: {metrics['average_time_per_image']:.4f} seconds")
    print(f"  Images per Second: {metrics['images_per_second']:.2f}")
    print()
    
    print("TOKEN METRICS:")
    print(f"  Total Tokens Generated: {metrics['tokens_generated']}")
    print(f"  Average Tokens per Response: {metrics['average_tokens_per_response']:.2f}")
    print()
    
    if "batch_stats" in metrics:
        batch_stats = metrics["batch_stats"]
        print("BATCH STATISTICS:")
        print(f"  Min Batch Time: {batch_stats['min_batch_time']:.4f} seconds")
        print(f"  Max Batch Time: {batch_stats['max_batch_time']:.4f} seconds")
        print(f"  Average Batch Time: {batch_stats['avg_batch_time']:.4f} seconds")
        print(f"  Median Batch Time: {batch_stats['median_batch_time']:.4f} seconds")
        print(f"  Std Dev Batch Time: {batch_stats['std_batch_time']:.4f} seconds")
        print()
    
    print("SYSTEM INFO:")
    sys_info = metrics["system_info"]
    print(f"  CPU Cores: {sys_info['cpu_count']}")
    print(f"  Total Memory: {sys_info['memory_total_gb']}")
    if sys_info["gpus"]:
        for i, gpu in enumerate(sys_info["gpus"]):
            print(f"  GPU {i}: {gpu['name']}")
            print(f"    Memory: {gpu['memory_used']}/{gpu['memory_total']} MB ({gpu['memory_util']*100:.1f}%)")
            print(f"    Load: {gpu['load']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Model Inference Benchmark using Hugging Face Transformers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="llava-hf/llava-1.5-7b-hf",
        help="Model name or path (HuggingFace transformers compatible)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=4,
        help="Batch size for inference"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference"
    )
    
    parser.add_argument(
        "--image-dir", 
        type=str, 
        default="sample_images",
        help="Directory containing images"
    )
    
    parser.add_argument(
        "--max-images", 
        type=int, 
        default=None,
        help="Maximum number of images to process"
    )
    
    parser.add_argument(
        "--download-images", 
        action="store_true",
        help="Force download sample images from Hugging Face datasets"
    )
    
    parser.add_argument(
        "--prompt", 
        type=str, 
        default=None,
        help="Custom prompt for the model"
    )
    
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=100,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.8,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--output-file", 
        type=str, 
        default=None,
        help="Output file to save results (JSON format)"
    )
    
    parser.add_argument(
        "--show-outputs", 
        action="store_true",
        help="Show generated text outputs"
    )
    
    args = parser.parse_args()
    
    print("Multimodal Model Inference Benchmark (Hugging Face Transformers)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Directory: {args.image_dir}")
    print(f"Max Images: {args.max_images}")
    print()
    
    # Check dependencies
    missing_deps = []
    if not TRANSFORMERS_AVAILABLE:
        missing_deps.append("transformers torch")
    if not PIL_AVAILABLE:
        missing_deps.append("Pillow")
    
    if missing_deps:
        print("ERROR: Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies:")
        print(f"pip install {' '.join(missing_deps)}")
        return 1
    
    try:
        # Handle image downloading/loading
        if args.download_images:
            download_count = args.max_images if args.max_images else 10
            image_files = download_sample_images_from_hf(args.image_dir, download_count)
            if not image_files:
                print("Failed to download sample images")
                return 1
        else:
            # Load images from directory (with interactive download option if needed)
            image_files = load_images_from_directory(args.image_dir, args.max_images)
        
        # Run inference
        metrics = run_multimodal_inference(
            model_name=args.model,
            image_files=image_files,
            batch_size=args.batch_size,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=args.device
        )
        
        # Print metrics
        print_metrics(metrics)
        
        # Show outputs if requested
        if args.show_outputs and "outputs" in metrics:
            print("\n" + "=" * 60)
            print("GENERATED OUTPUTS")
            print("=" * 60)
            for output in metrics["outputs"]:
                print(f"\nImage: {Path(output['image_path']).name}")
                print(f"Generated: {output['generated_text']}")
                print("-" * 40)
        
        # Save results if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output_file}")
            
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
