```
$ cd /home/esp/VLM_vLLM && timeout 30 python3 multimodal_inference_benchmark_hf_fixed.py --sequence-len 900 --max-images 1 --batch-size 1
```
```
Multimodal Model Inference Benchmark with Enhanced TTFT/TPOT Metrics
===========================================================================
Model: llava-hf/llava-1.5-7b-hf
Device: auto
Batch Size: 1
Image Directory: sample_images
Max Images: 1
Target Sequence Length: 900 tokens

Found 1 images in sample_images
Initializing model: llava-hf/llava-1.5-7b-hf
Loading model: llava-hf/llava-1.5-7b-hf
Using CUDA device: NVIDIA H100 PCIe
Loading LLaVA 1.5 model...
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████| 3/3 [00:04<00:00,  1.44s/it]
Successfully loaded llava-hf/llava-1.5-7b-hf
Model initialization time: 6.9130 seconds

Running inference on 1 images in 1 batches of size 1
============================================================
Processing batch 1/1 (1 images)
    Original tokens: 601, Target: 900
    Extending to exactly 900 tokens
    Final length: 900 (target: 900)
    Image 1: 900 input, 12 output tokens
              TTFT: 0.1594s, TPOT: 0.1304s
  Batch 1 completed in 1.6222s
    Tokens: 900 input, 12 output
    Avg TTFT: 0.1594s
    Avg TPOT: 0.1304s

============================================================
PERFORMANCE METRICS
============================================================
Model: llava-hf/llava-1.5-7b-hf
Device: cuda
Total Images: 1
Batch Size: 1
Number of Batches: 1

TIMING METRICS:
  Model Initialization: 6.9130 seconds
  Total Inference Time: 1.6333 seconds
  Average Time per Image: 1.6333 seconds
  Images per Second: 0.61

TOKEN METRICS:
  Total Input Tokens: 900
  Total Output Tokens: 12
  Average Input Tokens per Response: 900.00
  Average Output Tokens per Response: 12.00

TTFT (Time to First Token) METRICS:
  Average TTFT: 0.1594 seconds
  Median TTFT: 0.1594 seconds
  Min TTFT: 0.1594 seconds
  Max TTFT: 0.1594 seconds

TPOT (Time Per Output Token) METRICS:
  Average TPOT: 0.1304 seconds
  Median TPOT: 0.1304 seconds
  Min TPOT: 0.1304 seconds
  Max TPOT: 0.1304 seconds

BATCH STATISTICS:
  Min Batch Time: 1.6222 seconds
  Max Batch Time: 1.6222 seconds
  Average Batch Time: 1.6222 seconds
  Median Batch Time: 1.6222 seconds
  Std Dev Batch Time: 0.0000 seconds

SYSTEM INFO:
  CPU Cores: 64
  Total Memory: 61.839664459228516
  GPU 0: NVIDIA H100 PCIe
    Memory: 80616 MB
```
