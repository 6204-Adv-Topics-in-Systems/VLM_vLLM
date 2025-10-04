```
$ python3 /home/esp/VLM_vLLM/multimodal_inference_benchmark_hf_fixed.py --max-images 1
```


```
Multimodal Model Inference Benchmark with Enhanced TTFT/TPOT Metrics
===========================================================================
Model: llava-hf/llava-1.5-7b-hf
Device: auto
Batch Size: 2
Image Directory: sample_images
Max Images: 1

Found 1 images in sample_images
Initializing model: llava-hf/llava-1.5-7b-hf
Loading model: llava-hf/llava-1.5-7b-hf
/home/esp/anaconda3/envs/pytorch-uvm/lib/python3.9/site-packages/torch/cuda/__init__.py:182: UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 802: system not yet initialized (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:109.)
  return torch._C._cuda_getDeviceCount() > 0
Using CPU device
Loading LLaVA 1.5 model...
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████| 3/3 [00:09<00:00,  3.13s/it]
Successfully loaded llava-hf/llava-1.5-7b-hf
Model initialization time: 10.5071 seconds

Running inference on 1 images in 1 batches of size 2
============================================================
Processing batch 1/1 (1 images)


    Image 1: 601 input, 92 output tokens
              TTFT: 16.5252s, TPOT: 1.6344s
  Batch 1 completed in 165.2700s
    Tokens: 601 input, 92 output
    Avg TTFT: 16.5252s
    Avg TPOT: 1.6344s

============================================================
PERFORMANCE METRICS
============================================================
Model: llava-hf/llava-1.5-7b-hf
Device: cpu
Total Images: 1
Batch Size: 2
Number of Batches: 1

TIMING METRICS:
  Model Initialization: 10.5071 seconds
  Total Inference Time: 165.3002 seconds
  Average Time per Image: 165.3002 seconds
  Images per Second: 0.01

TOKEN METRICS:
  Total Input Tokens: 601
  Total Output Tokens: 92
  Average Input Tokens per Response: 601.00
  Average Output Tokens per Response: 92.00

TTFT (Time to First Token) METRICS:
  Average TTFT: 16.5252 seconds
  Median TTFT: 16.5252 seconds
  Min TTFT: 16.5252 seconds
  Max TTFT: 16.5252 seconds

TPOT (Time Per Output Token) METRICS:
  Average TPOT: 1.6344 seconds
  Median TPOT: 1.6344 seconds
  Min TPOT: 1.6344 seconds
  Max TPOT: 1.6344 seconds

BATCH STATISTICS:
  Min Batch Time: 165.2700 seconds
  Max Batch Time: 165.2700 seconds
  Average Batch Time: 165.2700 seconds
  Median Batch Time: 165.2700 seconds
  Std Dev Batch Time: 0.0000 seconds

SYSTEM INFO:
  CPU Cores: 64
  Total Memory: 61.839664459228516
```
