===========================================================================
Model: llava-hf/llava-1.5-7b-hf
Device: auto
Batch Size: 2
Image Directory: sample_images
Max Images: 10
Token Limits: 1024 input, 1024 output

No image files found in sample_images
Downloading sample images from HuggingFace to sample_images...
Trying dataset: huggan/smithsonian_butterflies_subset
`trust_remote_code` is not supported anymore.
Please check that the Hugging Face dataset 'huggan/smithsonian_butterflies_subset' isn't based on a loading script and remove `trust_remote_code`.
If the dataset is based on a loading script, please ask the dataset author to remove it and convert it to a standard format like Parquet.
README.md: 100%|██████████████████████| 609/609 [00:00<00:00, 110kB/s]
Repo card metadata block was not found. Setting CardData to empty.
dataset_infos.json: 1.65kB [00:00, 949kB/s]
data/train-00000-of-00001.parquet: 100%|█| 237M/237M [00:03<00:00, 69.
Generating train split: 100%|█| 1000/1000 [00:00<00:00, 3478.21 exampl
Successfully found dataset: huggan/smithsonian_butterflies_subset
`trust_remote_code` is not supported anymore.
Please check that the Hugging Face dataset 'huggan/smithsonian_butterflies_subset' isn't based on a loading script and remove `trust_remote_code`.
If the dataset is based on a loading script, please ask the dataset author to remove it and convert it to a standard format like Parquet.
Repo card metadata block was not found. Setting CardData to empty.
Loaded 20 samples from huggan/smithsonian_butterflies_subset
  Downloaded: hf_image_000.jpg
  Downloaded: hf_image_001.jpg
  Downloaded: hf_image_002.jpg
  Downloaded: hf_image_003.jpg
  Downloaded: hf_image_004.jpg
  Downloaded: hf_image_005.jpg
  Downloaded: hf_image_006.jpg
  Downloaded: hf_image_007.jpg
  Downloaded: hf_image_008.jpg
  Downloaded: hf_image_009.jpg
Successfully downloaded 10 images from HuggingFace dataset 'huggan/smithsonian_butterflies_subset'
Initializing model: llava-hf/llava-1.5-7b-hf
Max input tokens: 1024
Max output tokens: 1024
Loading model: llava-hf/llava-1.5-7b-hf
Using CUDA device: NVIDIA H100 PCIe
Loading LLaVA 1.5 model...
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|████████| 3/3 [00:04<00:00,  1.49s/it]
Successfully loaded llava-hf/llava-1.5-7b-hf
Model initialization time: 6.6818 seconds

Running inference on 10 images in 5 batches of size 2
============================================================
Processing batch 1/5 (2 images)
    Image 1: 601 input tokens, 66 output tokens
    Image 2: 601 input tokens, 96 output tokens
  Batch 1 completed in 7.0672s (+0.0022s loading)
    Total tokens: 1202 input, 162 output
Processing batch 2/5 (2 images)
    Image 1: 601 input tokens, 92 output tokens
    Image 2: 601 input tokens, 72 output tokens
  Batch 2 completed in 6.5969s (+0.0021s loading)
    Total tokens: 1202 input, 164 output
Processing batch 3/5 (2 images)
    Image 1: 601 input tokens, 85 output tokens
    Image 2: 601 input tokens, 80 output tokens
  Batch 3 completed in 6.7302s (+0.0022s loading)
    Total tokens: 1202 input, 165 output
Processing batch 4/5 (2 images)
    Image 1: 601 input tokens, 85 output tokens
    Image 2: 601 input tokens, 77 output tokens
  Batch 4 completed in 6.6349s (+0.0024s loading)
    Total tokens: 1202 input, 162 output
Processing batch 5/5 (2 images)
    Image 1: 601 input tokens, 75 output tokens
    Image 2: 601 input tokens, 87 output tokens
  Batch 5 completed in 6.5464s (+0.0023s loading)
    Total tokens: 1202 input, 162 output

============================================================
PERFORMANCE METRICS
============================================================
Model: llava-hf/llava-1.5-7b-hf
Device: cuda
Total Images: 10
Batch Size: 2
Number of Batches: 5
Max Input/Output Tokens: 1024/1024

TIMING METRICS:
  Model Initialization: 6.6818 seconds
  Total Inference Time: 33.5870 seconds
  Average Time per Image: 3.3587 seconds
  Images per Second: 0.30

TOKEN METRICS:
  Total Input Tokens: 6010
  Total Output Tokens: 815
  Average Input Tokens per Response: 601.00
  Average Output Tokens per Response: 81.50

BATCH STATISTICS:
  Min Batch Time: 6.5464 seconds
  Max Batch Time: 7.0672 seconds
  Average Batch Time: 6.7151 seconds
  Median Batch Time: 6.6349 seconds
  Std Dev Batch Time: 0.2080 seconds

SYSTEM INFO:
  CPU Cores: 64
  Total Memory: 61.839664459228516
  GPU 0: NVIDIA H100 PCIe
    Memory: 80616 MB
