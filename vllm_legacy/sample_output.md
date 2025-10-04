`$ python3 vlm_inference_with_vllm.py`

``` 
INFO 09-23 18:11:18 [__init__.py:216] Automatically detected platform cuda.
INFO 09-23 18:11:24 [utils.py:328] non-default args: {'max_model_len': 4096, 'disable_log_stats': True, 'model': 'llava-hf/llava-v1.6-mistral-7b-hf'}
INFO 09-23 18:11:45 [__init__.py:742] Resolved architecture: LlavaNextForConditionalGeneration
INFO 09-23 18:11:48 [__init__.py:1815] Using max model len 4096
INFO 09-23 18:11:54 [scheduler.py:222] Chunked prefill is enabled with max_num_batched_tokens=16384.
/home/esp/.local/lib/python3.10/site-packages/mistral_common/protocol/instruct/messages.py:74: FutureWarning: ImageChunk has moved to 'mistral_common.protocol.instruct.chunk'. It will be removed from 'mistral_common.protocol.instruct.messages' in 1.10.0.
  warnings.warn(msg, FutureWarning)
WARNING 09-23 18:12:00 [__init__.py:2974] We must use the `spawn` multiprocessing start method. Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing for more information. Reasons: CUDA is initialized
INFO 09-23 18:12:03 [__init__.py:216] Automatically detected platform cuda.
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:05 [core.py:654] Waiting for init message from front-end.
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:05 [core.py:76] Initializing a V1 LLM engine (v0.10.2) with config: model='llava-hf/llava-v1.6-mistral-7b-hf', speculative_config=None, tokenizer='llava-hf/llava-v1.6-mistral-7b-hf', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=llava-hf/llava-v1.6-mistral-7b-hf, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2","vllm.mamba_mixer","vllm.short_conv","vllm.linear_attention","vllm.plamo2_mamba_mixer","vllm.gdn_attention"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"cudagraph_mode":1,"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"pass_config":{},"max_capture_size":512,"local_cache_dir":null}
[W923 18:12:08.570831248 ProcessGroupNCCL.cpp:981] Warning: TORCH_NCCL_AVOID_RECORD_STREAMS is the default now, this environment variable is thus deprecated. (function operator())
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:08 [parallel_state.py:1165] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
(EngineCore_DP0 pid=3021) /home/esp/.local/lib/python3.10/site-packages/mistral_common/protocol/instruct/messages.py:74: FutureWarning: ImageChunk has moved to 'mistral_common.protocol.instruct.chunk'. It will be removed from 'mistral_common.protocol.instruct.messages' in 1.10.0.
(EngineCore_DP0 pid=3021)   warnings.warn(msg, FutureWarning)
(EngineCore_DP0 pid=3021) WARNING 09-23 18:12:09 [topk_topp_sampler.py:69] FlashInfer is not available. Falling back to the PyTorch-native implementation of top-p & top-k sampling. For the best performance, please install FlashInfer.
(EngineCore_DP0 pid=3021) Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:10 [gpu_model_runner.py:2338] Starting to load model llava-hf/llava-v1.6-mistral-7b-hf...
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:10 [gpu_model_runner.py:2370] Loading model from scratch...
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:10 [cuda.py:362] Using Flash Attention backend on V1 engine.
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:11 [weight_utils.py:348] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:00,  4.25it/s]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:03<00:03,  1.81s/it]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:06<00:02,  2.36s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:09<00:00,  2.60s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:09<00:00,  2.29s/it]
(EngineCore_DP0 pid=3021) 
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:21 [default_loader.py:268] Loading weights took 9.23 seconds
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:21 [gpu_model_runner.py:2392] Model loading took 14.0786 GiB and 10.247948 seconds
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:22 [gpu_model_runner.py:3000] Encoder cache will be initialized with a budget of 16384 tokens, and profiled with 5 image items of the maximum feature size.
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:27 [backends.py:539] Using cache directory: /home/esp/.cache/vllm/torch_compile_cache/e33390f0f5/rank_0_0/backbone for vLLM's torch.compile
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:27 [backends.py:550] Dynamo bytecode transform time: 4.99 s
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:30 [backends.py:161] Directly load the compiled graph(s) for dynamic shape from the cache, took 2.810 s
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:31 [monitor.py:34] torch.compile takes 4.99 s in total
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:33 [gpu_worker.py:298] Available KV cache memory: 55.15 GiB
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:34 [kv_cache_utils.py:864] GPU KV cache size: 451,744 tokens
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:34 [kv_cache_utils.py:868] Maximum concurrency for 4,096 tokens per request: 110.29x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|█| 67/67 [00:02<00:00, 23
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:39 [gpu_model_runner.py:3118] Graph capturing finished in 5 secs, took 0.67 GiB
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:39 [gpu_worker.py:391] Free memory on device (78.66/79.19 GiB) on startup. Desired GPU memory utilization is (0.9, 71.27 GiB). Actual usage is 14.08 GiB for weight, 1.97 GiB for peak activation, 0.07 GiB for non-torch memory, and 0.67 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=58333988249` to fit into requested memory, or `--kv-cache-memory=66273897472` to fully utilize gpu memory. Current kv cache memory in use is 59212694937 bytes.
(EngineCore_DP0 pid=3021) INFO 09-23 18:12:39 [core.py:218] init engine (profile, create kv cache, warmup model) took 17.82 seconds
INFO 09-23 18:12:40 [llm.py:295] Supported_tasks: ['generate']
INFO 09-23 18:12:40 [__init__.py:36] No IOProcessor plugins requested by the model
Adding requests:   0%|                                           | 0/1 [00:00<?, ?it/s]Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
Adding requests: 100%|███████████████████████████████████| 1/1 [00:00<00:00,  1.13it/s]
Processed prompts: 100%|█| 1/1 [00:01<00:00,  1.26s/it, est. speed input: 1731.16 toks/
==========================================================
LLM output: The image shows a nighttime scene of the iconic Big Ben clock tower, also known as the Elizabeth Tower, which is located at the north end of the Palace of Westminster in London, UK. The tower is illuminated and stands out against the dark sky. In the foreground, there is a blurred view of traffic on a busy city street, which gives the impression of the city's hustle and bustle. The lights from the vehicles and streetlights add
[rank0]:[W923 18:12:43.978839087 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
```
