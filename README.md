# Oct 3 Fri

- Add 'multimodal*'.py

```
pip install datasets
pip install accelerate
pip install transformers torch Pillow
```




# Wednes

vLLM doesn't support QWen series.
!link[https://docs.vllm.ai/en/v0.5.3/models/vlm.html]

Look at the sample output

# Example for VLM inference


https://docs.vllm.ai/en/v0.5.3/models/vlm.html


https://huggingface.co/datasets/HuggingFaceM4/FineVision?library=datasets


# Streaming example
vllm serve llava-hf/llava-v1.6-mistral-7b-hf \
  --port 8000 \
  --max-model-len 4096 \
  --dtype auto \
  --gpu-memory-utilization 0.9 


```
esp@esp:~/VLM_vLLM$ python3 vlm_inference_with_vllm_streaming.py 
==========================================================
Output:  The image shows a nighttime scene of the iconic Big Ben clock tower, also known as the Elizabeth Tower, which is located at the north end of the Palace of Westminster in London, UK. The tower is illuminated and stands out against the dark sky. In the foreground, there is a blurred view of traffic on a busy city street, which conveys a sense of motion and the hustle and bustle of urban life. The starry sky in the
TTFT:        1.7901 s
ITL:         0.0111 s
TPOT:        0.0111 s
E2E latency: 2.8871 s
Throughput:  90.24 tok/s
```
