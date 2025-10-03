import time
from io import BytesIO
import requests
from PIL import Image
import base64
from openai import OpenAI

# Point the OpenAI client to your local vLLM server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def profile_vlm():
    url = "https://h2o-release.s3.amazonaws.com/h2ogpt/bigben.jpg"
    image = Image.open(BytesIO(requests.get(url).content))
    b64 = pil_to_base64(image)

    # ChatML style message with an image_url payload
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "What is shown in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ]}
    ]

    # Sampling params (adjust as needed)
    params = dict(
        model="llava-hf/llava-v1.6-mistral-7b-hf",
        messages=messages,
        temperature=0.8,
        top_p=0.95,
        max_tokens=100,
        stream=True,
    )

    # Metrics
    t0 = time.perf_counter()
    first_token_time = None
    last_token_time = None
    token_timestamps = []
    num_output_tokens = 0

    # Stream tokens and record per-chunk time
    reply_text = []
    for event in client.chat.completions.create(**params):
        now = time.perf_counter()
        if event.choices and event.choices[0].delta and event.choices[0].delta.content:
            # Count rough “tokens” by chunks (you’ll refine with tokenizer if needed)
            chunk = event.choices[0].delta.content
            reply_text.append(chunk)
            num_output_tokens += 1
            token_timestamps.append(now)
            if first_token_time is None:
                first_token_time = now
        # End of stream signal (for some servers)
        if getattr(event, "choices", None) and getattr(event.choices[0], "finish_reason", None):
            last_token_time = now

    # Fall back if finish_reason didn’t set last_token_time
    if last_token_time is None:
        last_token_time = token_timestamps[-1] if token_timestamps else time.perf_counter()

    e2e = last_token_time - t0
    ttft = (first_token_time - t0) if first_token_time else e2e

    # Inter-token latency (ITL): average gap after first token
    itl = None
    if len(token_timestamps) >= 2:
        gaps = [token_timestamps[i] - token_timestamps[i - 1] for i in range(1, len(token_timestamps))]
        itl = sum(gaps) / len(gaps)

    # TPOT (time per output token): mean time from first token to end / (#tokens-1)
    tpot = None
    if num_output_tokens >= 2:
        tpot = (last_token_time - first_token_time) / (num_output_tokens - 1)

    # Throughput: tokens per second over the decode phase (after first token)
    throughput = None
    if num_output_tokens >= 2:
        decode_time = last_token_time - first_token_time
        throughput = (num_output_tokens - 1) / decode_time if decode_time > 0 else None

    print("==========================================================")
    print("Output:", "".join(reply_text))
    print(f"TTFT:        {ttft:.4f} s")
    print(f"ITL:         {itl:.4f} s" if itl is not None else "ITL:         N/A")
    print(f"TPOT:        {tpot:.4f} s" if tpot is not None else "TPOT:        N/A")
    print(f"E2E latency: {e2e:.4f} s")
    print(f"Throughput:  {throughput:.2f} tok/s" if throughput is not None else "Throughput:  N/A")

if __name__ == "__main__":
    profile_vlm()

