import time
from pathlib import Path
from PIL import Image
from vllm import LLM, SamplingParams

def run_llava_on_images():
    llm = LLM(model="llava-hf/llava-v1.6-vicuna-7b-hf", max_model_len=4096, disable_log_stats=False)
    prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1
    )

    # Find images named i1.jpeg, i2.jpeg, ..., i5.jpeg in current directory
    image_files = sorted(Path(".").glob("downloaded_images/i[1-5].jpg"))

    if not image_files:
        print("No images found matching i[1-5].jpeg in the current directory.")
        return
    

    for img_path in image_files:
        image = Image.open(img_path)
        t0 = time.perf_counter()
        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            },
            sampling_params=sampling_params
        )
        t1 = time.perf_counter()

        generated_text = "".join(o.outputs[0].text for o in outputs)
        print(f"Image: {img_path.name}")
        print(f"Time taken: {t1 - t0:.4f} seconds")
        print("==========================================================")
        print(f"LLM output: {generated_text}")
        print("\n")

if __name__ == "__main__":
    run_llava_on_images()
