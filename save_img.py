import os
import requests

# 5 sample images from the COCO 2017 validation set
image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "https://farm9.staticflickr.com/8118/8657457305_b14dcf0f91_z.jpg",
    "https://farm5.staticflickr.com/4058/4605161901_6d4ff9de88_z.jpg",
    "https://farm3.staticflickr.com/2823/9728433412_8a98d86eff_z.jpg",
    "https://farm7.staticflickr.com/6086/6092606616_89751b1c2f_z.jpg",
]

# Directory to save
save_dir = "downloaded_images"
os.makedirs(save_dir, exist_ok=True)

# Download and save as i1.jpg ... i5.jpg
for i, url in enumerate(image_urls, start=1):
    response = requests.get(url)
    with open(os.path.join(save_dir, f"i{i}.jpg"), "wb") as f:
        f.write(response.content)
    print(f"Saved i{i}.jpg")
