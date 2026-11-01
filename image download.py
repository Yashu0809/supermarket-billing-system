import os
import pandas as pd
from icrawler.builtin import GoogleImageCrawler

# Load the CSV
df = pd.read_csv("sample products.csv")

# Make sure output directory exists
base_dir = "products_images"
os.makedirs(base_dir, exist_ok=True)

# Download high-quality images for each product
for product in df['product_name'].dropna().unique():
    folder_name = product.strip().replace(" ", "").replace("/", "")
    product_dir = os.path.join(base_dir, folder_name)
    if os.path.exists(product_dir) and len(os.listdir(product_dir)) >= 40:
        print(f"✔ Already downloaded: {product}")
        continue
    
    print(f"🔍 Downloading: {product}")
    os.makedirs(product_dir, exist_ok=True)

    crawler = GoogleImageCrawler(storage={'root_dir': product_dir})
    crawler.crawl(keyword=f"{product} clear image of the product", 
                  max_num=40, min_size=(200, 200), file_idx_offset=0)