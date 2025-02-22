import os
import requests

def download_file(url, save_path):
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Stream download the file
    with requests.get(url, stream=True) as response:
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Downloaded: {save_path}")
        else:
            print(f"Failed to download. HTTP Status Code: {response.status_code}")

# Example: TU Dortmund PROTEINS dataset
url = "https://www.chrsmrrs.com/graphkerneldatasets/PROTEINS.zip"
save_path = "/root/gmb/dataset/PROTEIN/PROTEINS.zip"

download_file(url, save_path)

