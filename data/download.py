import os
import requests

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"File downloaded: {save_path}")
    else:
        print(f"Failed to download file from {url}")

if __name__ == "__main__":
    datasets = {
        "graph1": "https://example.com/graph1.edgelist",
        "graph2": "https://example.com/graph2.edgelist",
    }

    for name, url in datasets.items():
        save_path = f"data/raw/{name}.edgelist"
        download_file(url, save_path)