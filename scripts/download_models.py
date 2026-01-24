import os
import sys
import shutil
import urllib.request
from huggingface_hub import snapshot_download

# Define the models identified in the codebase
# Key: HuggingFace Repo ID
models_to_download = [
    # SegFormer for BEV Reconstruction
    "nvidia/segformer-b0-finetuned-ade-512-512",
    
    # Depth Anything V3 (referenced in da3_reconstruction.py)
    "depth-anything/DA3NESTED-GIANT-LARGE",
    
    # Text Encoders for Crafter/LVDM
    "google/t5-v1_1-large",
    "openai/clip-vit-large-patch14",
]

# Set the download directory
download_root = os.path.abspath("./checkpoints")
os.makedirs(download_root, exist_ok=True)

print(f"Downloading HuggingFace models to {download_root}...")

for model_id in models_to_download:
    print(f"\n[Processing] {model_id}")
    try:
        # local_dir_use_symlinks=False ensures actual files are downloaded
        local_model_path = os.path.join(download_root, model_id.replace("/", "_"))
        
        snapshot_download(
            repo_id=model_id, 
            local_dir=local_model_path, 
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
        )
        print(f"[Success] Downloaded to {local_model_path}")
    except Exception as e:
        print(f"[Error] Failed to download {model_id}: {e}")

print("\n---------------------------------------------------------")
print("Downloading GeoCalib Model...")
# GeoCalib is downloaded via torch.hub from GitHub Releases
geocalib_url = "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-distorted.tar"
geocalib_dir = os.path.join(download_root, "geocalib")
os.makedirs(geocalib_dir, exist_ok=True)
geocalib_file = os.path.join(geocalib_dir, "geocalib-distorted.tar")

try:
    if not os.path.exists(geocalib_file):
        print(f"Downloading from {geocalib_url}...")
        urllib.request.urlretrieve(geocalib_url, geocalib_file)
        print(f"[Success] Downloaded to {geocalib_file}")
    else:
        print(f"[Skipped] File already exists: {geocalib_file}")
except Exception as e:
    print(f"[Error] Failed to download GeoCalib: {e}")

print("\n---------------------------------------------------------")
print("Manual Download Check:")
print("1. MASt3R Checkpoint:")
print("   Ensure 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth' exists.")
print("   (It is usually manually placed in the project structure, not downloaded from HF automatically)")

print("\n--- How to use in Offline Mode ---")
print(f"1. Copy the '{download_root}' folder to your offline machine.")
print("\n[For HuggingFace Models]")
print("   Update the code path (e.g. from_pretrained('nvidia/segformer...')) to point to:")
print(f"   '{download_root}/nvidia_segformer-b0-finetuned-ade-512-512'")
print("\n[For GeoCalib]")
print("   The code looks in torch.hub.get_dir()/geocalib.")
print("   On the offline machine, create the directory:")
print("   mkdir -p ~/.cache/torch/hub/geocalib (or wherever torch.hub.get_dir() points)")
print(f"   and copy '{geocalib_file}' into it, renaming it to 'distorted.tar'.")
