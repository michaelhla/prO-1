import os
import subprocess
import sys
from dotenv import load_dotenv
from huggingface_hub import login, upload_folder, snapshot_download

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def ensure_hf_hub():
    try:
        import huggingface_hub
    except ImportError:
        install_package("huggingface_hub")

def sync_with_hf_hub(local_path, repo_id, upload=False, subfolder=None):
    """
    Syncs model with Hugging Face Hub.
    If upload=True, uploads from local_path to hub.
    If upload=False, downloads from hub to local_path.
    
    Args:
        local_path: Local path to model directory
        repo_id: Hugging Face repo ID
        upload: Whether to upload or download
        subfolder: Optional subfolder within the repo to sync
    """
    ensure_hf_hub()
    
    # Load API key from .env
    load_dotenv()
    hf_token = os.getenv('HUGGINGFACE_API_KEY')
    
    if not hf_token:
        raise ValueError("HUGGINGFACE_API_KEY not found in .env file")
        
    # Login to Hugging Face
    login(token=hf_token)
    
    if upload:
        # Upload to Hub
        upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            path_in_repo=subfolder,  # Specify subfolder in repo
            commit_message=f"Upload model from {local_path}"
        )
        print(f"Uploaded {local_path} to {repo_id}/{subfolder if subfolder else ''}")
    else:
        # Download from Hub
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_path,
            allow_patterns=f"{subfolder}/**" if subfolder else None,  # Only download files in subfolder
            token=hf_token
        )
        print(f"Downloaded {repo_id}/{subfolder if subfolder else ''} to {local_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Sync with Hugging Face Hub')
    parser.add_argument('local_path', help='Local path to model directory')
    parser.add_argument('-u', '--upload', action='store_true', help='Upload to Hub')
    parser.add_argument('-d', '--download', action='store_true', help='Download from Hub')
    
    args = parser.parse_args()
    
    if args.upload and args.download:
        raise ValueError("Cannot specify both upload and download")
    elif not (args.upload or args.download):
        raise ValueError("Must specify either upload (-u) or download (-d)")
    
    # Use the local_path's directory name as the subfolder name
    subfolder = os.path.basename(os.path.normpath(args.local_path))
    sync_with_hf_hub(
        local_path=args.local_path, 
        repo_id="mhla/prO-1", 
        upload=args.upload,
        subfolder=subfolder
    )
