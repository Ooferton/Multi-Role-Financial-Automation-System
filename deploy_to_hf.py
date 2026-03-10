import os
import sys
from huggingface_hub import HfApi

def deploy(token, repo_id):
    api = HfApi()
    
    print(f"Starting deployment to Hugging Face Space: {repo_id}")
    
    # Files/folders to ignore
    ignore_patterns = [
        "__pycache__",
        ".git",
        ".env",
        "data/*.db",
        "data/*.json",
        "data/*.csv",
        "logs/*",
        "*.log",
        ".openclaw/*",
        "*.lock"
    ]
    
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            token=token,
            ignore_patterns=ignore_patterns,
            delete_patterns=["*"] # Clean the space before upload (optional, use with care)
        )
        print("Deployment SUCCESSFUL!")
        print(f"View your Space: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Deployment FAILED: {e}")

if __name__ == "__main__":
    # In a real scenario, these would be passed or read from environment
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO_ID")
    
    if not hf_token or not hf_repo:
        print("Error: HF_TOKEN and HF_REPO_ID environment variables must be set.")
        print("Usage (Windows): $env:HF_TOKEN='...'; $env:HF_REPO_ID='...'; python deploy_to_hf.py")
        sys.exit(1)
        
    deploy(hf_token, hf_repo)
