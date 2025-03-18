import os
import wget
import zipfile
import shutil
from pathlib import Path

def download_and_setup_models():
    # Define paths
    current_dir = Path(__file__).parent
    infer_dir = current_dir
    
    # URLs for models (replace with actual URLs)
    contentvec_url = "https://huggingface.co/Kasparasman/RVC/resolve/main/pytorch_model.zip"
    predictor_url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"

    # Handle ContentVec model
    try:
        # Download ContentVec
        print("Downloading ContentVec model...")
        zip_path = infer_dir / "contentvec.zip"
        wget.download(contentvec_url, str(zip_path))
        
        # Extract zip
        print("\nExtracting ContentVec model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(infer_dir)
        
        # Rename directory if needed
        extracted_dir = next(infer_dir.glob('*contentvec*'), None)
        target_dir = infer_dir / 'contentvec'
        if extracted_dir and extracted_dir.name != 'contentvec':
            if target_dir.exists():
                shutil.rmtree(target_dir)
            os.rename(extracted_dir, target_dir)
        
        # Remove zip file
        # Remove zip file
        zip_path.unlink()
        
        # Verify contentvec directory exists and is a directory
        target_dir = infer_dir / 'contentvec'
        if not target_dir.exists() or not target_dir.is_dir():
            raise ValueError("ContentVec model extraction failed - directory not found")
        print("ContentVec model setup complete.")

    except Exception as e:
        print(f"Error setting up ContentVec model: {str(e)}")

    # Handle Predictor model
    try:
        print("\nDownloading Predictor model...")
        predictor_dir = infer_dir / 'predictor'
        predictor_dir.mkdir(exist_ok=True)
        
        # Download predictor model
        predictor_path = predictor_dir / "predictor.pt"
        wget.download(predictor_url, str(predictor_path))
        
        # Verify if it's a .pt file
        if predictor_path.suffix != '.pt':
            raise ValueError("Downloaded predictor file is not a .pt file")
            
        print("\nPredictor model setup complete.")

    except Exception as e:
        print(f"Error setting up Predictor model: {str(e)}")

if __name__ == "__main__":
    download_and_setup_models()