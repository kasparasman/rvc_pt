import os
import wget
import zipfile
import shutil
from pathlib import Path

def download_and_setup_models():
    """Downloads and sets up required models for inference."""

    # Define paths
    current_dir = Path(__file__).parent
    infer_dir = current_dir  # Ensure models are inside the inference directory

    # Model URLs
    contentvec_url = "https://huggingface.co/Kasparasman/RVC/resolve/main/pytorch_model.zip"
    predictor_url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"

    # Handle ContentVec model setup
    try:
        contentvec_dir = infer_dir / "contentvec"
        zip_path = infer_dir / "contentvec.zip"

        # Download and extract only if the contentvec directory is missing
        if not contentvec_dir.exists():
            print("Downloading ContentVec model...")
            wget.download(contentvec_url, str(zip_path))

            print("\nExtracting ContentVec model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(infer_dir)

            # Ensure correct naming of extracted directory
            extracted_dirs = list(infer_dir.glob('*contentvec*'))
            if extracted_dirs:
                extracted_dir = extracted_dirs[0]  # Assume the first match
                if extracted_dir != contentvec_dir:
                    shutil.move(str(extracted_dir), str(contentvec_dir))

            # Remove zip file
            zip_path.unlink()

            print("ContentVec model setup complete.")
        else:
            print("ContentVec model already exists, skipping download.")

    except Exception as e:
        print(f"Error setting up ContentVec model: {str(e)}")

    # Handle Predictor model setup
    try:
        predictor_dir = infer_dir / "predictor"
        predictor_path = predictor_dir / "predictor.pt"

        # Ensure the predictor directory exists
        predictor_dir.mkdir(exist_ok=True)

        # Check if predictor.pt exists inside predictor directory
        if not any(p.suffix == ".pt" for p in predictor_dir.iterdir()):
            print("\nDownloading Predictor model...")
            wget.download(predictor_url, str(predictor_path))
            print("\nPredictor model setup complete.")
        else:
            print("\nPredictor model already exists, skipping download.")

    except Exception as e:
        print(f"Error setting up Predictor model: {str(e)}")

if __name__ == "__main__":
    download_and_setup_models()
