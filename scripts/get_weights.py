import os
import subprocess
import zipfile

FIGSHARE_URL_TRAINED_MODELS = "https://ndownloader.figshare.com/files/68203198"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
ZIP_NAME = "trained_models.zip"
WEIGHTS_FOLDER_NAME = "trained_models"
WEIGHTS_FOLDER = os.path.join(ROOT, WEIGHTS_FOLDER_NAME)
ZIP_PATH = os.path.join(ROOT, ZIP_NAME)


def download_file_wget(figshare_url, output_path):
    """
    Downloads a file from Figshare using the system's wget command.

    Parameters:
        figshare_url (str): The URL to the Figshare file.
        output_path (str): The local path where the file will be saved.
    """
    print(f"Downloading file from {figshare_url} to {output_path}...")
    try:
        subprocess.run(
            [
                "wget",
                "-O",
                output_path,
                figshare_url,
            ],
            check=True,
        )
        print(f"File downloaded successfully to {output_path}.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to download file: {e}")
        return False


def extract_zip_file(zip_file_path, output_folder):
    """
    Extracts a zip file to the specified folder.

    Parameters:
        zip_file_path (str): The path to the zip file.
        output_folder (str): The folder where the contents will be extracted.
    """
    print(f"Unzipping file: {zip_file_path}...")
    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(output_folder)
        print(f"File successfully unzipped to {output_folder}.")
    except zipfile.BadZipFile:
        print("Error: The file is not a valid zip file.")
        return False
    return True


def download_and_setup_weights(figshare_url, repo_root, zip_path, weights_folder):
    """
    Downloads trained_models.zip from Figshare and extracts it into the
    repository root, yielding trained_models/.
    """
    if os.path.isdir(weights_folder):
        print(f"Weights folder already exists at {weights_folder}. Skipping download.")
        return True

    if os.path.isfile(zip_path) and os.path.getsize(zip_path) > 0:
        print(f"Using existing zip at {zip_path}; skipping download.")
    else:
        if not download_file_wget(figshare_url, zip_path):
            print("Failed to download NEAT-POCKET model weights.")
            return False

    if not os.path.isfile(zip_path) or os.path.getsize(zip_path) == 0:
        print(
            f"Download produced an empty file at {zip_path}. Check the Figshare URL."
        )
        return False

    if not extract_zip_file(zip_path, repo_root):
        print("Failed to extract NEAT-POCKET model weights.")
        return False

    if not os.path.isdir(weights_folder):
        print(
            f"Expected folder '{WEIGHTS_FOLDER_NAME}' was not found under "
            f"{repo_root} after extraction."
        )
        return False

    print(f"Setup complete. Weights are available at {weights_folder}.")
    return True


if __name__ == "__main__":
    if download_and_setup_weights(
        FIGSHARE_URL_TRAINED_MODELS, ROOT, ZIP_PATH, WEIGHTS_FOLDER
    ):
        print("Model weights have been successfully downloaded and set up.")
    else:
        print("Failed to set up model weights.")
