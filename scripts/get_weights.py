import os
import subprocess
import zipfile

FIGSHARE_URL_GEOM = "https://ndownloader.figshare.com/files/64213290"
FIGSHARE_URL_QM9 = "https://ndownloader.figshare.com/files/64211154"
ROOT = os.getcwd()
OUTPUT_FOLDER = os.path.join(ROOT, "trained_models")


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
                "--content-disposition",
                "--trust-server-names",
                "--user-agent=Mozilla/5.0",
                "--referer=https://figshare.com/",
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


def download_and_setup_weights(figshare_url_geom, figshare_url_qm9, output_folder):
    """
    Downloads trained model weights from Figshare, unzips them, and places them in the specified folder.

    Parameters:
        figshare_url_geom (str): The URL to the Figshare file containing the model weights for GEOM.
        figshare_url_qm9 (str): The URL to the Figshare file containing the model weights for QM9.
        output_folder (str): The folder where the unzipped files will be placed.
    """
    os.makedirs(output_folder, exist_ok=True)

    geom_zip_path = os.path.join(output_folder, "version_best_geom.zip")
    if not download_file_wget(figshare_url_geom, geom_zip_path):
        print("Failed to download GEOM model weights.")
        return False

    if not extract_zip_file(geom_zip_path, output_folder):
        print("Failed to extract GEOM model weights.")
        return False

    os.remove(geom_zip_path)

    qm9_zip_path = os.path.join(output_folder, "version_best_qm9.zip")
    if not download_file_wget(figshare_url_qm9, qm9_zip_path):
        print("Failed to download QM9 model weights.")
        return False

    if not extract_zip_file(qm9_zip_path, output_folder):
        print("Failed to extract QM9 model weights.")
        return False

    os.remove(qm9_zip_path)

    print("Setup complete. All zip files processed.")
    return True


if __name__ == "__main__":
    if download_and_setup_weights(FIGSHARE_URL_GEOM, FIGSHARE_URL_QM9, OUTPUT_FOLDER):
        print("Model weights have been successfully downloaded and set up.")
    else:
        print("Failed to set up model weights.")
