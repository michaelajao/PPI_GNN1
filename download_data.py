import os
import gdown
import zipfile
import shutil

def download_and_extract():
    # Create directories if they don't exist
    processed_dir = '../PPI_GNN1/human_features/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    # Google Drive file ID
    file_id = '1mpMB2Gu6zH6W8fZv-vGwTj_mmeitIV2-'
    output = 'processed_data.zip'
    
    try:
        # Download the file
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
        
        # Extract the zip file directly to the processed directory
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(processed_dir)
            
        # Clean up the zip file
        os.remove(output)
        print("Download and extraction completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
if __name__ == "__main__":
    download_and_extract()
