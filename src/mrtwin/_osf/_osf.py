"""
"""

__all__ = []

import os
# import requests

import numpy as np
import nibabel as nib

# from osfclient.models import Storage, File

# def fetch_all_files(self):
#     # Original URL for the storage
#     next_url = self._files_url
#     all_files = []

#     while next_url:
#         response = requests.get(next_url)
#         data = response.json()
        
#         # Collect files from this page
#         for file_data in data['data']:
#             file = File(file_data, self.session)
#             all_files.append(file)

#         # Update the URL to the next page of results
#         next_url = data['links'].get('next')

#     return all_files

# # Monkey patch the original Storage class
# Storage.files = fetch_all_files

from osfclient import OSF

# from .. import _prescription

# PREDATOR dataset project ID
DATASET_ID = "qkbca"

# Directory where data will be stored
base_dir = 'MR_datasets'

# Function to download and o rganize data per subject
def get_data(sub_id):
    
    # Initialize OSF client
    osf = OSF()

    # Replace 'your_osf_project_id' with your actual OSF project ID
    project = osf.project(DATASET_ID)

    # Replace 'your_storage_name' with the actual storage name (e.g., 'osfstorage')
    storage = project.storage("osfstorage")
    
    # Create a subject-specific directory
    sub_dir = os.path.join(base_dir, f"sub{sub_id:02d}")
    os.makedirs(sub_dir, exist_ok=True)
    
    # Data
    data = []
    
    for folder in storage.folders:
        # Check if file belongs to the subject (adjust according to your naming convention)
        if f"sub{sub_id:02d}" == folder.name:
            for file in folder.files:
                file_path = os.path.join(sub_dir, file.name)
                
                # Skip download if the file already exists
                if not os.path.exists(file_path):
                    with open(file_path, 'wb') as f:
                        file.write_to(f)
                
                # Load data
                data.append(nib.load(file_path).get_fdata().T)
    
    # stack maps
    data = np.stack(data, axis=0)
    
    # set prescription
    # data = _prescription.set_prescription(data, orig_res, data.shape[-3:], output_res)
    
    # normalize probability
    data = np.nan_to_num(data, posinf=0.0, neginf=0.0)
    
    return data.astype(np.float32)
            
    
