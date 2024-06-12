import os
import requests
from zipfile import ZipFile

# Change directory to JAAD dataset location
jaad_dir = './JAAD_dataset'
os.chdir(jaad_dir)

# Download JAAD clips
url = 'http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip'
file_name = url.split('/')[-1]

print(f"Downloading {file_name}...")
r = requests.get(url)
with open(file_name, 'wb') as f:
    f.write(r.content)
print("Download complete!")

# Extract JAAD clips
print("Extracting JAAD clips...")
with ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall(jaad_dir)
print("Extraction complete!")
