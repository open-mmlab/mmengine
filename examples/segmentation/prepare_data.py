import os
import shutil
import subprocess
import tarfile

try:
    # Install and upgrade opendatalab
    subprocess.check_call(['pip', 'install', 'opendatalab'],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
    print('Installed opendatalab.\n')
    subprocess.check_call(['pip', 'install', '-U', 'opendatalab'],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
    print('Upgraded opendatalab.\n')
except subprocess.CalledProcessError as e:
    print(f'Error: {e}')

# Check if the file exists
file_path = 'CamVid/raw/CamVid.tar.gz.00'
if not os.path.exists(file_path):
    print('Downloading dataset from https://opendatalab.com/CamVid.\n')
    print('You need register an account before downloading the dataset.\n')
    print('Please input your username and password to login to OpenDataLab.')
    # Login to OpenDataLab (Please manually enter your credentials when asked)
    subprocess.check_call(['odl', 'login'])
    # Download the dataset
    subprocess.check_call(['odl', 'get', 'CamVid'])

# Unzip dataset
with tarfile.open(file_path, 'r:gz') as tar:
    tar.extractall('data')
    print('Extracted dataset to /data.\n')

# Delete the CamVid folder
shutil.rmtree('CamVid')
