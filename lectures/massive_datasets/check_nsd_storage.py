# file: check_nsd_storage.py
import os, nibabel as nib, numpy as np

nsd_root = '/your/path/NSD'                  # after downloading NSD
example_func = f'{nsd_root}/nsdata/fmri/pp001/session01/bold001.nii.gz'
img          = nib.load(example_func)
vox          = img.header.get_data_shape()
bytes_per_scan = np.prod(vox) * 4            # 4 bytes per 32-bit float
print(f'One 7 T run: {bytes_per_scan/1e9:.2f} GB')