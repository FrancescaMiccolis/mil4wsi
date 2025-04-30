# Patch Extraction
 [CLAM](https://github.com/mahmoodlab/CLAM/blob/master/create_patches_fp.py) extracts .h5 coordinates of each patch.

Since this work  requires jpg, this [script](https://github.com/aimagelab/mil4wsi/blob/main/0-extract_patches/convert_h5_to_jpg.py) converts the coordinates into images.

Use this script inside the [CLAM](https://github.com/mahmoodlab/CLAM) repository



This script is designed to extract patches for a single resolution only. To accommodate multiple scales, you must rerun CLAM and this script for each desired scale. Once you have the patches for all scales, you can hierarchically organize them using the 1_sort_images script, ensuring you provide the correct paths for each scale.

Besides, use the custom_preset.csv as the config file in CLAM create_patches_fp.py (presets/*.csv). Hence, run 
```
git clone https://github.com/mahmoodlab/CLAM.git
cp mil4wsi/0-extract_patches/custom_preset.csv CLAM/presets/
```

However, the contour_fn in this file is relatively lenient to patches with too much blank. Change it according to your applications. Reference: https://github.com/mahmoodlab/CLAM/blob/master/docs/README.md

So, first of all run 
```
python CLAM/create_patches_fp.py --source /your/path/to/wsis --save_dir path/to/CLAM/patches/output/x20 --patch_size 256 --step_size 256 --preset custom_preset.csv --seg --patch --stitch --patch_level 0
```
(change the number of patch level according to your specific input images)
If the highest available resolution is the 20x, then the patch_level is 0. 
Given the structure of this framework, you should also extract the following two levels: at 10× and 5× resolution. So, you have to run:
```
python CLAM/create_patches_fp.py --source /your/path/to/wsis --save_dir path/to/CLAM/patches/output/x10 --patch_size 256 --step_size 256 --preset custom_preset.csv --seg --patch --stitch --patch_level 1
```
```
python CLAM/create_patches_fp.py --source /your/path/to/wsis --save_dir path/to/CLAM/patches/output/x5 --patch_size 256 --step_size 256 --preset custom_preset.csv --seg --patch --stitch --patch_level 2
```
Once finished those extractions, run 
```
python mil4wsi/0-extract_patches/convert_h5_to_jpg.py --output_dir path/to/CLAM/patches/output/x20/images --source_dir /your/path/to/wsis --slide_ext .mrxs --slurm_execution True
```
(change the slide_ext parameter according to your wsi extention)
Run this code also for 10x and 5x resolutions as
```
python mil4wsi/0-extract_patches/convert_h5_to_jpg.py --output_dir path/to/CLAM/patches/output/x10/images --source_dir /your/path/to/wsis --slide_ext .mrxs --slurm_execution True
```
```
python mil4wsi/0-extract_patches/convert_h5_to_jpg.py --output_dir path/to/CLAM/patches/output/x5/images --source_dir /your/path/to/wsis --slide_ext .mrxs --slurm_execution True
```

N.B. The `output_dir` parameter must be inside `save_dir` path of CLAM patches; `source_dir` is the same path as `source`.
The `slurm_execution` parameter must be True if you want to run the code using SLURM as executor, while must be False if you want to run the code directly from the command line. 