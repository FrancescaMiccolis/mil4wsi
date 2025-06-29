# Hierarchical Organization

This script helps to reorganize patches of different resolutions.

Given the initial folder path (e.g. x5, x10, x20) it will reorganize the patches into


# Example:
```
Slide_id
tumor_001_tumor/
    0_x_21056_y_119296:
        0__x_21056_y_119296/
            0__x_21568_y_119808.jpg
        0__x_21056_y_119296.jpg
        0__x_21056_y_120320/
            0__x_21056_y_120320.jpg
            0__x_21056_y_120832.jpg
            0__x_21568_y_120320.jpg
            0__x_21568_y_120832.jpg
        0__x_21056_y_120320.jpg
        0__x_22080_y_119296/
            0__x_22080_y_119296.jpg
            0__x_22080_y_119808.jpg
            0__x_22592_y_119296.jpg
            0__x_22592_y_119808.jpg
        0__x_22080_y_119296.jpg
        0__x_22080_y_120320/
            0__x_22080_y_120320.jpg
            0__x_22080_y_120832.jpg
            0__x_22592_y_120320.jpg
            0__x_22592_y_120832.jpg
        0__x_22080_y_120320.jpg
    0_x_21056_y_119296.jpg
    /0_x_21056_y_121344
        0__x_21056_y_121344/
            0__x_21056_y_121344.jpg
            0__x_21056_y_121856.jpg
            0__x_21568_y_121344.jpg
            0__x_21568_y_121856.jpg
        0__x_21056_y_121344.jpg
        0__x_22080_y_121344/
            0__x_22080_y_121344.jpg
            0__x_22080_y_121856.jpg
            0__x_22592_y_121344.jpg
            0__x_22592_y_121856.jpg
        0__x_22080_y_121344.jpg
    0_x_21056_y_121344.jpg
```


# Launch

```
python mil4wsi/1-sort_images/sort_hierarchy.py --sourcex5  path/to/CLAM/patches/output/x5/images/ --sourcex10 path/to/CLAM/patches/output/x10/images/ --sourcex20 path/to/CLAM/patches/output/x20/images/ --dest path/to/step1_output/ --slurm_execution False
```
The `sourcex5` (as for `sourcex10` and `sourcex20`) is the path configured as `output_dir` of the step 0-extract_patches for the specific resolution.  
The `slurm_execution` parameter must be True if you want to run the code using SLURM as executor, while must be False if you want to run the code directly from the command line. 