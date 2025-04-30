# Extract DINO feats 

- download [DINO](https://github.com/facebookresearch/dino) repository
- store repository path into Environment variable ```export DINO_REPO=PATH```  ```export MIL4WSI_PATH=PATH```
- collect a csv with three columns: slide_name (image), label (0/1), phase (train/test)
  example:


| image  | label | phase |
| ------------- | ------------- | ------------|
| name_1  | 0  | train |
| name_2  | 1  | test  |


Launch the feature extraction through submitit!
```
python mil4wsi/2-extract_features/features_extraction.py --extractedpatchespath HIERARCHICAL_PATH --savepath DESTINATION_PATH --pretrained_weights1 CHECKPOINTDINO20x --pretrained_weights2 CHECKPOINTDINO10x --pretrained_weights3 CHECKPOINTDINO5x --propertiescsv CSV_PATH
```

EXAMPLE:
```
python mil4wsi/2-extract_features/features_extraction.py --extractedpatchespath path/to/step1_output/ --savepath path/to/step2_output/ --pretrained_weights1 mil4wsi/2-extract_features/extract_tree/dino/x20/checkpoint.pth --pretrained_weights2 mil4wsi/2-extract_features/extract_tree/dino/x10/checkpoint.pth --pretrained_weights3 mil4wsi/2-extract_features/extract_tree/dino/x5/checkpoint.pth --propertiescsv labels.csv
```
