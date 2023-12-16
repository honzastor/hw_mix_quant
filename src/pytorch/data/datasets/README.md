## Datasets Folder
This folder is intended for storing datasets used in this project. As datasets can be large and are not suitable for storage in a GitHub repository, this is a placeholder file.

### Getting Started
To use the existing data loader scripts (else modify or create new ones), download the desired datasets into this folder and inside them, organize the data into two subfolders: `train` and `val`.

#### Train Folder
The `train` folder should contain subfolders, each named after a class.
Each subfolder should contain images corresponding to that class.

#### Validation Folder
The `val` folder is structured similarly to the train folder.
It should contain a set of images used for validating the model's performance.

Example ImageNet dataset structure:
```
imagenet/
│
├── train/
│   ├── n02051845/  # Class folder (e.g., "pelican")
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   ├── n01443537/  # Another class folder (e.g., "goldfish")
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   └── ...         # More class folders
│
└── val/
    ├── n02051845/  # Class folder (e.g., "pelican")
    │   ├── image1.png
    │   └── ...
    ├── n01443537/  # Another class folder (e.g., "goldfish")
    │   ├── image1.png
    │   └── ...
    └── ...         # More class folders
```