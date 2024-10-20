# ISIC 2024 Skin Cancer Malignancy Detection

## Project Description

This project aims to develop an algorithm to detect skin cancer malignancy based on the images and metadata of each samples. 

## Performance Evaluation Parameter

### Partial Area Under the ROC Curve (pAUC)
- The model was evaluated with the partial area under the ROC curve (pAUC) parameter above 80% true positive rate (TPR) for binary classification of malignant examples.

## Data Outline
- The data used for training and validation set comes from the SLICE-3D dataset developed by the International Skin Imaging Collaboration.

#### Image data

- 401,059 15mm-by-15mm field-of-view images of skin lesion crops that centered on distinct lesions from the 3D total body photographs (TBP) in JPEG. The images ranges from ... pixels to 

#### Tabular data

- Metadata of the 401,059 samples that include patient demographics (age, sex), anatomical location, lesion characteristics (location, size, shape, and color features), lesion metrics (area, perimeter, color irregularity, contrast, asymmetry), 3D coordinates, etc. 

## Model Overview

### Resnet-34

https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html

#### Image Processing

https://www.sciencedirect.com/science/article/abs/pii/S0010482597000206?via%3Dihub 

#### Training 

### Light Gradient Boosting Machine (LGBM)

#### Hyperparameter Tuning 

#### Training

## Model Performance 

| **Model** | **Early Stopping Epoch Parameter** | **Epochs** | **Valid Set pAUC** |
|------------------|-------------------------------|------------|-------------------|
| **Resnet34 Fold 0**         | 15  | 13        | 0.120          |
| **Resnet34 Fold 1**         | 15  | 11        | 0.192          |
| **Resnet34 Fold 2**         | 15  | 43        | 0.194          |
| **Resnet34 Fold 3**         | 15  | 16        | 0.193          |
| **Resnet34 Fold 4**         | 15  | 6        | 0.195          |
| **LGBM**         | 50  | 561        | 0.149          |

*Note:
- **Early Stopping Epoch Parameter** : Number of training epochs to wait before stopping if the model's performance on the validation set does not improve
- **Valid Set pAUC** : pAUC results from assessing data from the validation dataset during model training

## Model Availability

Model is available at [Hugging Face](https://huggingface.co/albertw1706/resnet34_skin_cancer_malignancy_detection)

## Future Works
- Apply multiprocessing for each image preprocessing technique to save notebook run-time. 

## Contributors
- Albert Widjaja = Lead contributor and primary developer
- Andrew Ngadiman = Lead contributor and primary developer

## References and Sources

- International Skin Imaging Collaboration. SLICE-3D 2024 Challenge Dataset. International Skin Imaging Collaboration https://doi.org/10.34970/2024-slice-3d (2024). Creative Commons Attribution-Non Commercial 4.0 International License. The dataset was generated by the International Skin Imaging Collaboration (ISIC) and images are from the following sources: Hospital Clínic de Barcelona, Memorial Sloan Kettering Cancer Center, Hospital of Basel, FNQH Cairns, The University of Queensland, Melanoma Institute Australia, Monash University and Alfred Health, University of Athens Medical School, and Medical University of Vienna.
