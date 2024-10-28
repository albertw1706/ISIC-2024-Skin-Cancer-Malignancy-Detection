# ISIC 2024 Skin Cancer Malignancy Detection

## Project Description

Early detection is crucial for surviving skin cancer, but many people don't have access to skin specialists. In recent years, AI algorithms with dermoscopy have proven valuable in aiding clinicians with the diagnosis of melanoma, basal cell carcinoma, and squamous cell carcinoma. Screening tools could particularly help underserved communities catch skin cancer early, which is vital for patient survival. However, screening with dermatoscopes would be a challenge as specialized tools like such are generally available only at dermatology clinics. For AI to help people in general medical practices or at home, it needs to work well with lower-quality images. This challenge has motivated this project, which aims to develop an AI algorithm to detect skin cancer malignancy based on 3D total body photography images that resemble cell phone photos and metadata of each patient.

## Performance Evaluation Parameter

### Partial Area Under the ROC Curve (pAUC)
- The model was evaluated with the partial area under the ROC curve (pAUC) parameter above 80% true positive rate (TPR) for binary classification of malignant examples.

## Data Outline
- The data used for training and validation set comes from the SLICE-3D dataset developed by the International Skin Imaging Collaboration. The total samples were 401,059 with 393 samples were malignant and 400,666 samples were benign.

#### Image data

- 401,059 15mm-by-15mm field-of-view images of skin lesion crops that centered on distinct lesions from the 3D total body photographs (TBP) in JPEG. 

#### Tabular data

- Metadata of the 401,059 samples that include patient demographics (age, sex), anatomical location, lesion characteristics (location, size, shape, and color features), lesion metrics (area, perimeter, color irregularity, contrast, asymmetry), 3D coordinates, etc. 

## Model Overview

### Resnet-34

The model used for the image classification task was the Resnet-34 pre-trained on the ImageNet-1K dataset with input size 224x224 obtained from [Pytorch](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html)

#### Image Preprocessing

Most techniques applied in the image preprocessing steps were inspired by Hoshyar et al. (2014). The preprocessing steps, in the following order, are: 
- Image resize : To resize the image to 224x224 as the model input
- Dullrazor algorithm : Developed by Lee et al. (1997), is used to remove thick hairs that may obscure lesions in the image
- Median Filter :
- Wiener Filter :
- Contrast Enhancement : 

#### Training

The training process consists of 5-fold cross-validation to ensure the models were trained on the whole dataset. A weighted random sampler was used to apply weight in the sampling method due to a severely imbalanced dataset where 20,000 samples were used for training and 4,000 samples were used for validation per epoch. The weights were assigned based on the number of samples per class with each sample's weight being inversely proportional to its class frequency. The batch size was effectively set to 512 by using 8 mini-batches of size 64 and accumulating gradients across these mini-batches before updating the model weights. The maximum epochs per model was 100 with early stopping implemented on 15 epochs. The learning rate starts from 0.01 and is automatically reduced when the validation loss stops improving (plateaus) for 5 consecutive epochs. This reduction helps the model converge better by allowing it to make finer adjustments in later stages of training. Finally, the AdamW optimizer (Adam optimizer with weight decay applied directly to the weights themselves during each update step) was implemented to reduce overfitting and achieve smoother convergence.

-------------
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

- Hoshyar, A. N., Al-Jumaily, A., & Hoshyar, A. N. (2014). The beneficial techniques in preprocessing step of skin cancer Detection System Comparing. Procedia Computer Science, 42, 25–31. https://doi.org/10.1016/j.procs.2014.11.029
- International Skin Imaging Collaboration. SLICE-3D 2024 Challenge Dataset. International Skin Imaging Collaboration https://doi.org/10.34970/2024-slice-3d (2024). Creative Commons Attribution-Non Commercial 4.0 International License. The dataset was generated by the International Skin Imaging Collaboration (ISIC) and images are from the following sources: Hospital Clínic de Barcelona, Memorial Sloan Kettering Cancer Center, Hospital of Basel, FNQH Cairns, The University of Queensland, Melanoma Institute Australia, Monash University and Alfred Health, University of Athens Medical School, and Medical University of Vienna.
- Lee, T., Ng, V., Gallagher, R., Coldman, A., & McLean, D. (1997). Dullrazor®: A software approach to hair removal from images. Computers in Biology and Medicine, 27(6), 533–543. https://doi.org/10.1016/s0010-4825(97)00020-6

