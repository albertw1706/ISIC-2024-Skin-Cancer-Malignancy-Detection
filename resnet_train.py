from PIL import Image
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from torch.nn import Linear, BCEWithLogitsLoss
from torch.optim import AdamW
import cv2
import math
from scipy.signal import wiener
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from torch.optim import AdamW
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        if (h * r) < 224:
            dim = (width, math.ceil(h * r))
        else:
            dim = (width, math.floor(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def dullrazor (image):
    # Convert the original image to grayscale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY )

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1,(17,17))

    # Perform the blackHat filtering on the grayscale image to find the 
    # hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting 
    # algorithm
    ret,thresh2 = cv2.threshold(blackhat,5,255,cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    img_fnl = cv2.inpaint(image,thresh2,1,cv2.INPAINT_TELEA)
    
    return img_fnl 

def apply_wiener_filter(img):
    # Convert to float32 for filtering
    img_float = np.float32(img)
    # Apply Wiener filter (assuming a 3x3 filter size)
    filtered_img = (wiener(img_float, (5, 5)))
    
    return np.uint8(filtered_img) 

def apply_wiener (img):
    # Split into R, G, B channels
    b_channel, g_channel, r_channel = cv2.split(img)

    # Apply Wiener filter to each channel
    b_filtered = apply_wiener_filter(b_channel)
    g_filtered = apply_wiener_filter(g_channel)
    r_filtered = apply_wiener_filter(r_channel)

    # Merge channels back
    filtered_color_img = cv2.merge([b_filtered, g_filtered, r_filtered])
    
    return filtered_color_img

def apply_enhanced(img):
    # converting to LAB color space
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    iimg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(iimg, cv2.COLOR_LAB2RGB)
    return enhanced_img

class MyTransform(torch.nn.Module):
    def forward(self, img):
        img = image_resize(img, width = 224, height = 224)
        img = dullrazor(img)
        img = cv2.medianBlur(img,5)
        img = apply_wiener(img)
        img = apply_enhanced(img)
        return img
    
class ISICDataset(Dataset):
    def __init__(self, df, test_file, transform_aug=None, transform_func=None, mode="train"):
        self.df = df
        self.test_file = test_file
        self.transform_aug = transform_aug
        self.transform_func = transform_func
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        t = str(self.df.iloc[idx]['isic_id'])
        label = self.df.iloc[idx]['target']
        with Image.open(f'./enhanced/image/{t}.jpg') as img:
            img = self.transform_aug(img)
            return t, img, torch.tensor(label, dtype=torch.int64)
    
def score(solution: np.ndarray, submission: np.ndarray, min_tpr: float=0.80) -> float:
    v_gt = abs(solution-1)
    v_pred = np.array([1.0 - x for x in submission])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return partial_auc

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler=None, patience=5, accumulation_steps=16):
        self.model = model.to(DEVICE)
        self.criterion = criterion.to(DEVICE)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.accumulation_steps = accumulation_steps
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.metrics = {
            x: MeanMetric().to(DEVICE) 
            for x in ("loss", "pAUC", "val_loss", "val_pAUC")
        }
        self.history = pd.DataFrame(columns=list(self.metrics), index=pd.Index([], name="epoch"))
        
    def fit(self, train_loader, val_loader, epochs, verbose):
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch:04d}/{epochs:04d}")
            self.train_one_epoch(train_loader, verbose, self.accumulation_steps)
            self.validate_one_epoch(val_loader, verbose)
            
            # Learning Rate Scheduler step (if using ReduceLROnPlateau)
            if self.scheduler:
                self.scheduler.step(self.metrics['val_loss'].compute())
            print(self.metrics['val_loss'].compute().item())

            # Early Stopping check
            if self.metrics['val_loss'].compute().item() < self.best_val_loss:
                self.best_val_loss = self.metrics['val_loss'].compute().item()
                self.best_model_wts = self.model.state_dict()
                self.early_stopping_counter = 0
                print(f"New best model found with val_loss: {self.best_val_loss:.4f}")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.patience:
                    print("Early stopping triggered.")
                    break
            self.update_history(epoch)
        # Return best weights
        return self.best_model_wts
    
    def train_one_epoch(self, loader, verbose, accumulation_steps):
        self.model.train()
        accumulation_counter = 0
        
        with tqdm(total=len(loader), disable=not verbose) as progress_bar:
            for batch in loader:
                images, labels = batch[1].to(DEVICE), batch[2].to(DEVICE)
                labels = torch.unsqueeze(labels, 1)

                # Forward pass
                predict = self.model(images)
                loss = self.criterion(predict, labels.float())

                # Normalize loss to account for accumulation
                loss = loss / accumulation_steps
                loss.backward()

                # Update weights after accumulation_steps
                if (accumulation_counter + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else : 
                    accumulation_counter += 1

                # Update metrics
                predictions = torch.sigmoid(predict).detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                pAUC = self.calculate_pAUC(labels_np, predictions)

                self.metrics["loss"].update(loss.item() * accumulation_steps)  # Multiply to get the actual loss value
                self.metrics["pAUC"].update(pAUC)

                progress_bar.set_description(
                    f"loss {self.metrics['loss'].compute():.3f} "
                    f"pAUC {self.metrics['pAUC'].compute():.3f}"
                )
                progress_bar.update(1)

            if accumulation_counter % accumulation_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
    
    
    def validate_one_epoch(self, loader, verbose):
        self.model.eval()
        
        with tqdm(total=len(loader), disable=not verbose) as progress_bar, torch.no_grad():
            for batch in loader:
                images, labels = batch[1].to(DEVICE), batch[2].to(DEVICE)
                predict = self.model(images)
                labels = torch.unsqueeze(labels, 1)
                loss = self.criterion(predict, labels.float())
                
                predictions = torch.sigmoid(predict).detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()

                pAUC = self.calculate_pAUC(labels_np, predictions)  # Calculate pAUC
                
                self.metrics["val_loss"].update(loss.item())
                self.metrics["val_pAUC"].update(pAUC)
                
                progress_bar.set_description(
                    f"val_loss {self.metrics['val_loss'].compute():.3f} "
                    f"val_pAUC {self.metrics['val_pAUC'].compute():.3f}"
                )
                progress_bar.update(1)
    
    def calculate_pAUC(self, labels_np, predictions) -> float:
        """
        Calculate the pAUC (partial Area Under the Curve) for the predictions.
        This method wraps the `score` function.
        """
        return score(labels_np, predictions, min_tpr=0.8)
                
    def update_history(self, epoch: int) -> None:
        metrics = {}

        for metric_name, metric in self.metrics.items():
            value = metric.compute().item()
            print(f"{metric_name}: {value:.3f}")
            metrics[metric_name] = value
            metric.reset()

        self.history.loc[epoch] = metrics


if __name__ == '__main__':
    train_image_directory = 'train-image.hdf5'
    train_meta_directory = 'train-metadata.csv'

    # Image transformation
    transforms_aug = v2.Compose([v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225]),
                            v2.RandomVerticalFlip(p=0.5),
                            v2.RandomHorizontalFlip(p=0.5), 
                            v2.RandomRotation((1, 359))])

    transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])])

    # Reads the CSV and split the files
    df = pd.read_csv(train_meta_directory)
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True, random_state = 42)

    train_ds = ISICDataset(train_df, test_file=train_image_directory, transform_aug=transforms_aug, transform_func=transforms)
    val_ds = ISICDataset(val_df, test_file=train_image_directory, transform_aug=transforms_aug, transform_func=transforms, mode="valid")

    # Load the pretrained ResNet-50 model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Modify the final fully connected layer for binary classification, specify optimizer and loss function 
    model.fc = Linear(model.fc.in_features, out_features=1)
    optimizer = AdamW(model.parameters(), lr= 0.01)
    criterion = BCEWithLogitsLoss()
    # Add learning rate scheduler and specify 5-split CV
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

    #FROM HERE THIS IS KFOLD
    # Split the samples into folds in dataframe
    df['kfold'] = 999
    y = df['target'].values

    for f,(tidx,vidx) in enumerate(skf.split(X=df,y=y)):
        df.loc[vidx,'kfold'] = f
    
    df.to_csv("train_folds.csv", index=False)

    train_data = pd.read_csv('./train_folds.csv')

    # Train each samples based on the fold provided in the CSV
    def train_on_fold(fold,  model, train_data):
        train_df_fold = train_data.query("kfold!=@fold").reset_index(drop=True)
        val_df_fold = train_data.query("kfold==@fold").reset_index(drop=True)

        trainset = ISICDataset(train_df_fold, test_file=train_image_directory, transform_aug=transforms_aug, transform_func=transforms)
        
        valset = ISICDataset(val_df_fold, test_file=train_image_directory, transform_aug=transforms_aug, transform_func=transforms, mode="valid")
        
        y_train = train_df_fold['target'].values
        y_val = val_df_fold['target'].values

        train_class_sample_count = np.array(
            [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

        train_weight = 1. / train_class_sample_count
        train_samples_weight = np.array([train_weight[t] for t in y_train])
        train_samples_weight = torch.from_numpy(train_samples_weight)

        val_class_sample_count = np.array(
            [len(np.where(y_val == t)[0]) for t in np.unique(y_val)])

        val_weight = 1. / val_class_sample_count
        val_samples_weight = np.array([val_weight[t] for t in y_val])
        val_samples_weight = torch.from_numpy(val_samples_weight)

        train_sampler = WeightedRandomSampler(train_samples_weight.type('torch.DoubleTensor'), num_samples=20000)
        val_sampler = WeightedRandomSampler(val_samples_weight.type('torch.DoubleTensor'), num_samples=4000)

        train_loader_fold = DataLoader(
            trainset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            sampler=train_sampler,
            pin_memory=True,
        )

        val_loader_fold = DataLoader(
            valset,
            batch_size = 64,
            shuffle=False,
            num_workers=4,
            sampler = val_sampler,
            pin_memory=True,
        )

        trainer = Trainer(model, criterion, optimizer, scheduler=scheduler, patience=15, accumulation_steps=8)
        best_weight = trainer.fit(train_loader_fold, val_loader_fold, epochs=100, verbose=True)
        torch.save(best_weight, f'model_fold-{fold}_15_epoch.pth')

    train_on_fold(0, model, train_data)
    torch.cuda.empty_cache()
    train_on_fold(1, model, train_data)
    torch.cuda.empty_cache()
    train_on_fold(2, model, train_data)
    torch.cuda.empty_cache()
    train_on_fold(3, model, train_data)
    torch.cuda.empty_cache()
    train_on_fold(4, model, train_data)