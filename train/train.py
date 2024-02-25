import os
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from imagesloader.imagesloader import ImagesLoader
from sklearn import preprocessing
import timm
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
from sklearn.model_selection import StratifiedKFold


class CustomDataset(Dataset):
    def __init__(self, df, image_size, transform=None):
        self.df = df
        self.files = df['file'].values

        # Do label encoding
        label_encoder = preprocessing.LabelEncoder()
        df['encoded_label'] = label_encoder.fit_transform(df['label'])

        self.labels = df['encoded_label'].values

        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get file_path and label for index
        label = self.labels[idx]
        file_path = self.files[idx]

        # Read an image with OpenCV
        image = cv2.imread(file_path)

        # Convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        augmented = self.transform(image=image)
        image = augmented['image']

        # Normalize because ToTensorV2() doesn't normalize the image
        image = image / 255

        # Convert label to tensor
        label = torch.tensor(label)

        return image, label


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, train_dataloader, validate_dataloader, seed=98):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader

        set_seed(seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = self.model.to(self.device)

    def train_one_epoch(self):
        # Training mode
        self.model.train()

        # Init lists to store y and y_pred
        final_y = []
        final_y_pred = []
        final_loss = []

        # Iterate over data
        for step, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            X = batch[0].to(self.device)
            y = batch[1].to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward: Get model outputs
                y_pred = self.model(X)

                # Forward: Calculate loss
                loss = self.criterion(y_pred, y.long())

                # Covert y and y_pred to lists
                y = y.detach().cpu().numpy().tolist()
                y_pred = y_pred.detach().cpu().numpy().tolist()

                # Extend original list
                final_y.extend(y)
                final_y_pred.extend(y_pred)
                final_loss.append(loss.item())

                # Backward: Optimize
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

        # Calculate statistics
        loss = np.mean(final_loss)
        final_y_pred = np.argmax(final_y_pred, axis=1)
        metric = self.calculate_metric(final_y, final_y_pred)

        return metric, loss

    def validate_one_epoch(self):
        # Validation mode
        self.model.eval()

        final_y = []
        final_y_pred = []
        final_loss = []

        # Iterate over data
        for step, batch in tqdm(enumerate(self.validate_dataloader), total=len(self.validate_dataloader)):
            X = batch[0].to(self.device)
            y = batch[1].to(self.device)

            with torch.no_grad():
                # Forward: Get model outputs
                y_pred = self.model(X)

                # Forward: Calculate loss
                loss = self.criterion(y_pred, y.long())

                # Covert y and y_pred to lists
                y = y.detach().cpu().numpy().tolist()
                y_pred = y_pred.detach().cpu().numpy().tolist()

                # Extend original list
                final_y.extend(y)
                final_y_pred.extend(y_pred)
                final_loss.append(loss.item())

        # Calculate statistics
        loss = np.mean(final_loss)
        final_y_pred = np.argmax(final_y_pred, axis=1)
        metric = self.calculate_metric(final_y, final_y_pred)

        return metric, loss

    def calculate_metric(self, y, y_pred):
        metric = accuracy_score(y, y_pred)
        return metric

    def fit(self, epochs):
        acc_list = []
        loss_list = []
        val_acc_list = []
        val_loss_list = []

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            acc, loss = self.train_one_epoch()

            if self.validate_dataloader:
                val_acc, val_loss = self.validate_one_epoch()
                print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
                val_acc_list.append(val_acc)
                val_loss_list.append(val_loss)

            print(f'Loss: {loss:.4f} Acc: {acc:.4f}')
            acc_list.append(acc)
            loss_list.append(loss)

        return acc_list, loss_list, val_acc_list, val_loss_list

    def visualize_history(self, acc, loss, val_acc, val_loss):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].plot(range(len(loss)), loss, color='darkgrey', label='train')
        ax[0].plot(range(len(val_loss)), val_loss, color='cornflowerblue', label='valid')
        ax[0].set_title('Loss')

        ax[1].plot(range(len(acc)), acc, color='darkgrey', label='train')
        ax[1].plot(range(len(val_acc)), val_acc, color='cornflowerblue', label='valid')
        ax[1].set_title('Metric (Accuracy)')

        for i in range(2):
            ax[i].set_xlabel('Epochs')
            ax[i].legend(loc="upper right")
        plt.show()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


def set_seed(seed=98):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # In general seed PyTorch operations
    torch.manual_seed(seed)

    # If you are using CUDA on 1 GPU, seed it
    torch.cuda.manual_seed(seed)

    # If you are using CUDA on more than 1 GPU, seed them all
    torch.cuda.manual_seed_all(seed)

    # Certain operations in Cudnn are not deterministic, and this line will force them to behave!
    torch.backends.cudnn.deterministic = True

    # Disable the inbuilt cudnn auto-tuner that finds the best algorithm to use for your hardware.
    torch.backends.cudnn.benchmark = False


def check_model(df, image_size, batch_size, learning_rate, lr_min, epochs, backbone, n_folds, transform):
    # Create a new column for cross-validation folds
    df["kfold"] = -1

    # Initialize the kfold class
    skf = StratifiedKFold(n_splits=n_folds)

    # Fill the new column
    for fold, (train_, val_) in enumerate(skf.split(X=df, y=df.label)):
        df.loc[val_, "kfold"] = fold

    final_acc_list = []
    final_loss_list = []
    for fold in range(n_folds):
        print(f"Fold: {fold}")
        train_df = df[df.kfold != fold].reset_index(drop=True)
        validate_df = df[df.kfold == fold].reset_index(drop=True)

        train_dataset = CustomDataset(train_df, image_size, transform=transform)
        validate_dataset = CustomDataset(validate_df, image_size)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        n_classes = loader.get_number_of_classes()
        print(f'Total classes: {n_classes}')

        # Crate the model (use pretrained models made by Ross Wightman)
        model = timm.create_model(backbone,
                                  pretrained=True,
                                  num_classes=n_classes)

        # Prepare the model for training
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=np.ceil(len(train_dataloader.dataset) / batch_size) * epochs,
            eta_min=lr_min
        )

        trainer = Trainer(model, criterion, optimizer, scheduler, train_dataloader, validate_dataloader, 98)

        acc, loss, val_acc, val_loss = trainer.fit(epochs)

        trainer.visualize_history(acc, loss, val_acc, val_loss)

        final_acc = val_acc[-1]
        final_loss = val_loss[-1]

        final_acc_list.append(final_acc)
        final_loss_list.append(final_loss)

    print(f"Final mean val accuracy: {np.mean(final_acc_list)}")
    print(f"Final mean val loss: {np.mean(final_loss_list)}")


if __name__ == '__main__':
    # Load and prepare the data
    loader = ImagesLoader()
    loader.get_data('../data/google_api_images')
    df = loader.data

    image_size = 600
    batch_size = 16
    learning_rate = 0.0001
    lr_min = 1e-5
    epochs = 2
    backbone = 'resnet18'
    n_folds = 5

    # Transform for the training dataset
    transform = A.Compose([A.Rotate(p=0.6, limit=(-30, 30), crop_border=True),
                           A.Resize(image_size, image_size),
                           A.HorizontalFlip(p=0.6),
                           A.CoarseDropout(max_holes=2, max_height=64, max_width=64, p=0.3),
                           ToTensorV2()])

    check_model(df, image_size, batch_size, learning_rate, lr_min, epochs, backbone, n_folds, transform)
