import numpy as np
import torch
import torch.nn as nn
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
    def __init__(self, model, criterion, optimizer, scheduler, train_dataloader, validate_dataloader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                loss = self.criterion(y_pred, y)

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
                loss = self.criterion(y_pred, y)

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




if __name__ == '__main__':
    # Load and prepare the data
    loader = ImagesLoader()
    loader.get_data('../data/google_api_images')
    df = loader.data

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=98)
    train_df, validate_df = train_test_split(train_df, test_size=0.2, random_state=98)

    train_dataset = CustomDataset(train_df, 600)
    validate_dataset = CustomDataset(validate_df, 600)

    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # for (image_batch, label_batch) in train_dataloader:
    #     print(image_batch.shape)
    #     print(label_batch.shape)
    #     break

    # Crate the model (use pretrained models made by Ross Wightman)

    n_classes = loader.get_number_of_classes()
    backbone = 'resnet18'

    learning_rate = 0.0001

    lr_min = 1e-5
    epochs = 5

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

