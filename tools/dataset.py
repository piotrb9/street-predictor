import joblib
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn import preprocessing


class CustomDataset(Dataset):
    def __init__(self, df, image_size, transform=None):
        self.df = df
        self.files = df['file'].values

        # Do label encoding
        self.label_encoder = preprocessing.LabelEncoder()
        df['encoded_label'] = self.label_encoder.fit_transform(df['label'])

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

    def save_label_encoder(self, path):
        # Save label encoder
        joblib.dump(self.label_encoder, path)

