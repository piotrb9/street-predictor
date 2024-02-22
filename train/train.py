import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from imagesloader.imagesloader import ImagesLoader
from sklearn import preprocessing


class CustomDataset(Dataset):
    def __init__(self, df, image_size, transform=None, mode="val"):
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


if __name__ == '__main__':
    loader = ImagesLoader()
    loader.get_data('../data/google_api_images')
    df = loader.data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=98)

    example_dataset = CustomDataset(df, 600)

    batch_size = 128

    example_dataloader = DataLoader(example_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    )

    for (image_batch, label_batch) in example_dataloader:
        print(image_batch.shape)
        print(label_batch.shape)
        break