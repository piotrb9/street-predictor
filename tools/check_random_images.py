"""
Load the data and display random images from the dataset. Only for dataset exploration purposes.
"""
import random
import cv2
import matplotlib.pyplot as plt
from imagesloader.imagesloader import ImagesLoader
from config_variables import data_path

if __name__ == '__main__':
    loader = ImagesLoader()
    loader.get_data(f'../{data_path}')
    df = loader.data

    fig, ax = plt.subplots(2, 3, figsize=(10, 6))

    for i in range(2):
        for j in range(3):
            idx = random.randint(0, len(df)-1)
            label = df.label[idx]

            # Read an image with OpenCV
            image = cv2.imread(df.file[idx])

            # Convert the image to RGB color space.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            ax[i, j].imshow(image)
            ax[i, j].set_title(f"Label: {label}")
            ax[i, j].axis('off')

    plt.tight_layout()
    plt.show()
