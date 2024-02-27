"""Make predictions using the model"""
import numpy as np
import torch
import albumentations as A
from PIL import Image
import timm
import joblib
from imagesloader.imagesloader import ImagesLoader
from albumentations.pytorch import ToTensorV2
from config_variables import image_size, model_path, label_encoder_path, data_path, backbone


class Predictor:
    def __init__(self, model, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def predict(self, image_path, transform):
        self.model.eval()  # Set the model to evaluation mode

        # Open the image file and apply transformations
        with Image.open(image_path) as img:
            input_tensor = transform(image=np.array(img))
            input_batch = input_tensor['image'].unsqueeze(0)  # Create a mini-batch as expected by the model

        # Move the input and model to GPU for speed if available
        input_batch = input_batch.to(self.device)

        # Convert the input tensor to FloatTensor
        input_batch = input_batch.float()

        with torch.no_grad():
            # Make a prediction
            output = self.model(input_batch)

        # The output has unnormalized scores. To get probabilities run a softmax on it
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        return probabilities


def decode_predictions(probabilities, label_encoder_path='../data/models/label_encoder.pkl'):
    label_encoder = joblib.load(label_encoder_path)

    data = []
    # Print probability for every label
    for i, prob in enumerate(probabilities):
        # print(f"{label_encoder.inverse_transform([i])[0]}: {prob:.4f}")
        data.append((label_encoder.inverse_transform([i])[0], prob.item()))

    return data


def get_probabilities(file_path):
    loader = ImagesLoader()
    loader.get_data(f'../{data_path}')
    n_classes = loader.get_number_of_classes()

    model = timm.create_model(backbone,
                              pretrained=True,
                              num_classes=n_classes)

    predictor = Predictor(model, f"../{model_path}")

    transform = A.Compose([A.Resize(image_size, image_size),
                           ToTensorV2()])

    probabilities = predictor.predict(file_path, transform)

    decoded_predictions = decode_predictions(probabilities, f"../{label_encoder_path}")

    return decoded_predictions
