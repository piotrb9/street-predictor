"""Make predictions using the model"""
import numpy as np
import torch
import albumentations as A
from PIL import Image
import timm
import joblib
from imagesloader.imagesloader import ImagesLoader
from albumentations.pytorch import ToTensorV2


class Predictor:
    def __init__(self, model, model_path):
        self.model = model
        self.model.load_state_dict(torch.load(model_path))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = self.model.to(self.device)

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


if __name__ == "__main__":
    loader = ImagesLoader()
    loader.get_data('../data/google_api_images')
    n_classes = loader.get_number_of_classes()
    image_size = 600

    backbone = 'resnet18'
    model = timm.create_model(backbone,
                              pretrained=True,
                              num_classes=n_classes)

    predictor = Predictor(model, "../data/models/model.pth")

    transform = A.Compose([A.Resize(image_size, image_size),
                           ToTensorV2()])

    probabilities = predictor.predict("../data/google_api_images/Grodzka+36_90.jpg", transform)
    print(probabilities)

    predicted_label_index = np.argmax(probabilities.cpu().numpy())
    print(f"Predicted class: {predicted_label_index}")

    label_encoder = joblib.load('../data/models/label_encoder.pkl')

    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
    print(f"Predicted class: {predicted_label}")

    # Print probability for every label
    for i, prob in enumerate(probabilities):
        print(f"{label_encoder.inverse_transform([i])[0]}: {prob:.4f}")
