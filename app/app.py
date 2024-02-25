import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict import Predictor, decode_predictions
import albumentations as A
import timm
from imagesloader.imagesloader import ImagesLoader
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)

        sorted_results = get_probabilities(file_path)
        sorted_results = sorted(sorted_results, key=lambda x: x[1], reverse=True)

        sorted_results = sorted_results[:7]

        return render_template('results.html', results=sorted_results, filename=filename)


def get_probabilities(file_path):
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

    probabilities = predictor.predict(file_path, transform)

    decoded_predictions = decode_predictions(probabilities)

    return decoded_predictions


if __name__ == '__main__':
    app.run(debug=True)
