import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict import get_probabilities


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/feedback', methods=['GET', 'POST'])
def handle_feedback():
    feedback = request.form['feedback']
    filename = request.form['filename']
    correct_label = request.form['correctLabel']
    predicted_label = request.form['predictedLabel']

    print(f'Feedback: {feedback}')
    print(f'Filename: {filename}')
    print(f'Correct label: {correct_label}')
    print(f'Predicted label: {predicted_label}')

    if feedback == 'no':
        text = f'The label was: {correct_label} however the model predicted: {predicted_label}'
    else:
        text = f'The model predicted the correct label: {predicted_label}'
    return render_template('thanks.html', text=text)


@app.route('/uploader', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)

        sorted_results = get_probabilities(file_path)
        sorted_results = sorted(sorted_results, key=lambda x: x[1], reverse=True)

        return render_template('results.html', results=sorted_results, filename=filename)


if __name__ == '__main__':
    app.run(debug=True)
