import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict import get_probabilities

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


if __name__ == '__main__':
    app.run(debug=True)
