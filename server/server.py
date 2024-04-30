from flask import Flask, request, jsonify
from PIL import Image
import cv2
import io
from classifier import full_system
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if a valid image file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        image = Image.open(io.BytesIO(file.read()))

        image.save('image.jpeg')


        label = classify_image('image.jpeg')
        print(label)

        return jsonify({'label': label})

def classify_image(image_path):
    try:
        im = cv2.imread(image_path)
        im = cv2.resize(im, (im.shape[1]//4, im.shape[0]//4))
        classification = full_system(im)
        return classification
    except Exception as e:
        print(e)
        return 'error'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')