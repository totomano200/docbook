import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

# Load the saved model from a file
model = load_model("model.h5")

app = Flask(__name__)
api = Api(app)
CORS(app)

class ImageUpload(Resource):
    def post(self):
        image = request.files['image']
        # Save the file to the uploads folder
        filename = secure_filename(image.filename)
        image.save(os.path.join('uploads', filename))

        # Load the saved model from a file
        model = load_model("model.h5")

        # Load and preprocess the image
        img = cv2.imread(os.path.join('uploads', filename))
        img = cv2.resize(img, (224, 224))
        img = img.astype("float") / 255.0
        img = np.expand_dims(img, axis=0)

        # Make the prediction
        prediction = model.predict(img)
        result = "yes" if prediction[0][1] > prediction[0][0] else "no"
        
        # Return the result as a JSON response
        return jsonify({"prediction": result})

api.add_resource(ImageUpload, '/scan_tumor')

if __name__ == '__main__':
    app.run(debug=True)
