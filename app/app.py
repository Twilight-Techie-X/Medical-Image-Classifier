import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)

models = {
    'brain_tumor': tf.keras.models.load_model('../models/brain_tumor_model.h5'),
    'bone_fracture': tf.keras.models.load_model('../models/bone_fracture_model.h5')
}

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form.get('model_type')
    file = request.files['file']
    image = Image.open(file.stream)
    processed_image = preprocess_image(image, target_size=(224, 224))
    
    if model_type not in models:
        return jsonify(error="Invalid model type"), 400
    
    model = models[model_type]
    prediction = model.predict(processed_image).tolist()
    return jsonify(prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)