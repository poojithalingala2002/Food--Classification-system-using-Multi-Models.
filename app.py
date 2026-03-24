import os
import json
import uuid
import numpy as np
import redis
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# -----------------------------------
# 🔹 Redis Connection
# -----------------------------------
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# -----------------------------------
# 🔹 Load Models
# -----------------------------------
cnn_model = load_model(os.path.join("model results", "data.h5"))
vgg_model = load_model(os.path.join("model results", "vgg16_model.h5"))
resnet_model = load_model(os.path.join("model results", "res.h5"))

# -----------------------------------
# 🔹 Class Names (STANDARD FORMAT)
# -----------------------------------
class_names = [
    'apple_pie','baked_potato','burger','butter_naan','chai','chapati','cheesecake',
    'chicken_curry','chole_bhature','crispy_chicken','dal_makhani','dhokla','donut',
    'fried_rice','fries','hot_dog','ice_cream','idli','jalebi','kaathi_rolls',
    'kadai_paneer','kulfi','masala_dosa','momos','omelette','paani_puri',
    'pakode','pav_bhaji','pizza','samosa','sandwich','sushi','taco','taquito'
]

IMG_SIZE = 256


@app.route('/')
def home():
    return render_template("index.html", classes=class_names)


@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']
    model_name = request.form['model_name']
    actual_class = request.form['actual_class']

    upload_folder = os.path.join("static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    unique_name = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(upload_folder, unique_name)
    file.save(filepath)

    image_path = "/" + filepath

    img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)

    if model_name == "cnn":
        model = cnn_model
        metrics_key = "CNN_matrix"

    elif model_name == "vgg16":
        model = vgg_model
        metrics_key = "vgg16_matrix"

    else:
        model = resnet_model
        metrics_key = "resnet_matrix"

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]   # already correct format
    confidence = float(np.max(predictions)) * 100

    # -----------------------------------
    # 🔹 Fetch Food Properties (FIXED)
    # -----------------------------------
    redis_key = f"food:{predicted_class}"   # DIRECT MATCH

    food_json = r.get(redis_key)
    properties = json.loads(food_json) if food_json else {}

    # -----------------------------------
    # 🔹 Fetch Model Metrics
    # -----------------------------------
    metrics_json = r.get(metrics_key)

    if metrics_json:
        metrics = json.loads(metrics_json)

        test_accuracy = metrics.get("test_accuracy", 0)
        test_accuracy = round(float(test_accuracy) * 100, 2)

        class_metrics = metrics.get("classification_report", {})
        predicted_metrics = class_metrics.get(predicted_class, {})

        confusion_matrix = predicted_metrics.get("confusion_matrix", [])
        precision = round(predicted_metrics.get("precision", 0) * 100, 2)
        recall = round(predicted_metrics.get("recall", 0) * 100, 2)

    else:
        test_accuracy = 0
        confusion_matrix = []
        precision = 0
        recall = 0

    return render_template("index.html",
                           prediction=predicted_class,
                           actual_class=actual_class,
                           image_path=image_path,
                           food_properties=properties,
                           classes=class_names,
                           test_accuracy=test_accuracy,
                           confusion_matrix=confusion_matrix,
                           precision=precision,
                           recall=recall)


if __name__ == "__main__":
    app.run(debug=True)