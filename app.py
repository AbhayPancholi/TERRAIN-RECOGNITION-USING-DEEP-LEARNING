from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import cv2

app = Flask(__name__, static_folder='static')  # Specify the static folder

# Load the trained model for terrain classification
model = load_model(r'D:\Project\Mini-Project-III\terrain.h5')

# Define class labels
class_labels = ['Grassy_Terrain', 'Marshy_Terrain', 'Other_Image', 'Rocky_Terrain', 'Sandy_Terrain']

# Define roughness and slipperiness scores based on predicted terrain
roughness_scores = {'Rocky_Terrain': 3, 'Sandy_Terrain': 2, 'Grassy_Terrain': 1, 'Marshy_Terrain': 2, 'Other_Image': 0}
slipperiness_scores = {'Rocky_Terrain': 2, 'Sandy_Terrain': 1, 'Grassy_Terrain': 1, 'Marshy_Terrain': 2, 'Other_Image': 0}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def detect_roughness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    mean = np.mean(gradient_magnitude)
    std = np.std(gradient_magnitude)
    roughness_index = std / mean
    return roughness_index

def slipperiness_percentage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    threshold_value = 5000  # Adjust this threshold according to the observed values

    if variance > threshold_value:
        percentage = 0  # Not slippery
    else:
        percentage = ((threshold_value - variance) / threshold_value) * 100

    return percentage

def roughness_level(roughness_index):
    high_threshold = 1.5
    low_threshold = 0.5

    if roughness_index > high_threshold:
        return 3  # High
    elif roughness_index < low_threshold:
        return 1  # Low
    else:
        return 2  # Medium

def slipperiness_level(slipperiness_percentage):
    high_threshold = 80
    low_threshold = 20

    if slipperiness_percentage > high_threshold:
        return 3  # High
    elif slipperiness_percentage < low_threshold:
        return 1  # Low
    else:
        return 2  # Medium

def predict_terrain(img_path):
    # Load image for roughness and slipperiness prediction
    image_for_roughness_slipperiness = cv2.imread(img_path)  # Corrected line

    # Make predictions for terrain classification
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)

    # Map predicted probabilities to class labels
    predicted_class_index = np.argmax(predictions)

    # Ensure the predicted class index is within the valid range
    if 0 <= predicted_class_index < len(class_labels):
        predicted_class = class_labels[predicted_class_index].lower()
    else:
        predicted_class = "N/A"

    # Predict roughness
    predicted_roughness = detect_roughness(image_for_roughness_slipperiness)
    roughness_lvl = roughness_level(predicted_roughness)


    # Predict slipperiness
    predicted_slipperiness = slipperiness_percentage(image_for_roughness_slipperiness)
    slipperiness_lvl = slipperiness_level(predicted_slipperiness)

    return predicted_class, roughness_lvl, slipperiness_lvl

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file from the request
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploads_dir = os.path.join('static', 'uploads')  # Create an 'uploads' directory within your project if not present
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)

            img_path = os.path.join(uploads_dir, uploaded_file.filename)
            uploaded_file.save(img_path)

            predicted_class, predicted_roughness, predicted_slipperiness = predict_terrain(img_path)

            # Pass only the filename to the template, assuming it's in the 'uploads' directory
            return render_template('index - Copy.html', 
                                   predicted_class=predicted_class, 
                                   predicted_roughness=predicted_roughness, 
                                   predicted_slipperiness=predicted_slipperiness,
                                   img_filename=uploaded_file.filename)
    return render_template('index - Copy.html')

if __name__ == '__main__':
    app.run(debug=True)
