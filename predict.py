import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# Load the improved model
model = tf.keras.models.load_model("pneumonia_cnn_improved.keras")

# Enter the path of the X-ray image
image_path = input("Enter X-ray image path: ")

# Load and preprocess image
image = load_img(image_path, target_size=(224, 224))
image_array = img_to_array(image)
image_array = image_array / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Make prediction
prediction = model.predict(image_array, verbose=0)[0][0]

# Display result
if prediction >= 0.5:
    result = "PNEUMONIA"
    confidence = prediction * 100
else:
    result = "NORMAL"
    confidence = (1 - prediction) * 100

print("\nPrediction:", result)
print("Confidence: {:.2f}%".format(confidence))