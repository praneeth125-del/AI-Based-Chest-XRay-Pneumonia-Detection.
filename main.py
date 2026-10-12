import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    "chest_xray/test",
    image_size=(224, 224),
    batch_size=32
)

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

# Load saved model
model = tf.keras.models.load_model("pneumonia_cnn.keras")
print("Saved model loaded")

# Evaluate
test_loss, test_accuracy = model.evaluate(test_data)
print("Test Accuracy:", test_accuracy)

# Predictions
y_true = []
y_pred = []

for images, labels in test_data:
    predictions = model.predict(images, verbose=0)
    predictions = (predictions > 0.5).astype(int).flatten()

    y_true.extend(labels.numpy())
    y_pred.extend(predictions)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["NORMAL", "PNEUMONIA"]
))

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NORMAL", "PNEUMONIA"]
)

disp.plot()
plt.title("Chest X-Ray Pneumonia Detection")
plt.show()