import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from classes import CLASS_NAMES

# Load trained model
model = tf.keras.models.load_model("model/traffic_sign_cnn.keras")

# Image path
image_path = "test_image.jpg"

# Load image
img = Image.open(image_path).convert("RGB")

# Resize image
img_resized = img.resize((32, 32))

# Convert to numpy
img_array = np.array(img_resized)

# Normalize
img_array = img_array.astype("float32") / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array, verbose=0)

# Get predicted class
predicted_class = np.argmax(prediction)

# Get confidence
confidence = np.max(prediction) * 100

# Display result
print("----------------------------------")
print("Traffic Sign Recognition")
print("----------------------------------")
print("Class ID:", predicted_class)
print("Traffic Sign:", CLASS_NAMES[predicted_class])
print("Confidence: {:.2f}%".format(confidence))
print("----------------------------------")

# Show image
plt.imshow(img)
plt.axis("off")
plt.show()