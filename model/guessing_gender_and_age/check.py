import cv2 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model("guessing_gender_and_age/gender_age_model.keras")

img_path = "C:/Users/rohan/Desktop/AI/CNN/archive/UTKFace/26_1_0_20170103180946896.jpg.chip.jpg"

def preprocesing(image_path, img_size=96):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize properly
    return np.expand_dims(img, axis=0), img

img_input, img_display = preprocesing(img_path)

pred_gender, pred_age = model.predict(img_input)
gender = "Male" if pred_gender[0][0] >= 0.5 else "Female"
age = int(pred_age[0][0])

# Convert image for display
img_display_uint8 = (img_display * 255).astype(np.uint8)

plt.imshow(cv2.cvtColor(img_display_uint8, cv2.COLOR_BGR2RGB))
plt.title(f"Predicted Gender: {gender}\nPredicted Age: {age}")
plt.axis("off")
plt.show()
