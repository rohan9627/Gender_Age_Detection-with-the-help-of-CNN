import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model("guessing_gender_and_age/gender_age_model.keras")

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Preprocess function
def preprocess_frame(face, img_size=96):
    face = cv2.resize(face, (img_size, img_size))
    face = face / 255.0
    return np.expand_dims(face, axis=0)


# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        input_face = preprocess_frame(face)

        pred_gender, pred_age = model.predict(input_face, verbose=0)

        gender = "Male" if pred_gender[0][0] >= 0.5 else "Female"
        age = int(pred_age[0][0])

        # Draw bounding box and label
        label = f"{gender}, Age: {age}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # If no face detected
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Gender and Age Prediction", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()
