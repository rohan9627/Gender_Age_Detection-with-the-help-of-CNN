from model_struct import build_model
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

img_size = 96
dataset = "C:/Users/rohan/Desktop/AI/CNN/archive/UTKFace"


def load_data():
    images,genders,ages = [],[],[]

    for filename in os.listdir(dataset):
        try:
            age,gender,_ = filename.split('_')
            age = int(age)
            gender = int(gender)

            img_path = os.path.join(dataset,filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img,(img_size,img_size))
            img = img/255.0

            images.append(img)
            genders.append(gender)
            ages.append(age)

        except:
            continue

    return np.array(images) ,np.array(ages),np.array(genders)

X ,y_age, y_gender = load_data()


X_train, X_test, age_train, age_test, gender_train, gender_test = train_test_split(
    X, y_age, y_gender, test_size=0.2, random_state=42
)

model = build_model()

history = model.fit(
    X_train,
    {"gender": gender_train, "age": age_train},
    validation_split=0.2,
    epochs=125,
    batch_size=8,
    callbacks=[early_stop]
)

model.evaluate(X_test, {"gender": gender_test, "age": age_test})

# pred_gender , pred_age = model.predict(X_test)

# pred_gender = (pred_gender>=0.5).astype(int)

# for i in range(10):
#     plt.imshow(X_test[i])
#     plt.axis("off")

#     predicted_gender_label = 'male' if pred_gender[i] == 1 else 'female'
#     true_gender_label = 'male' if gender_test[i] == 1 else 'female'

#     predicted_age_value = int(pred_age[i])
#     true_age_value = int(age_test[i])

#     plt.title(f"Predicted: {predicted_gender_label}, {predicted_age_value}\nTrue: {true_gender_label}, {true_age_value}")
#     plt.show()
model.save("gender_age_model.keras")


        