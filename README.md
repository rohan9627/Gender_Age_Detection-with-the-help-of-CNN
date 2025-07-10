# Gender_Age_Detection-with-the-help-of-CNN
This is an ongoing deep learning project that aims to detect a person's gender and estimate their age using facial images. The model is built using a Convolutional Neural Network (CNN) architecture and trained on the UTKFace dataset.

The goal of this project is to build a deep learning model that can:
Predict the gender (Male/Female) of a person from a facial image.
Estimate the age of the person.

This project is useful in applications like:
Demographic analysis
Targeted advertising
Access control and personalization

We are using the UTKFace dataset, which contains over 20,000 face images labeled with:
Age (0–116 years)
Gender (0 = Male, 1 = Female)
Ethnicity (not used currently)
Each image file is named in the format:
[age]_[gender]_[race]_[date&time].jpg

The model consists of:
Shared convolutional layers for feature extraction
Two separate dense output heads:
One for gender classification (binary classification)
One for age regression (continuous output)
Libraries used:
TensorFlow / Keras
NumPy
OpenCV (for face detection in testing)
Matplotlib (for visualization)
