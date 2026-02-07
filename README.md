# Multi-class-Dog-Breed-Classification-using-Neural-Networks-Deep-Learning

An end-to-end Deep Learning project using **TensorFlow 2.x** and **TensorFlow Hub** to identify 120+ different breeds of dogs. This project demonstrates the implementation of Transfer Learning on a large-scale image dataset sourced from Kaggle.

##  Project Overview
The goal of this project is to build a "Dog Vision" model that can take an image of a dog and predict its breed. This is a **fine-grained image classification** problem, meaning the model must distinguish between very similar-looking classes (e.g., different types of Terriers). The code has necessary comments for better understanding so that it can be used for educational purposes.



##  Technical Stack
* **Framework:** TensorFlow 2.19 (using Keras 3)
* **Feature Extractor:** MobileNetV2 (via TensorFlow Hub)
* **Data Handling:** NumPy, Pandas, and `tf.data` for high-performance pipelines
* **Visualization:** Matplotlib for prediction analysis

##  Methodology
1. **Data Preprocessing:** Images are resized to 224x224 and normalized. Labels are one-hot encoded.
2. **Batching:** Implemented `tf.data` to create efficient batches ($32$ images per batch) to optimize GPU memory.
3. **Model Architecture:** * **Input Layer:** $224 \times 224 \times 3$ (RGB Images)
    * **Hidden Layer:** Pre-trained MobileNetV2 feature vector (ImageNet weights).
    * **Output Layer:** Dense layer with Softmax activation for 120 classes.
4. **Training:** Optimized using the Adam optimizer and Categorical Crossentropy loss.



##  Results & Predictions
The model is capable of:
* Predicting the breed of a dog from a test dataset.
* Providing a **Confidence Score** (Probability) for each prediction.
* **Top-10 Analysis:** Sorting the highest probability indices to show the most likely matches.
* **Custom Testing:** Identifying breeds from external images uploaded by the user.

##  Installation & Setup
To run this project locally or in Colab, install the required dependencies:

```bash
pip install tensorflow tensorflow-hub pandas numpy matplotlib
