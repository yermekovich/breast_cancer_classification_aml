# breast_cancer_classification_aml

## Overview
This project leverages deep learning techniques to classify breast ultrasound images into three categories: normal, benign, and malignant. It employs transfer learning with ResNet architectures and deploys the model as a web application using Streamlit.

## Features
- Data preprocessing with augmentation.
- Transfer learning using ResNet50 and ResNet18.
- Model evaluation with precision, recall, F1-score, and accuracy.
- Streamlit-based web application for real-time image classification.

## Dataset
The dataset is obtained from Kaggle: **Breast Ultrasound Images Dataset**.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd breast-cancer-classification
2.Install dependencies:
bash

pip install -r requirements.txt
3. Usage
Training
Run the training script to train the models:

```bash

python final.py

4. Deployment
Run the Streamlit application:

```
streamlit run app.py

Results
The best-performing model achieves:

Accuracy: 89.87%
F1-Score: 89.83%
Precision: 90.07%
Recall: 89.87%
License

This project is licensed under the AITU License.
