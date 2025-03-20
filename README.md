# Skin Cancer Detection Using Hybrid Deep Learning

## **Overview**
This project focuses on **skin cancer detection** using deep learning models. It utilizes **hybrid deep learning** by combining multiple neural networks (**VGG16, ResNet, and CapsNet**) to improve classification accuracy. 

The goal is to classify skin lesions as **benign** or **malignant** using **ensemble learning with majority voting**.

## **Project Structure**
This repository contains the following Jupyter notebooks:

- **`Capsnet.ipynb`** - Implements **Capsule Network (CapsNet)**, which helps retain spatial hierarchies in image classification.
  
- **`Custom input.ipynb`** - Allows users to **test custom images** for classification using the trained ensemble model.

- **`Majority voting.ipynb`** - Implements **ensemble learning**, combining predictions from multiple models (VGG16, ResNet, and CapsNet) using **majority voting**.

- **`Resnet.ipynb`** - Implements the **ResNet** model, known for solving the vanishing gradient problem in deep networks.

- **`vgg16.ipynb`** - Implements the **VGG16** model, a deep CNN architecture, to classify skin lesions.

## **Methodology**
1. **Preprocessing:** Images are resized, normalized, and augmented.
2. **Model Training:** Three deep learning models (**VGG16, ResNet, and CapsNet**) are trained on the dataset.
3. **Ensemble Learning:** Predictions from all three models are combined using **majority voting**.
4. **Evaluation:** The model is evaluated using:
   - **Accuracy**
   - **Precision**
   - **Recall**
   - **F1-score**
   - **Sensitivity**
   - **Specificity**
5. **Custom Testing:** Users can input their own images for classification.

## **Results**
- **CapsNet achieved the highest accuracy (96.87%)**, followed by VGG16 (93.75%) and ResNet (87.25%).
- The **ensemble model** using majority voting achieved an accuracy of **90.62%**, demonstrating improved robustness over individual models.

## **Requirements**
To run this project, install the following dependencies:

```bash
pip install tensorflow keras numpy opencv-python matplotlib scikit-learn
