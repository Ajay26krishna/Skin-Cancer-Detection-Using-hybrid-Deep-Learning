
Skin Cancer Detection Using Hybrid Deep Learning

Overview
This project focuses on skin cancer detection using deep learning models. It leverages hybrid deep learning by combining multiple neural networks (VGG16, ResNet, and CapsNet) to improve classification accuracy. The goal is to distinguish between benign and malignant skin lesions using an ensemble model with majority voting.

The dataset used for training and evaluation consists of dermoscopic images of skin lesions, sourced from the International Skin Imaging Collaboration (ISIC) dataset.

Project Structure
The repository consists of the following key Jupyter notebooks:

Capsnet.ipynb - Implements the Capsule Network (CapsNet) model for skin cancer detection. CapsNet is known for its ability to retain spatial hierarchies in image classification.

Custom input.ipynb - Allows users to test custom images for classification using the trained ensemble model. Users can upload an image, preprocess it, and classify it as either benign or malignant.

Majority voting.ipynb - Implements ensemble learning by combining predictions from multiple models (VGG16, ResNet, and CapsNet) using a majority voting approach to improve classification performance.

Resnet.ipynb - Implements the ResNet (Residual Network) model for skin cancer classification. ResNet is effective in deep learning tasks due to its residual connections, which help in avoiding the vanishing gradient problem.

vgg16.ipynb - Implements VGG16, a well-known convolutional neural network (CNN) model, for classifying skin lesions into benign or malignant categories.

Methodology
Preprocessing: Images are resized, normalized, and augmented to improve generalization.
Model Training: Three deep learning models (VGG16, ResNet, and CapsNet) are trained on the dataset.
Ensemble Learning: The predictions of all three models are combined using a majority voting approach.
Evaluation: The final model is evaluated using metrics such as accuracy, precision, recall, F1-score, sensitivity, and specificity.
Custom Testing: Users can input their own skin lesion images for classification.

Results
The CapsNet model achieved the highest accuracy (96.87%), followed by VGG16 (93.75%) and ResNet (87.25%).
The ensemble model using majority voting achieved an accuracy of 90.62%, demonstrating improved robustness over individual models.

Requirements
Python 3.x
TensorFlow & Keras
OpenCV
NumPy
Matplotlib
Scikit-learn

How to Use

Clone the repository:
git clone https://github.com/your-username/skin-cancer-detection.git
cd skin-cancer-detection

Install dependencies:
pip install -r requirements.txt

Run the notebooks in Jupyter or Google Colab.

Acknowledgment
This project was developed as part of an academic dissertation at CVR College of Engineering.
