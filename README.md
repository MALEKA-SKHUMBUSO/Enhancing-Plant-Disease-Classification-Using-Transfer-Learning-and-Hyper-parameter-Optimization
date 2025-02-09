# Enhancing-Plant-Disease-Classification-Using-Transfer-Learning-and-Hyper-parameter-Optimization
Table of Contents
Project Overview
Directory Structure
Data Directory Organization
Model Performance
Usage Instructions
Conclusion
Project Overview
This project focuses on enhancing plant disease classification using deep learning techniques. We utilized several pre-trained models (MobileNet, InceptionV3, and VGG16) for image classification on a plant disease dataset. The aim is to identify plant diseases across different classes (such as "Potato Early Blight," "Tomato Leaf Mold," etc.) with high accuracy and recall.

The model achieves performance improvements through:

Transfer Learning: Leveraging pre-trained models for better generalization.
Hyperparameter Optimization: Tuning model parameters for optimal performance.
Directory Structure
This repository follows a specific directory structure to organize the code, data, and model results for easier navigation and reproducibility.

bash
Copy
Edit
Plant-Disease-Classification/
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   ├── mobilenet/
│   ├── inceptionv3/
│   └── vgg16/
│
├── notebooks/
│   └── model-training.ipynb
│
├── results/
│   ├── mobilenet_results.txt
│   ├── inceptionv3_results.txt
│   └── vgg16_results.txt
│
├── scripts/
│   ├── preprocess.py
│   ├── train_model.py
│   └── evaluate_model.py
│
└── README.md


Explanation of Directories:
data/: Contains the datasets required for training, validation, and testing the models.

train/: Directory containing training images, organized by disease class (e.g., 'Potato Early Blight', 'Tomato Leaf Mold').
validation/: Directory containing images used for validation.
test/: Directory for testing images (if applicable).
models/: Contains directories for each model (MobileNet, InceptionV3, VGG16), which store the trained model weights and configuration.

mobilenet/, inceptionv3/, vgg16/: Each model directory contains the corresponding trained weights and configuration files.
notebooks/: Jupyter notebooks for model training and experimentation.

model-training.ipynb: Notebook that trains and evaluates models.
results/: Contains the evaluation results of each model.

mobilenet_results.txt, inceptionv3_results.txt, vgg16_results.txt: Text files storing the classification report, including precision, recall, F1-score, and accuracy metrics for each model.
scripts/: Python scripts for preprocessing data, training models, and evaluating their performance.

preprocess.py: Script for data preprocessing, such as resizing images, normalizing, and augmenting data.
train_model.py: Script for training the model.
evaluate_model.py: Script for evaluating model performance after training.
Data Directory Organization
The data/ directory is organized by train, validation, and test subdirectories. Each of these directories contains images from different plant disease classes.

Importance of Data Organization:
Training Data: These images are used to train the models and learn features for classification. The images should be grouped by disease class for the model to learn and differentiate between them.
Validation Data: This data is used during the model training process to tune hyperparameters and avoid overfitting. It provides an intermediate measure of performance.
Test Data: After training, the model is evaluated on this data to assess generalization performance. It is important that the test data remains unseen during training to ensure unbiased results.
Where to Place Data:
Training images go into data/train/ (subdirectories should represent each disease class).
Validation images go into data/validation/ (organized similarly to the training data).
Test images go into data/test/ (if applicable).
Model Performance
MobileNet Results
Accuracy: 96%
Macro Average: Precision (0.96), Recall (0.96), F1-score (0.96)
Weighted Average: Precision (0.96), Recall (0.96), F1-score (0.96)
MobileNet achieves the highest accuracy among the models tested. It performs excellently across all classes, especially in terms of recall, ensuring that most instances of each class are detected.

InceptionV3 Results
Accuracy: 92%
Macro Average: Precision (0.93), Recall (0.92), F1-score (0.92)
Weighted Average: Precision (0.93), Recall (0.92), F1-score (0.92)
InceptionV3 performs slightly worse than MobileNet but still achieves high accuracy and recall across classes, particularly with perfect recall for some classes like "Tomato Leaf Mold."

VGG16 Results
Accuracy: 89%
Macro Average: Precision (0.91), Recall (0.89), F1-score (0.88)
Weighted Average: Precision (0.91), Recall (0.89), F1-score (0.88)
VGG16's accuracy is lower compared to MobileNet and InceptionV3. While the recall is high for many classes, precision suffers in some cases, leading to a lower overall F1-score.

Summary:
MobileNet performs best overall, achieving high accuracy, recall, and F1-score.
InceptionV3 follows closely, with high recall and good performance across most classes.
VGG16 performs well in terms of recall, but its precision and overall accuracy are lower compared to the other two models.
Usage Instructions
Requirements
Python 3.x
TensorFlow
Keras
OpenCV
NumPy
Matplotlib
Seaborn
Setup
Clone the repository:

bash
Copy
Edit
git clone https://github.com/MALEKA-SKHUMBUSO/Enhancing-Plant-Disease-Classification-Using-Transfer-Learning-and-Hyper-parameter-Optimization.git
Install required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Place the image dataset into the data/ directory, organizing it into train/, validation/, and test/ subdirectories.

Run the training script to train the model:

bash
Copy
Edit
python scripts/train_model.py
After training, evaluate the model:

bash
Copy
Edit
python scripts/evaluate_model.py
Conclusion
This project demonstrates the effectiveness of using transfer learning with pre-trained models to classify plant diseases. MobileNet provides the best results in terms of accuracy and recall, while InceptionV3 and VGG16 are also strong contenders with good overall performance. The performance can be further improved by fine-tuning models or experimenting with different architectures or hyperparameters.

Feel free to explore the different models, hyperparameters, and training strategies to enhance the performance for specific plant disease classes!
