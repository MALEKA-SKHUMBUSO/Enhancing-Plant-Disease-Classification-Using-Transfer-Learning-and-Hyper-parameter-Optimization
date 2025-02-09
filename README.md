# Enhancing-Plant-Disease-Classification-Using-Transfer-Learning-and-Hyper-parameter-Optimization
This project aims to classify plant diseases using deep learning models: MobileNet, InceptionV3, and VGG16. The models are trained and evaluated on a plant disease dataset, with hyperparameter optimization included in each model’s training script for improved performance.

Dataset
The dataset used in this project can be found on Kaggle: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

New Plant Diseases Dataset
This dataset contains images of different plant diseases, classified into various plant species and disease types.

Importance of Data Organization
The data is organized into separate folders for training, validation, and testing. Each folder contains subfolders for each class of plants, as follows:

train/: Contains images for training the models.
validation/: Contains images for validating the models during training.
test/: Contains images for testing the models' performance.
This structure ensures smooth data processing during the training and evaluation phases.

Models
Three deep learning models are used in this project:

MobileNet: A lightweight model ideal for mobile and embedded devices.
InceptionV3: A deeper architecture with multiple convolutional layers for more accurate predictions.
VGG16: A classical model known for its simplicity and strong performance in image classification tasks.
Each model is trained using its own dedicated script, and hyperparameter optimization is included directly within each script to achieve the best results.

Hyperparameter Optimization
Each model's script contains embedded hyperparameter optimization, fine-tuning key hyperparameters like:

Learning rate
Batch size
Number of epochs
Optimizer choices
This ensures that each model is trained with the optimal settings, leading to enhanced accuracy and performance.

Results
The models were evaluated based on precision, recall, and F1-score for each class. The results for each model were summarized, showing the overall accuracy and performance metrics for each class.

Summary of Results
MobileNet: Achieved an accuracy of 96%, with high precision and recall for most classes.
InceptionV3: Achieved an accuracy of 92%, with strong precision but slightly lower recall for some classes.
VGG16: Achieved an accuracy of 89%, with good precision and recall, though slightly less consistent across classes compared to other models.
Model Training and Hyperparameter Tuning Scripts
1. Plant_MobileNet.py
Trains the MobileNet model on the dataset.
Includes hyperparameter tuning for learning rate, batch size, and number of epochs to maximize model performance.
2. Plant_InceptionV3.py
Trains the InceptionV3 model on the dataset.
Hyperparameter optimization is applied to fine-tune the model's performance for higher accuracy.
3. Plant_VGG16.py
Trains the VGG16 model on the dataset.
Hyperparameter tuning ensures optimal settings for faster convergence and higher accuracy.
Each script handles its own preprocessing, training, and evaluation, with the hyperparameter optimization embedded within the script to find the best configuration for each model.

Conclusion
This project demonstrates the use of deep learning models for plant disease classification. The models (MobileNet, InceptionV3, and VGG16) are trained and evaluated on a real-world dataset, with hyperparameter optimization applied to enhance performance. MobileNet showed the best performance, but InceptionV3 and VGG16 also provided strong results.


