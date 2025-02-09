# Enhancing-Plant-Disease-Classification-Using-Transfer-Learning-and-Hyper-parameter-Optimization
This project aims to classify plant diseases using deep learning models, including MobileNet, InceptionV3, and VGG16. It involves training the models on a dataset containing images of plants, evaluating their performance, and providing a detailed comparison of their results.

Importance of Data Organization
To ensure smooth execution and easy access to various resources, the data is organized into separate folders for training, validation, and testing. Each folder contains subfolders for each class of plants, organized as follows:

train/: Contains images used for training the models.
validation/: Contains images used for validating the models during training.
test/: Contains images used to test the performance of the trained models.
This structure helps in efficiently loading and processing the data during training and evaluation phases.

Models
Three deep learning models were used for classification:

MobileNet: Known for its lightweight architecture, ideal for mobile and embedded devices.
InceptionV3: A deeper architecture with multiple convolutional layers for more accurate predictions.
VGG16: A classical model known for its simplicity and strong performance in image classification tasks.
Each model is trained and saved in a separate folder to keep track of the individual results.

Results
The performance of each model was evaluated using precision, recall, and F1-score across multiple classes. The accuracy and other metrics were calculated for each model, providing insights into their effectiveness for plant disease classification.

Summary of Results
MobileNet achieved an overall accuracy of 96%, demonstrating strong performance with high precision and recall across most classes.
InceptionV3 achieved an accuracy of 92%, with a high precision rate for most classes, but slightly lower recall for some.
VGG16 showed an accuracy of 89%, with some variations in precision and recall, but still performing reasonably well.
Training the Models
To train the models, the following steps are followed:

Preprocessing: The dataset is preprocessed to ensure that the images are resized and normalized.
Model Training: Each model (MobileNet, InceptionV3, and VGG16) is trained using the training dataset, with the validation dataset used to tune the model during training.
Evaluation: After training, the models are evaluated on the test set to obtain performance metrics like accuracy, precision, recall, and F1-score.
Scripts
The project contains the following Python scripts:

preprocess.py: Handles the preprocessing of images, including resizing and normalizing them.
train_model.py: Trains the deep learning models on the preprocessed data.
evaluate_model.py: Evaluates the trained models and generates performance reports.
Conclusion
This project demonstrates the effectiveness of deep learning models in classifying plant diseases. MobileNet, InceptionV3, and VGG16 provide varying levels of accuracy, and their performance is compared to determine the most suitable model for deployment.
