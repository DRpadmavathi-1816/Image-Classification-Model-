# Image-Classification-Model

Image Classification project using CNN in TensorFlow:

# INTERNSHIP INFORMATION 

- Name : Dhupam Renuka Padmavathi 
- Company : CODETECH IT SOLUTIONS PVT Ltd.
- Domain : Machine learning 
- Duration : 6 Weeks 
- Mentor : Neela Shantosh Kumar 


# 🧠 Image Classification with Convolutional Neural Network (CNN)

This project implements an image classification pipeline using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained on the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 classes.



📁 Project Structure

File/Folders	Description

- Image classification.ipynb	Jupyter notebook with complete code for loading data, building and training the CNN, and evaluating performance.




📊 Dataset - CIFAR-10

CIFAR-10 is a collection of images grouped into 10 different categories:

- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.


Contains:

- 50,000 training images

- 10,000 testing images


Dataset is automatically downloaded using tf.keras.datasets.


🛠️ Tech Stack

- Python

- TensorFlow / Keras

- Matplotlib

- NumPy



🚀 Model Architecture

The CNN model follows this structure:

Input: 32x32 RGB image
↓
Conv2D → ReLU → MaxPooling
↓
Conv2D → ReLU → MaxPooling
↓
Flatten
↓
Dense → ReLU
↓
Dropout
↓
Dense (Output Layer with Softmax)



🔍 Key Steps in the Notebook

1. Import Libraries


2. Load CIFAR-10 Dataset


3. Normalize the Image Data


4. Visualize Sample Images


5. Build CNN Model


6. Compile the Model (using adam optimizer and sparse_categorical_crossentropy loss)


7. Train the Model


8. Evaluate the Model


9. Plot Accuracy & Loss Curves


10. Make Predictions



📈 Model Performance

 Evaluation includes: 
- Accuracy on test set

- Loss/accuracy visualized over epochs


📷 Sample Output

You can visualize:

- A batch of sample images with their true labels.

- Plots of training vs. validation accuracy and loss over epochs.



# 🧪 How to Run

1. Open the notebook: Image classification.ipynb


2. Run all cells (preferably in Google Colab or Jupyter)


3. The CIFAR-10 dataset will auto-download


4. Follow the training and evaluation steps



✅ Requirements

Make sure the following packages are installed:

pip install tensorflow matplotlib numpy


  Future Improvements

- Implement data augmentation
- Experiment with deeper architectures or pretrained models like ResNet
- Add confusion matrix and classification report



🙌 Credits

CIFAR-10 Dataset

TensorFlow & Keras Documentation


