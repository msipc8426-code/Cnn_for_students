# CNN for Students 🧠

Hey! 👋

This is a simple **Convolutional Neural Network (CNN)** project built with **TensorFlow/Keras**.

The model learns to recognize handwritten digits using the **MNIST dataset**.

The goal of this project is to give students a simple way to understand how a CNN works and experiment with it.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/msipc8426-code/Cnn_for_students.git
```

### 2. Go into the project folder

```bash
cd Cnn_for_students
```

### 3. Install TensorFlow

Make sure you have Python installed, then run:

```bash
pip install tensorflow
```

### 4. Run the CNN

```bash
python cnn.py
```

That's it! 🎉

The program will automatically load the MNIST dataset, train the CNN, and evaluate it on the test data.

## 📁 Project Structure

```text
Cnn_for_students/
│
├── cnn.py
└── README.md
```

## 🧩 What's Inside?

The CNN follows a simple pipeline:

```text
MNIST Images
     ↓
Convolution
     ↓
Max Pooling
     ↓
Convolution
     ↓
Max Pooling
     ↓
Flatten
     ↓
Dense Layer
     ↓
Prediction
```

Feel free to modify the model, change the number of epochs or filters, add new layers, and see what happens! 🚀

Happy learning! 😊
