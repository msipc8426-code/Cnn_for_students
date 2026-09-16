# CNN for students

A small convolutional neural network that classifies handwritten digits (MNIST),
plus a web UI where you can draw a digit and see what the model predicts.

## Setup

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python cnn.py
```

This trains for 5 epochs, prints the test accuracy and saves `mnist_cnn.keras`.

## Run the UI

```bash
python app.py
```

Open http://localhost:5000, draw a digit in the box and press **Predict**.
The page shows the predicted digit, the confidence and the probability of every class.
