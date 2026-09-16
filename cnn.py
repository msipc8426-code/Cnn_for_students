import os

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_cnn.keras")


def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train(epochs=5, batch_size=64, model_path=MODEL_PATH):
    (train_images, train_labels), (test_images, test_labels) = load_data()

    model = build_model()
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")

    model.save(model_path)
    print(f"Saved model to {model_path}")
    return model


def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `python cnn.py` first to train one."
        )
    return tf.keras.models.load_model(model_path)


if __name__ == "__main__":
    train()
