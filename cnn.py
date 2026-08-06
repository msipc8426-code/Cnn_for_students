import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 1. Load and preprocess the data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalize to 0-1

# 2. Build the CNN architecture
model = models.Sequential([
    # Convolutional layer learns 32 features using 3x3 filters
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Max pooling reduces the spatial size by half
    layers.MaxPooling2D((2, 2)),
    
    # Second convolution and pooling block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten 2D matrices into a 1D vector for dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # Output layer with 10 units for the 10 digit classes (0-9)
    layers.Dense(10, activation='softmax')
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# 5. Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")
