# ==========================================
# Fashion MNIST Image Classification using CNN
# Author: Kalyan Kumar Reddy
# ==========================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------
# Load Fashion MNIST Dataset
# ------------------------------------------

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class Names
class_names = [
    'T-shirt',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle Boot'
]

print("Training Images:", train_images.shape)
print("Testing Images:", test_images.shape)

# ------------------------------------------
# Display Sample Images
# ------------------------------------------

plt.figure(figsize=(8,8))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap='gray')
    plt.xlabel(class_names[train_labels[i]])

plt.tight_layout()
plt.show()

# ------------------------------------------
# Normalize Images
# ------------------------------------------

train_images = train_images / 255.0
test_images = test_images / 255.0

# ------------------------------------------
# Reshape Images for CNN
# ------------------------------------------

train_images = train_images.reshape((-1,28,28,1))
test_images = test_images.reshape((-1,28,28,1))

# ------------------------------------------
# Build CNN Model
# ------------------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    ),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        10,
        activation='softmax'
    )

])

# ------------------------------------------
# Compile Model
# ------------------------------------------

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------------------------
# Train Model
# ------------------------------------------

history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    validation_data=(test_images, test_labels)
)

# ------------------------------------------
# Evaluate Model
# ------------------------------------------

test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels
)

print("\nTest Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# ------------------------------------------
# Predict Test Images
# ------------------------------------------

predictions = model.predict(test_images)

index = 15

plt.figure(figsize=(4,4))
plt.imshow(test_images[index].reshape(28,28), cmap='gray')
plt.title("Predicted : " + class_names[np.argmax(predictions[index])])
plt.axis('off')
plt.show()

print("Actual Label    :", class_names[test_labels[index]])
print("Predicted Label :", class_names[np.argmax(predictions[index])])

# ------------------------------------------
# Plot Accuracy
# ------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    marker='o',
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    marker='o',
    label='Validation Accuracy'
)

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------
# Plot Loss
# ------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    history.history['loss'],
    marker='o',
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    marker='o',
    label='Validation Loss'
)

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------
# Save Model
# ------------------------------------------

model.save("FashionMNIST_CNN_Model.keras")

print("\nModel saved successfully!")