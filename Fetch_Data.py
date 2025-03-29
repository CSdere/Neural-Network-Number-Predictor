import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import fetch_openml

# File paths for saved NumPy arrays
X_path = "X_mnist.npy"
y_path = "y_mnist.npy"

if os.path.exists(X_path) and os.path.exists(y_path):
    # Load pre-saved data
    print("Loading MNIST data from local .npy files...")
    X = np.load(X_path)
    y = np.load(y_path)
else:
    # First time: fetch from OpenML and save to .npy
    print("Downloading MNIST data from OpenML...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist['data'], mnist['target'].astype(int)

    # Normalize pixel values to [0, 1]
    X = X / 255.0

    # Save for future use
    np.save(X_path, X)
    np.save(y_path, y)
    print("Data saved locally for future use.")

# Visualize the first 10 digits
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.title(str(y[i]))
    plt.axis('off')
plt.show()
