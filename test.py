from NeuralNet import NeuralNet
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_model(filename="best_model.npz"):
    data = np.load(filename)
    return NeuralNet(data['w1'], data['b1'], data['w2'], data['b2'], data['w3'], data['b3'])

nn = load_model("best_model.npz")
print("Loaded model successfully")


def load_digit_image(filepath):
    # Load and convert to grayscale
    img = Image.open(filepath).convert("L")  # 'L' mode = grayscale

    # Resize to 28x28 if needed
    img = img.resize((28, 28))

    # Invert colors if background is white and digit is dark
    img = np.array(img)
    if np.mean(img) > 127:
        img = 255 - img  # invert if white background

    # Normalize to match training data
    img = img / 255.0

    # Flatten to shape (1, 784)
    img_flat = img.reshape(1, -1)

    return img_flat

# digit_img = load_digit_image("Test_Images/.png")
# output = nn.forward_prop(digit_img)
# prediction = np.argmax(output)
# print("Predicted digit:", prediction)

X = np.load("X_mnist.npy")
y = np.load("y_mnist.npy")

rand = np.random.randint(0, 70000)

output = nn.forward_prop(X[rand])
prediction = np.argmax(output)
print("Actual digit", y[rand])
print("Predicted digit:", prediction)
plt.imshow(X[rand].reshape(28, 28), cmap='gray')
plt.title(str(y[rand]))
plt.axis('off')
plt.show()

