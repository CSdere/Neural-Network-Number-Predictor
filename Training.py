import numpy as np
from NeuralNet import NeuralNet

# Load data
X = np.load("X_mnist.npy")
y = np.load("y_mnist.npy")

# One-hot encode labels
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

Y = one_hot(y)

# Split into training and testing
X_train, X_test = X[:60000], X[60000:]
Y_train, Y_test = Y[:60000], Y[60000:]

def save_model(nn, filename="best_model.npz"):
    np.savez(filename,
             w1=nn.w1,
             b1=nn.b1,
             w2=nn.w2,
             b2=nn.b2,
             w3=nn.w3,
             b3=nn.b3)

input_size = 784
hidden1_size = 128
hidden2_size = 64
output_size = 10


# He Initialization for ReLU (recommended)
w1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2. / input_size)
b1 = np.zeros((1, hidden1_size))

w2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2. / hidden1_size)
b2 = np.zeros((1, hidden2_size))

w3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2. / hidden2_size)
b3 = np.zeros((1, output_size))

nn = NeuralNet(w1, b1, w2, b2, w3, b3, lr=0.01)


best_test_acc = 0.0  # or np.inf if you want to track loss instead

for epoch in range(5000):
    # Training pass
    output_train = nn.forward_prop(X_train)
    loss_train = nn.compute_loss(Y_train, output_train)
    nn.backward_prop(X_train, Y_train)
    acc_train = nn.compute_accuracy(Y_train, output_train)

    # Test pass
    output_test = nn.forward_prop(X_test)
    loss_test = nn.compute_loss(Y_test, output_test)
    acc_test = nn.compute_accuracy(Y_test, output_test)

    # Save best model based on test accuracy
    if acc_test > best_test_acc:
        best_test_acc = acc_test
        save_model(nn)
        print(f"Epoch {epoch+1}: New best model saved! Test Acc: {acc_test*100:.2f}%")

    print(f"Epoch {epoch+1} | Train Loss: {loss_train:.4f} | Train Acc: {acc_train*100:.2f}% | "
          f"Test Loss: {loss_test:.4f} | Test Acc: {acc_test*100:.2f}%")
    
def load_model(filename="best_model.npz"):
    data = np.load(filename)
    return NeuralNet(data['w1'], data['b1'], data['w2'], data['b2'], data['w3'], data['b3'])

nn = load_model("best_model.npz")