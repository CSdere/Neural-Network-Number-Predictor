import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('results.csv')

x = np.linspace(0, np.pi * 2, 1000)
y = np.sin(x)

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='True sin(x)', alpha=0.7)
plt.scatter(df['x'], df['predicted'], label='Predicted', alpha=0.7, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Neural Network Approximation of sin(x)')
plt.legend()
plt.grid(True)
plt.show()
