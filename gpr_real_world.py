import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# RBF kernel for multivariate input
def rbf_kernel(X1, X2, length_scales, variance=1.0):
    N1, D = X1.shape
    N2 = X2.shape[0]
    K = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            sq_diff = np.sum((X1[i] - X2[j]) ** 2 / length_scales ** 2)
            K[i, j] = variance * np.exp(-0.5 * sq_diff)
    return K

# Load dataset
df = pd.read_excel('data/AirQualityUCI.xlsx')

# Data preprocessing
df.dropna(inplace=True)

# Select features and target
D = 2
y = df['AH'].to_numpy()
features = ['C6H6(GT)', 'T']
X = df[features].to_numpy()

# Split data
X_train = X[:7485, :]
X_test = X[7485:, :]
y_train = y[:7485]
y_test = y[7485:]

# Define hyperparameters
length_scales = np.ones(D)
variance = 1.0
noise = 0.1

# Compute kernel matrix
K = rbf_kernel(X_train, X_train, length_scales, variance) + noise ** 2 * np.eye(X_train.shape[0])
K_test = rbf_kernel(X_train, X_test, length_scales, variance)

# Predictive mean and covariance
L = np.linalg.cholesky(K + 1e-6 * np.eye(X_train.shape[0]))
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
y_pred_mean = np.dot(K_test.T, alpha)

# Predictive variance
v = np.linalg.solve(L, K_test)
K_test_test = rbf_kernel(X_test, X_test, length_scales, variance)
y_pred_cov = K_test_test - np.dot(K_test.T, v)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', s=50, label='Training Data')
plt.colorbar()

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_mean, cmap='viridis_r', marker='x', s=50, label='Predictive Mean')
plt.colorbar()

plt.xlabel('C6H6(GT)')
plt.ylabel('Temperature (T)')
plt.title('Gaussian Process Regression (Multivariate Input)')
plt.legend()
plt.show()
