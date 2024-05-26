import numpy as np
import matplotlib.pyplot as plt
import math, time

# Parameters
noise = 1
len_scale = 2.5

# Kernel function
def kernel_function(x1, x2, len_scale):
    dist_sq = np.linalg.norm(x1 - x2) ** 2
    term = -1 / (2 * len_scale ** 2)
    return noise * np.exp(dist_sq * term)

# Covariance matrix
def cov_matrix(x1, x2):
    n = x1.shape[0]
    m = x2.shape[0]
    cov_mat = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            cov_mat[i][j] = kernel_function(x1[i], x2[j], len_scale)
    return cov_mat

# GPR train
def GPR_train(trainX, trainY):
    K = cov_matrix(trainX, trainX)
    K_inv = np.linalg.inv(K + noise * np.identity(len(trainX)))
    return K, K_inv

# GPR predict
def GPR_predict(trainX, trainY, testX, K_inv):
    K1 = cov_matrix(trainX, testX)
    K2 = cov_matrix(testX, testX)
    K3 = K2 - np.matmul(K1.T, np.matmul(K_inv, K1)) + noise * np.identity(len(testX))

    mean_prediction = np.matmul(K1.T, np.matmul(K_inv, trainY))
    std_prediction = np.sqrt(np.diag(K3))

    return mean_prediction, std_prediction

# Data generation
trainX = np.linspace(0, 10, num=1000).reshape(-1, 1)
trainY = (trainX * np.sin(trainX)).ravel()

testX = np.linspace(0, 10, num=1000).reshape(-1, 1)
testY = (testX * np.sin(testX)).ravel()

# Train GPR model
print('Training started')
K, K_inv = GPR_train(trainX, trainY)
print("Training complete")

# Predict using GPR model
print('Testing started')
start_time = time.time()
mean_prediction, std_prediction = GPR_predict(trainX, trainY, testX, K_inv)
end_time = time.time()

print("Testing time is", round(end_time - start_time, 2), "seconds")
plt.plot(testX, testY, color='black', label='True Function')
plt.plot(testX, mean_prediction, ls=':', lw=2, color='red', label='GPR Prediction')
plt.fill_between(testX.ravel(),
                 mean_prediction - 1.96 * std_prediction,
                 mean_prediction + 1.96 * std_prediction,
                 alpha=0.2, color='blue', label='95% confidence interval')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
