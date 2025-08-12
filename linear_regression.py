import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([2, 4, 6, 8, 10])
y = np.array([4, 8, 12, 16, 20])

learning_rate = 0.01
num_iterations = 1000
m = 0  # slope
b = 0  # intercept

for i in range(num_iterations):
    y_pred = m * x + b
    error = y_pred - y
    cost = np.mean(error ** 2)

    m_gradient = (2 / len(x)) * np.dot(error, x)
    b_gradient = (2 / len(x)) * np.sum(error)

    m = m - (learning_rate * m_gradient)
    b = b - (learning_rate * b_gradient)

model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

print(f"Manual model slope: {m:.4f}, y-intercept: {b:.4f}")
print(f"Sklearn model slope: {model.coef_[0]:.4f}, y-intercept: {model.intercept_:.4f}")
print("Both models should yield similar results for the same data.")

print("Test data")

x_test = np.array([12, 14, 16])
y_test_pred = model.predict(x_test.reshape(-1, 1))

print("Predictions for test data:", y_test_pred)
print("Manual predictions for test data:", m * x_test + b)
print("Sklearn predictions for test data:", model.predict(x_test.reshape(-1, 1)))