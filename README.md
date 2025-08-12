# Linear Regression Implementation Comparison

This project compares a manual implementation of linear regression using gradient descent with scikit-learn's built-in LinearRegression model.

## Project Overview

- Implements linear regression from scratch using gradient descent
- Compares results with scikit-learn's implementation
- Includes visualization of both models' fits
- Demonstrates prediction on test data

## Trained dataset
```python
x = np.array([2, 4, 6, 8, 10])
y = np.array([4, 8, 12, 16, 20])
```

## Results

| Metric          | Manual Implementation | scikit-learn |
|-----------------|-----------------------|--------------|
| Slope (m)       | 1.9990               | 2.0000       |
| Intercept (b)   | 0.0075               | 0.0000       |
| Test Prediction | [23.995, 27.993, 31.991] | [24.0, 28.0, 32.0] |

<img width="1200" height="1200" alt="linear_regression_plot" src="https://github.com/user-attachments/assets/5fc4d778-a47e-4f05-ad90-beb18e8b5d08" />


## Installation

```bash
git clone https://github.com/Mordekai66/linear-regression-comparison.git
cd linear-regression-comparison
pip install -r requirements.txt
```
