import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data for different skewness levels
x = np.linspace(-4, 4, 1000)

# Symmetric distribution (normal)
symmetric = norm.pdf(x, loc=0, scale=1)  # Normal distribution

# Positive skewness (right skewed)
positive_skew = norm.pdf(x, loc=-0.5, scale=1)

# Negative skewness (left skewed)
negative_skew = norm.pdf(x, loc=0.5, scale=1)

# Plot the distributions
plt.figure(figsize=(10, 6))
plt.plot(x, symmetric, label='Symmetric Distribution (Skewness = 0)', linewidth=2)
plt.plot(x, positive_skew, label='Positive Skewness (Right Skewed)', linestyle='--', linewidth=2)
plt.plot(x, negative_skew, label='Negative Skewness (Left Skewed)', linestyle='-.', linewidth=2)

plt.title('Distributions with Different Skewness', fontsize=16)
plt.xlabel('Value', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5)
plt.show()
