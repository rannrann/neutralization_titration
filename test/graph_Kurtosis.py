import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data for different kurtosis levels
x = np.linspace(-4, 4, 1000)
normal = norm.pdf(x, loc=0, scale=1)  # Normal distribution (kurtosis=3)

# Generate high kurtosis (leptokurtic)
high_kurtosis = norm.pdf(x, loc=0, scale=0.7)

# Generate low kurtosis (platykurtic)
low_kurtosis = norm.pdf(x, loc=0, scale=1.5)

# Plot the distributions
plt.figure(figsize=(10, 6))
plt.plot(x, normal, label='Normal Distribution (Kurtosis = 3)', linewidth=2)
plt.plot(x, high_kurtosis, label='High Kurtosis (Leptokurtic)', linestyle='--', linewidth=2)
plt.plot(x, low_kurtosis, label='Low Kurtosis (Platykurtic)', linestyle='-.', linewidth=2)

plt.title('Distributions with Different Kurtosis', fontsize=16)
plt.xlabel('Value', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5)
plt.show()
