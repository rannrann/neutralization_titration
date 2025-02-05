import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Draw the digital scale
scale_body = patches.Rectangle((2, 1), 6, 1, edgecolor='black', facecolor='lightgray')
ax.add_patch(scale_body)
ax.text(5, 1.5, 'Digital Scale', fontsize=10, ha='center')

# Draw the graduated cylinder on the scale
cylinder = patches.Rectangle((4, 2), 1, 4, edgecolor='black', facecolor='lightblue')
ax.add_patch(cylinder)
ax.text(4.5, 6.2, 'Graduated Cylinder (20 mL)', fontsize=10, ha='center')

# Add water level in the cylinder
water = patches.Rectangle((4, 2), 1, 2, edgecolor='none', facecolor='blue', alpha=0.5)
ax.add_patch(water)
ax.text(4.5, 4.2, 'Water: 10 mL', fontsize=9, ha='center', color='white')

# Draw the beaker
beaker = patches.Rectangle((1, 4), 2, 3, edgecolor='black', facecolor='lightblue')
ax.add_patch(beaker)
ax.text(2, 7.2, 'Beaker', fontsize=10, ha='center')

# Draw water pouring from the beaker
ax.plot([2, 4.5], [4, 5], color='blue', linestyle='--', linewidth=1)
ax.text(2.5, 5.5, 'Pouring Water', fontsize=9, ha='left', color='blue')

# Set the axis limits and labels
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Title and display
plt.title('Water Measurement Experiment', fontsize=14)
plt.show()
