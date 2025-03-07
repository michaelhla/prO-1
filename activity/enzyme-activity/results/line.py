import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Read the CSV file
df = pd.read_csv('/Users/michaelhla/Documents/Code/prO-1/data/enzyme-activity/results/moclobemide_processed.csv_results.csv')

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['original_activity'], df['calculated_reward'], alpha=0.5)

# Calculate the regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(df['original_activity'], df['calculated_reward'])
line = slope * df['original_activity'] + intercept

# Plot the regression line
plt.plot(df['original_activity'], line, color='red', label=f'y = {slope:.3f}x + {intercept:.3f}')

# Add R² value to the plot
plt.text(0.05, 0.95, f'R² = {r_value**2:.3f}', 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))

# Customize the plot
plt.xlabel('Original Activity')
plt.ylabel('Calculated Reward')
plt.title('Original Activity vs Calculated Reward')
plt.legend()

# Save the plot
plt.savefig('/Users/michaelhla/Documents/Code/prO-1/data/enzyme-activity/results/moclobemide_processed.csv_correlation.png', 
            dpi=300, 
            bbox_inches='tight')
plt.close()