import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from reward import calculate_reward, BindingEnergyCalculator

# Read the CSV file
df = pd.read_csv('data.csv')

# Initialize calculator
calculator = BindingEnergyCalculator(device="cuda")

# Calculate rewards for each sequence
rewards = []
for idx, row in df.iterrows():
    reward = calculate_reward(
        sequence=row['sequence'],
        reagents=[row['substrates']], 
        products=None,
        calculator=calculator
    )
    rewards.append(reward)

df['reward'] = rewards

# Create correlation plot
plt.figure(figsize=(8, 6))
sns.regplot(data=df, x='reward', y='LogSpActivity', scatter_kws={'alpha':0.5})

# Calculate correlation coefficient and p-value
r, p = stats.pearsonr(df['reward'], df['LogSpActivity'])
r2 = r**2

# Add regression equation and R2 to plot
plt.text(0.05, 0.95, f'y = {r:.3f}x + {p:.3f}\nR² = {r2:.3f}', 
         transform=plt.gca().transAxes, 
         fontsize=10,
         verticalalignment='top')

plt.title('Correlation between Calculated Reward and LogSpActivity')
plt.xlabel('Calculated Reward')
plt.ylabel('LogSpActivity')

plt.tight_layout()
plt.savefig('reward_correlation.png')
plt.close()
