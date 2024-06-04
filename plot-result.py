import pandas as pd
import matplotlib.pyplot as plt
import seaborn as  sns
data = {'Model': ['Logistic Regression', 'Deep Neural Network'],
        'Accuracy': [0.6816, 0.8031]}
df = pd.DataFrame(data)


# Set the style
sns.set_theme(style="darkgrid")

# Create a color palette
palette = plt.get_cmap('Set1')

# Plot the data
plt.figure(figsize=(10, 6))
plt.bar(df['Model'], df['Accuracy'], color=palette(1), edgecolor='black')

# Add titles and labels
plt.title('Model Comparison on Phishing Dataset', loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")

# Show the plot
plt.show()
