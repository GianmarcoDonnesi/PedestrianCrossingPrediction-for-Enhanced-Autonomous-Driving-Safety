import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the results
with open('./results.pkl', 'rb') as f:
    results = pickle.load(f)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Define a custom color palette
custom_palette = {
    'None': '#1f77b4',  # blue
    'Traffic': '#2ca02c',  # green
    'Vehicle': '#d62728',  # red
    'Appearance': '#ffdf00',  # yellow
    'Attributes': '#ff7f0e'  # orange
}

# Set up the matplotlib figure
plt.figure(figsize = (15, 5))

# Plot Accuracy
plt.subplot(1, 3, 1)
sns.barplot(data=results_df, x = 'ablation', y = 'accuracy', palette=custom_palette, width=0.6)
plt.title('Test Accuracy')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.xlabel('Ablation')

# Plot Recall
plt.subplot(1, 3, 2)
sns.barplot(data = results_df, x = 'ablation', y = 'recall', palette=custom_palette, width=0.6)
plt.title('Test Recall')
plt.ylim(0, 1)
plt.ylabel('Recall')
plt.xlabel('Ablation')

# Plot F1 Score
plt.subplot(1, 3, 3)
sns.barplot(data = results_df, x = 'ablation', y = 'f1_score', palette=custom_palette, width=0.6)
plt.title('Test F1 Score')
plt.ylim(0, 1)
plt.ylabel('F1 Score')
plt.xlabel('Ablation')

plt.tight_layout()
plt.show()
