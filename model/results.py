import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the results
with open('./results.pkl', 'rb') as f:
    results = pickle.load(f)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Set up the matplotlib figure
plt.figure(figsize = (15, 5))

# Plot Accuracy
plt.subplot(1, 3, 1)
sns.barplot(data=results_df, x = 'ablation', y = 'accuracy')
plt.title('Validation Accuracy')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.xlabel('Ablation')
