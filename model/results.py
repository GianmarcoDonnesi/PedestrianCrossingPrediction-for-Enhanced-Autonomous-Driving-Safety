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
