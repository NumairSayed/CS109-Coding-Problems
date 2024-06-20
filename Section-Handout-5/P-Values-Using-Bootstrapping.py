import numpy as np
from tqdm import tqdm

# Given data
island_a_heights = np.array([13, 12, 7, 16, 9, 11, 7, 10, 9, 8, 9, 7, 16, 7, 9, 8, 13, 10, 11, 9, 13, 13, 10, 10, 9, 7, 7, 6, 7, 8, 12,
                             13, 9, 6, 9, 11, 10, 8, 12, 10, 9, 10, 8, 14, 13, 13, 10, 11, 12, 9])
island_b_heights = np.array([8, 8, 16, 16, 9, 13, 14, 13, 10, 12, 10, 6, 14, 8, 13, 14, 7, 13, 7, 8, 4, 11, 7, 12, 8, 9, 12, 8, 11, 10,
                             12, 6, 10, 15, 11, 12, 3, 8, 11, 10, 10, 8, 12, 8, 11, 6, 7, 10, 8, 5])

observed_var_diff = np.var(island_b_heights, ddof=1) - np.var(island_a_heights, ddof=1)
observed_mean_diff= np.mean(island_b_heights) - np.mean(island_a_heights)
# Combined dataset
combined_heights = np.concatenate((island_a_heights, island_b_heights))

# Bootstrapping
n_iterations = 10000
bootstrap_var_diffs = []
bootstrap_mean_diffs = []
for _ in tqdm(range(n_iterations)):
    # Resample with replacement
    resampled = np.random.choice(combined_heights, size=len(combined_heights), replace=True)
    
    # Split the resampled data into two groups
    resampled_a = resampled[:len(island_a_heights)]
    resampled_b = resampled[len(island_a_heights):]
    
    # Calculate the variance difference for the resampled groups
    var_diff = np.var(resampled_b, ddof=1) - np.var(resampled_a, ddof=1)
    mean_diff = np.mean(resampled_b)-np.mean(resampled_a)
    bootstrap_var_diffs.append(var_diff)
    bootstrap_mean_diffs.append(mean_diff)

# Calculate the p-value
p_value_variance = np.sum(np.abs(bootstrap_var_diffs) >= np.abs(observed_var_diff)) / n_iterations
p_value_mean     = np.sum(np.abs(bootstrap_mean_diffs) >= np.abs(observed_mean_diff)) / n_iterations

print(p_value_variance,p_value_mean)
