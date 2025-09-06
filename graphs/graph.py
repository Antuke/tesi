import pandas as pd
import matplotlib.pyplot as plt
import numpy as np





# Load the dataset from a CSV file
try:
    # Make sure to replace 'your_data.csv' with the actual path to your file
    df = pd.read_csv('/user/asessa/dataset tesi/labels_imdb_train.csv')
except FileNotFoundError:
    print("Error: 'your_data.csv' not found. Please make sure the CSV file is in the correct directory.")
    exit()

# Define the direct mapping from integer codes to labels
age_groups = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
age_map = {i: group for i, group in enumerate(age_groups)}
gender_map = {0: 'Male', 1: 'Female'}

# Apply the mappings to create new descriptive columns
df['Age Group'] = df['Age'].map(age_map)
df['Gender'] = df['Gender'].map(gender_map)

# Ensure the age groups are plotted in the correct order by setting it as a categorical type
df['Age Group'] = pd.Categorical(df['Age Group'], categories=age_groups, ordered=True)

# --- MODIFIED SECTION ---
# Group by Age Group and Gender and get the count of samples.
# This is the main change: we will use these counts directly.
age_gender_counts = df.groupby(['Age Group', 'Gender'], observed=False).size().unstack(fill_value=0)

# --- Plotting Section ---
fig, ax = plt.subplots(figsize=(12, 8))

# Define the colors for male and female
colors = {'Male': 'blue', 'Female': 'pink'}

# Plot the stacked bar chart using the calculated counts
# The `plot` function from pandas can simplify stacked bar creation
age_gender_counts.plot(kind='bar', stacked=True, color=colors, ax=ax)

# Add labels and title
ax.set_xlabel('Age Group', fontweight='bold')
ax.set_ylabel('Number of Samples', fontweight='bold')
ax.set_title('Distribution of Samples by Age Group and Gender', fontweight='bold')
ax.legend(title='Gender')
ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels

# Add numerical labels inside each segment of the bar
for c in ax.containers:
    # Create labels for segments, but don't label very small segments
    labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in c]
    ax.bar_label(c, labels=labels, label_type='center', color='white', weight='bold')

# Add a label on top of each bar showing the total count for that group
# To do this, we calculate the total for each bar
totals = age_gender_counts.sum(axis=1)
for i, total in enumerate(totals):
    ax.text(i, total + 5, round(total), ha='center', weight='bold') # '+ 5' is a small offset for visibility


plt.tight_layout()

# Save the plot to a file
plt.savefig('age_gender_distribution_counts.png', dpi=300)

print("Graph has been saved successfully as 'age_gender_distribution_counts.png'")