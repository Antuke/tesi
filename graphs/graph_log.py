import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- SETTINGS ---
use_true_age_numbers = False # This flag is working correctly

# --- SCRIPT START ---

# Load the dataset
try:
    # Using the path you provided
    df = pd.read_csv('/user/asessa/dataset tesi/train_final.csv')
except FileNotFoundError:
    print("Error: The specified CSV file was not found. Please check the path.")
    exit()

# --- Data Preparation (This part is correct) ---
age_groups = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
# age_groups = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]
gender_map = {0: 'Male', 1: 'Female'}

if use_true_age_numbers:
    print("Processing with true age numbers. Binning ages into groups...")
    bins = [-1, 2, 9, 19, 29, 39, 49, 59, 69, np.inf]
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=age_groups, right=True)
else:
    print("Processing with age group codes. Mapping codes to labels...")
    age_map = {i: group for i, group in enumerate(age_groups)}
    df['Age Group'] = df['Age'].map(age_map)

df['Gender'] = df['Gender'].map(gender_map)
df['Age Group'] = pd.Categorical(df['Age Group'], categories=age_groups, ordered=True)

age_gender_counts = df.groupby(['Age Group', 'Gender'], observed=False).size().unstack(fill_value=0)
totals = age_gender_counts.sum(axis=1)
age_gender_percentage = age_gender_counts.div(totals, axis=0).fillna(0) * 100

colors = {'Male': 'blue', 'Female': 'lightpink'}

# --- Create the Panel Chart ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    figsize=(12, 10), 
    sharex=True, 
    gridspec_kw={'height_ratios': [1, 1]}
)
fig.suptitle('Distribution of Samples by Age Group and Gender', fontsize=16, fontweight='bold')

# ===== PLOT 1: Total Sample Distribution (This was already correct) =====
ax1.bar(totals.index, totals.values, color='gray')
ax1.set_yscale('log')
ax1.set_title('Total Number of Samples per Group (Log Scale)')
ax1.set_ylabel('Number of Samples')
for i, total in enumerate(totals):
    if total > 0:
        ax1.text(i, total, f'{int(total)}', ha='center', va='bottom')

# ===== PLOT 2: Gender Proportions (100% Stacked) =====
# --- THIS IS THE CORRECTED LINE ---
# We explicitly set the column order to ['Male', 'Female'] so Male is on the bottom.
# Then we provide the colors in a list that matches this explicit order.
age_gender_percentage[['Male', 'Female']].plot(
    kind='bar', 
    stacked=True, 
    color=[colors['Male'], colors['Female']], 
    ax=ax2
)

ax2.set_title('Gender Proportions within Each Group')
ax2.set_ylabel('Percentage (%)')
ax2.set_xlabel('Age Group', fontweight='bold')
ax2.legend(title='Gender') # Legend will now be correct
ax2.set_ylim(0, 100)

for c in ax2.containers:
    labels = [f'{w:.1f}%' if w > 1 else '' for w in c.datavalues]
    ax2.bar_label(c, labels=labels, label_type='center', color='white', weight='bold')

# --- Final Formatting ---
plt.xticks(rotation=45, ha='right')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('distribution_panel_chart_final.png', dpi=300)

print("Panel chart has been saved as 'distribution_panel_chart_final.png'")