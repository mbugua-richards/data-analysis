import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pyreadstat
import os

# Create a directory to store the images
if not os.path.exists('output_images'):
    os.makedirs('output_images')

# Load the datasets
file_a, meta_a = pyreadstat.read_sav('FileA.sav')
file_b, meta_b = pyreadstat.read_sav('FileB.sav')

# Ensure correct column names for file_a
file_a.columns = ['ParticipantID', 'Group', 'CFMT', 'SpotlightSize', 'Aprime']

# Rename columns in file_b to match the data structure
file_b.columns = ['Subject', 'Sex', 'Age', 'VectionStrength', 'SickWell', 'UnexpectedVection']

# Inspect the data
print("File A:")
print(file_a.head())
print(file_a.info())
print("\nFile B:")
print(file_b.head())
print(file_b.info())

# Check for missing values
print("\nMissing values in File A:")
print(file_a.isnull().sum())
print("\nMissing values in File B:")
print(file_b.isnull().sum())

# Data cleaning and outlier removal
def remove_outliers(df, columns, z_threshold=3):
    for column in columns:
        z_scores = np.abs(stats.zscore(df[column]))
        df = df[z_scores < z_threshold]
    return df

# For File A
columns_to_check_a = ['CFMT', 'SpotlightSize', 'Aprime']
file_a_clean = remove_outliers(file_a, columns_to_check_a)

# For File B
columns_to_check_b = ['VectionStrength']
file_b_clean = remove_outliers(file_b, columns_to_check_b)

print(f"\nRows in File A before cleaning: {len(file_a)}, after cleaning: {len(file_a_clean)}")
print(f"Rows in File B before cleaning: {len(file_b)}, after cleaning: {len(file_b_clean)}")

# Visualize distributions before and after cleaning
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, col in enumerate(columns_to_check_a):
    sns.histplot(file_a[col], ax=axes[0, i], kde=True)
    sns.histplot(file_a_clean[col], ax=axes[1, i], kde=True)
    axes[0, i].set_title(f'{col} - Before')
    axes[1, i].set_title(f'{col} - After')
plt.tight_layout()
plt.savefig('output_images/distributions.png')
plt.close()

# 1. Face Recognition Analysis

def face_recognition_model(data):
    # Create interaction term
    data['CFMT_Spotlight'] = data['CFMT'] * data['SpotlightSize']
    
    # Fit linear mixed-effects model
    model = smf.mixedlm("Aprime ~ CFMT + SpotlightSize + CFMT_Spotlight", 
                        data=data, groups=data["ParticipantID"])
    results = model.fit()
    
    return results

# Run the model and visualize results
results_face = face_recognition_model(file_a_clean)
print("\nFace Recognition Model Results:")
print(results_face.summary())

# Plot interaction effect
plt.figure(figsize=(10, 6))
sns.scatterplot(data=file_a_clean, x='SpotlightSize', y='Aprime', hue='CFMT')
plt.title('Interaction between Spotlight Size and CFMT on Aprime')
plt.savefig('output_images/face_recognition_interaction.png')
plt.close()

# 2. Cybersickness Analysis

def cybersickness_model(data):
    # Convert SickWell to binary (0 for 'Well', 1 for 'Sick')
    data['Sick'] = (data['SickWell'] == 'Sick').astype(int)
    # Convert UnexpectedVection to binary (0 for 'No', 1 for 'Yes')
    data['UnexpectedVection_Binary'] = (data['UnexpectedVection'] == 'Yes').astype(int)
    
    model = smf.logit("Sick ~ VectionStrength + UnexpectedVection_Binary", data=data)
    results = model.fit()
    return results

results_cyber = cybersickness_model(file_b_clean)
print("\nCybersickness Model Results:")
print(results_cyber.summary())

# Plot results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=file_b_clean, x='VectionStrength', y='SickWell', hue='UnexpectedVection')
plt.title('Effect of Vection Strength and Unexpected Vection on Cybersickness')
plt.savefig('output_images/cybersickness_scatter.png')
plt.close()

# Calculate odds and probability
odds_unexpected = np.exp(results_cyber.params['UnexpectedVection_Binary'])
prob_sick = results_cyber.predict({'VectionStrength': 10, 'UnexpectedVection_Binary': 1})[0]

print(f"\nOdds of feeling sick with Unexpected Vection: {odds_unexpected:.2f}")
print(f"Probability of being sick (Unexpected Vection, Vection Strength=10): {prob_sick:.2f}")

print("\nAll plots have been saved in the 'output_images' directory.")