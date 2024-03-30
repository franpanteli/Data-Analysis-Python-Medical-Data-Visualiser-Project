import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = np.where((df['weight'] / ((df['height']/100)**2)) > 25, 1, 0)

# Normalise data: 0 for good, 1 for bad
norm_dict = {1: 0, 2: 1, 3: 1}
df['cholesterol'] = df['cholesterol'].map(norm_dict)
df['gluc'] = df['gluc'].map(norm_dict)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` with selected columns
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # Group and reformat data to split by 'cardio', showing counts of each feature
    df_cat = pd.DataFrame(df_cat.groupby(['variable', 'value', 'cardio'])['value'].count()).rename(columns={'value': 'total'}).reset_index()

    # Draw the catplot with 'sns.catplot()'
    sns.set_style("whitegrid")
    fig = sns.catplot(x='variable', y='total', col='cardio', hue='value', data=df_cat, kind='bar').fig

    # Save and return the plot
    fig.savefig('catplot.png')
    return fig

# Draw Heat Map
def draw_heat_map():
    # Clean the data based on given conditions
    df_heat = df[
    (df['ap_lo'] <= df['ap_hi']) &
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['weight'] >= df['weight'].quantile(0.025)) &
    (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate correlation matrix
    corr = df_heat.corr()

    # Generate mask for upper triangle
    mask = np.triu(corr)

    # Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12,12))

    # Draw the heatmap with 'sns.heatmap()'
    sns.set_style("whitegrid")
    sns.heatmap(corr, annot=True, square=True, mask=mask, fmt=".1f", center=0.88, cbar_kws={"shrink":0.5})

    # Save and return the plot
    fig.savefig('heatmap.png')
    return fig
