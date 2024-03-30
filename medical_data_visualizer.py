# Import modules 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
"""
    -> We are importing the data from the project 'medical_examination.csv' file 
    -> This is the medical data which we are provided as part of the problem 
    -> We are taking this CSV data and importing it into the project in a variable called df
"""
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
"""
    -> In this line, we are adding a column to the dataset, called 'overweight'
    -> We set it equal to the values which we want it to store
    -> We want this to be an entire column of boolean values, which tell us if the person is overweight or not
    -> We do this using the numpy .where method 
    -> We set this equal to the BMI of the people (using a calculation), and then asking the computer to calculate if these 
        BMIs are over 25 or not 
"""
df['overweight'] = np.where((df['weight'] / ((df['height']/100)**2)) > 25, 1, 0)

# Normalise data: 0 for good, 1 for bad
"""
	-> To normalise the data 
	-> We first define a dictionary of three key-value pairs which we want to normalise the results between 
	-> The bottom two lines in this section take the 'cholesterol' and 'gluc' columns of the data frame and normalise them 
        according to these dictionary values 
	-> We are using the .map method to perform this normalisation 
"""
norm_dict = {1: 0, 2: 1, 3: 1}
df['cholesterol'] = df['cholesterol'].map(norm_dict)
df['gluc'] = df['gluc'].map(norm_dict)

# Draw Categorical Plot
"""
    -> Defining the draw_cat_plot function 
    -> This is a plotting function, which returns a figure 
    -> We want to generate a figure 
    -> We first generate a data frame for the cat plot <- This is equal to the df_cat variable: 
        -> This is done with the .melt method  <- We are transforming the data frame into a new one for plotting categorical data
            -> It is transforming the data from wide to long format
            -> Melting something is telling it which columns to transform into rows 
            -> The arguments to this tell the code which columns in the data frame to melt and which ones to leave alone 
        -> The second line in this section defines a new data frame and sets it equal to the variable called `df_cat`
            -> This groups the data frame we just manipulated according to its different attributes ('variable', 'value' and 'cardio')
            -> We then count the number of elements in each of these columns, using the .count method 
            -> We rename the values in the columns and reset the index of the database 

    -> We then create a heatmap for the plot with the seaborn module:  
        -> We have the database which we want to create the heatmap with, df_cat 
        -> We first set the style of the plot to whitegrid using the .set_style method 
        -> Then we create the catplot with the seaborn module using the .catplot method and set this equal to the `fig` variable 
        -> So we have the figure which we want the function to return set equal to the `fig` variable 
            -> We then save and return this figure 
            -> Running this function will return a catplot
"""

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
"""
    -> Defining the draw_heat_map function 
    -> This is another plotting function, which returns a second figure 
    -> This is the function which we want to return a heatmap 
    -> We first define a dataframe which we want to create this for -> This is the df_heat variable 
    -> We are setting the columns of this dataframe equal to the columns of the `df` dataframe, but we are also  filtering out the 
        data which we don't want 
    -> We are assuming that invalid data is the data at the extremes of this distribution 
    -> We then define a heat map and set it equal to the `corr` variable 
    -> We then define another variable called mask
        -> This takes the triangle at the top of the heat map matrix 
        -> We are zeroing out the lower part of this matrix  
    -> Then we generate the figure which we want the function to return 
        -> We do this using 12x12 subplots 
        -> Setting the style we want the seaborn heatmap to use 
        -> Then creating and saving the heatmap we want the function to return, and having it return it 
    -> So we've created a function which generates heatmaps  
    -> We are first generating an array of correlations using .corr(), and then we are converting them into a heatmap using seaborn  
"""

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