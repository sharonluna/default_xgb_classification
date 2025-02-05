## Utils file #####################################
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from scipy.special import softmax

def count_na(df: pd.DataFrame, 
             threshold: Optional[float] = None,
             verbose: bool = True) -> pd.Series:
    """
    Calculate the percentage of missing values (NA) for each column in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to analyze for missing values
        threshold (float, optional): If provided, only return columns with NA percentage 
                                   above this value (0-100). Defaults to None.
        verbose (bool): Whether to print the number of columns with missing values. 
                       Defaults to True.
    
    Returns:
        pd.Series: 
            - Series with percentage of missing values (columns as index)
    """

    # Calculate NA percentages
    na_counts = df.isna().sum()
    na_percentages = 100 * na_counts / df.shape[0]
    
    # Filter for columns that have any missing values
    na_percentages = na_percentages[na_percentages > 0]
    
    # Apply threshold if provided
    if threshold is not None:
        na_percentages = na_percentages[na_percentages > threshold]
    
    na_percentages = na_percentages.round(2)
    
    # Print number of columns with missing values if verbose
    if verbose:
        print(f"Number of columns with missing values: {len(na_percentages)}")
    
    return na_percentages

def plot_na(na_counts: pd.Series, threshold = None)-> None:
    """
    Plot percentage of missing values for each column in a dataset.

    Args:
        na_counts (pd.Series): Series containing percentage of missing values per column
        threshold (float, optional): Define reference horizontal line. Defaults to None. If None plot at 40%
        
    Returns:
        None: Displays the plot
    """  

    if threshold is None:
        threshold = 40

    fig = plt.figure(figsize=(18,6))
    ax = sns.pointplot(x=na_counts.index,y=na_counts.values,color='blue')
    plt.xticks(rotation =90,fontsize =7)
    ax.axhline(threshold, ls='--',color='red')
    plt.title("Percentage of Missing Values by Column")
    plt.ylabel("%")
    plt.xlabel("columns")
    plt.show()
    plt.close()

def drop_high_na_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop columns with NA values exceeding the specified threshold based on training data.
    
    Args:
        train_df (pd.DataFrame): Training dataset
        test_df (pd.DataFrame): Test dataset
        threshold (float, optional): Maximum allowable fraction of NA values (0.0 to 1.0). 
            Defaults to 0.7.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - Cleaned training dataset with high-NA columns removed
            - Cleaned test dataset with the same columns removed
            
    Raises:
        ValueError: If threshold is not between 0 and 1
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    
    # Calculate NA percentages in training set
    train_na_percent = train_df.isna().mean()
    columns_to_drop = train_na_percent[train_na_percent > threshold].index
    
    train_cleaned = train_df.drop(columns=columns_to_drop)
    test_cleaned = test_df.drop(columns=columns_to_drop)
    
    return train_cleaned, test_cleaned

def interactive_stacked_plot(df: pd.DataFrame, column: str, group: str, title: str = None) -> None:
    """
    Generates an interactive stacked bar plot showing the relative (percentage)
    and absolute (count) frequency distribution of a specified categorical column,
    grouped by another column. Each stack sums to 100% within each 
    primary category on the x-axis.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data to be plotted.
        column (str): The main column to display along the x-axis.
        group (str): The categorical column used to color-code the stacks in the plot.
        title (str, optional): Title 

    Returns:
        None: Displays the plot directly.
    """
    # Create a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()
    
    # Convert group column to category if it's numeric
    if pd.api.types.is_numeric_dtype(plot_df[group]):
        plot_df[group] = plot_df[group].astype(str).astype('str')
    
    percentage_data = (
        plot_df
        .fillna({group: "Unknown"})
        .groupby([column, group])
        .size()
        .reset_index(name='count')
    )
    
    percentage_data['percent'] = (
        percentage_data.groupby(column)['count']
        .transform(lambda x: 100 * x / x.sum())
    )

    fig = px.bar(
        percentage_data, 
        x=column, 
        y='percent', 
        color=group, 
        barmode='stack', 
        text='count',
        template="simple_white"
    )

    fig.update_traces(texttemplate='%{text} (%{y:.2f}%)') 

    if title is None:
        title = f"Frequency - {column} per {group}"

    fig.update_layout(
        title=title,
        xaxis_title=column,
        yaxis_title="Frequency",
        legend_title=group
    )
    
    fig.show()


def get_feature_importances_shap_values(shap_values, features, threshold = None, verbose = True):
    '''
    Prints the feature importances based on SHAP values in an ordered way.

    Parameters:
        hhap_values: The SHAP values calculated from a shap.Explainer object
        features: The name of the features, on the order presented to the explainer
        verbose: Whether to print feature importances. Defaults to True.
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    
    # Calculates the normalized version
    importances_norm = softmax(importances)

    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}

    # filter threshold
    if threshold is not None:
        feature_importances = {fea: imp for fea, imp in feature_importances.items() if imp > threshold}
        feature_importances_norm = {fea: imp for fea, imp in feature_importances_norm.items() if imp > threshold}

    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    # Prints the feature importances
    if verbose:
        for k, v in feature_importances.items():
            print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

    return feature_importances

