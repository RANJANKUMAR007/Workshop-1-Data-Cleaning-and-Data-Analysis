# Workshop-1-Data-Cleaning-and-Data-Analysis
# Name:Ranjan Kumar G
# Reg no:212223240138
# Aim
To load, clean, and visualize the supermarket sales data. This includes detecting/removing outliers (IQR) and performing univariate (count plots) and multivariate (pairplots) analysis.

# Materials Required
Python 3.x
Pandas
Seaborn
Matplotlib
jupytr or google colab


# Steps
Load Data: The supermarket (2).csv file is loaded into a Pandas DataFrame.

Clean Data: The script checks for and removes any missing values or duplicate rows.

Detect Outliers: Boxplots are generated for all numerical columns to visually identify outliers.

Remove Outliers: The Interquartile Range (IQR) method is applied to remove statistical outliers from the dataset.

Visualize: The script generates count plots for categorical data and a pairplot for numerical data to analyze relationships.

# Program
~~~
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for plots
sns.set(style="whitegrid")

# Define the file path
file_path = 'supermarket (2).csv'

try:
    # Load the dataset
    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape}")

    # --- 1. Data Cleaning Process ---
    print("\n--- 1. Data Cleaning ---")
    initial_rows = df.shape[0]
    
    # Remove missing values
    df_cleaned = df.dropna()
    rows_dropped_nulls = initial_rows - df_cleaned.shape[0]
    
    # Remove duplicate rows
    initial_rows_no_nulls = df_cleaned.shape[0]
    df_cleaned = df_cleaned.drop_duplicates()
    rows_dropped_duplicates = initial_rows_no_nulls - df_cleaned.shape[0]
    
    print(f"Removed {rows_dropped_nulls} rows with missing values.")
    print(f"Removed {rows_dropped_duplicates} duplicate rows.")
    print(f"Shape after cleaning: {df_cleaned.shape}")

    # --- 2. Boxplot Method to Detect Outliers ---
    print("\n--- 2. Detecting Outliers (Boxplots) ---")
    
    # Automatically select numerical columns
    numerical_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
    
    # Exclude 'gross margin percentage' as it was found to be constant
    if 'gross margin percentage' in numerical_cols:
        numerical_cols.remove('gross margin percentage') 
        
    print(f"Generating boxplots for: {', '.join(numerical_cols)}")

    # Dynamically create a grid for the plots
    num_cols_plot = len(numerical_cols)
    num_rows_plot = (num_cols_plot + 2) // 3  # 3 columns per row
    
    plt.figure(figsize=(15, num_rows_plot * 5))
    for i, col in enumerate(numerical_cols):
        plt.subplot(num_rows_plot, 3, i + 1)
        sns.boxplot(y=df_cleaned[col])
        plt.title(f'Boxplot of {col}')
        plt.ylabel(col)
        
    plt.tight_layout()
    plt.savefig('boxplots_for_outliers_simple.png')
    print("Saved boxplots to 'boxplots_for_outliers_simple.png'")

    # --- 3. IQR Method to Remove Outliers ---
    print("\n--- 3. Removing Outliers (IQR) ---")
    print(f"Shape before IQR outlier removal: {df_cleaned.shape}")
    
    df_no_outliers = df_cleaned.copy()
    
    # Loop through each numerical column to find and remove outliers
    for col in numerical_cols:
        Q1 = df_no_outliers[col].quantile(0.25)
        Q3 = df_no_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter the DataFrame
        rows_before = df_no_outliers.shape[0]
        df_no_outliers = df_no_outliers[
            (df_no_outliers[col] >= lower_bound) & 
            (df_no_outliers[col] <= upper_bound)
        ]
        rows_removed = rows_before - df_no_outliers.shape[0]
        if rows_removed > 0:
            print(f"Removed {rows_removed} outliers from '{col}'.")

    print(f"Shape after IQR outlier removal: {df_no_outliers.shape}")

    # --- 4. Count plot method for univariate analysis ---
    print("\n--- 4. Univariate Analysis (Count Plots) ---")
    
    # Automatically select categorical columns
    # Exclude high-cardinality or ID-like object columns
    categorical_cols = df_no_outliers.select_dtypes(include='object').columns.tolist()
    exclude_cols = ['Invoice ID', 'Date', 'Time'] # These aren't true categories
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

    print(f"Generating count plots for: {', '.join(categorical_cols)}")
    
    # Dynamically create a grid for the plots
    num_cat_cols = len(categorical_cols)
    num_cat_rows = (num_cat_cols + 1) // 2 # 2 columns per row
    
    plt.figure(figsize=(14, num_cat_rows * 6))
    for i, col in enumerate(categorical_cols):
        plt.subplot(num_cat_rows, 2, i + 1)
        # Order bars by frequency for better readability
        sns.countplot(data=df_no_outliers, x=col, order=df_no_outliers[col].value_counts().index)
        plt.title(f'Count Plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        
        # Automatically rotate labels if they have many unique values
        if df_no_outliers[col].nunique() > 5:
            plt.xticks(rotation=45, ha='right')
        
    plt.tight_layout()
    plt.savefig('countplots_univariate_simple.png')
    print("Saved count plots to 'countplots_univariate_simple.png'")

    # --- 5. PairPlot method for multivariate analysis ---
    print("\n--- 5. Multivariate Analysis (Pairplot) ---")
    
    # Use a subset of key numerical columns for the pairplot
    pairplot_cols = ['Unit price', 'Quantity', 'Total', 'Rating']
    
    # Ensure the columns still exist after cleaning
    pairplot_cols_existing = [col for col in pairplot_cols if col in df_no_outliers.columns]
    
    if pairplot_cols_existing:
        # Using diag_kind='kde' for a cleaner (kernel density) diagonal plot
        sns.pairplot(df_no_outliers[pairplot_cols_existing], diag_kind='kde')
        plt.suptitle('Pairplot of Key Numerical Variables (Post-Cleaning)', y=1.02)
        plt.savefig('pairplot_multivariate_simple.png')
        print("Saved pairplot to 'pairplot_multivariate_simple.png'")
    else:
        print("Could not generate pairplot - key columns not found.")

except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
~~~
# Output 
<img width="1644" height="540" alt="image" src="https://github.com/user-attachments/assets/5649f51c-f991-410b-9aa7-51375db6f3d9" />
<img width="1636" height="539" alt="image" src="https://github.com/user-attachments/assets/e6a45d5e-7c49-463d-bcdd-6ee4a34d46aa" />
<img width="1648" height="652" alt="image" src="https://github.com/user-attachments/assets/410d5351-9996-43e8-9da4-56e0a984ac6f" />
<img width="1615" height="646" alt="image" src="https://github.com/user-attachments/assets/4853bf4f-e2a9-4045-ba5c-e7833627bb7c" />
<img width="1316" height="454" alt="image" src="https://github.com/user-attachments/assets/84eae647-326c-460c-8166-d855f3677431" />
<img width="1226" height="324" alt="image" src="https://github.com/user-attachments/assets/04fd611d-c496-49f1-a815-68fc1808a101" />
<img width="1240" height="477" alt="image" src="https://github.com/user-attachments/assets/4bec6d84-4293-457f-963d-ce067a407ec4" />

