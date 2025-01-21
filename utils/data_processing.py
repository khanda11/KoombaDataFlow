import pandas as pd
import re

def extract_numeric_value(s):
    if isinstance(s, str):
        match = re.search(r"([-+]?\d+(\.\d+)?)", s)
        return float(match.group(1)) if match else None
    return s if isinstance(s, (int, float)) else None

def preprocess_data(df):
    df.columns = df.columns.str.strip().str.replace("\xa0", "")
    df["Completed"] = pd.to_datetime(df["Completed"].str.replace("EST", "").str.strip(), errors="coerce")

    for col in df.columns:
        if col not in ["Team", "Unique ID", "Client", "Client's Provider", "Completed"]:
            df[col] = df[col].apply(extract_numeric_value)

    return df.dropna(subset=["Unique ID", "Completed"])
'''import pandas as pd
import re

def extract_numeric_value(s):
    """
    Extract numeric values from string inputs.
    Returns None if no valid numeric value is found.
    """
    if isinstance(s, str):
        match = re.search(r"([-+]?\d+(\.\d+)?)", s)
        return float(match.group(1)) if match else None
    return s if isinstance(s, (int, float)) else None

def standardize_column_names(df):
    """
    Standardize column names by removing special characters and extra whitespace.
    """
    return df.columns.str.strip().str.replace("\xa0", "").str.lower()

def align_and_combine_datasets(dataframes):
    """
    Align and combine multiple datasets based on matching column names.
    
    Parameters:
    dataframes (list): List of pandas DataFrames to combine
    
    Returns:
    pandas.DataFrame: Combined dataset with aligned columns
    """
    if not dataframes:
        return pd.DataFrame()
    
    # Standardize column names for all dataframes
    for i, df in enumerate(dataframes):
        df.columns = standardize_column_names(df)
        dataframes[i] = df
    
    # Find common columns across all dataframes
    common_cols = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_cols = common_cols.intersection(set(df.columns))
    
    # Ensure required columns are present
    required_cols = {"unique id", "completed", "client", "client's provider", "team"}
    if not required_cols.issubset(common_cols):
        missing = required_cols - common_cols
        raise ValueError(f"Missing required columns across datasets: {missing}")
    
    # Combine datasets using only common columns
    combined_df = pd.concat(
        [df[list(common_cols)] for df in dataframes],
        ignore_index=True
    )
    
    # Process combined dataset
    combined_df["completed"] = pd.to_datetime(
        combined_df["completed"].str.replace("EST", "").str.strip(), 
        errors="coerce"
    )
    
    # Convert numeric columns
    for col in combined_df.columns:
        if col not in ["team", "unique id", "client", "client's provider", "completed"]:
            combined_df[col] = combined_df[col].apply(extract_numeric_value)
    
    # Remove rows with missing required data
    combined_df = combined_df.dropna(subset=["unique id", "completed"])
    
    # Sort by date
    combined_df = combined_df.sort_values("completed")
    
    return combined_df

def preprocess_data(df_list):
    """
    Preprocess multiple datasets and combine them.
    
    Parameters:
    df_list (list): List of pandas DataFrames to process
    
    Returns:
    pandas.DataFrame: Processed and combined dataset
    """
    try:
        return align_and_combine_datasets(df_list)
    except Exception as e:
        raise Exception(f"Error preprocessing data: {str(e)}")'''