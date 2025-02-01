import pandas as pd
import numpy as np
from scipy import stats

def calculate_metrics(df, unique_id_col, time_col):
    """
    Calculate aggregated metrics across all clients in the filtered dataset.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    unique_id_col (str): Column name for unique identifier
    time_col (str): Column name for time/date values
    
    Returns:
    pd.DataFrame: Aggregated metrics
    """
    if df.empty:
        return pd.DataFrame()
        
    if unique_id_col not in df.columns:
        raise KeyError(f"Required column '{unique_id_col}' not found in DataFrame")
    
    if time_col not in df.columns:
        raise KeyError(f"Required column '{time_col}' not found in DataFrame")
        

    df[time_col] = pd.to_datetime(df[time_col])
    
    # excluding the ID colum
    numeric_columns = df.select_dtypes(include=['number']).columns
    numeric_columns = [col for col in numeric_columns if col != unique_id_col]
    

    metrics = {
        'Total Clients': len(df[unique_id_col].unique()),
        'Total Measurements': len(df),
        'Average Measurements per Client': len(df) / len(df[unique_id_col].unique()),
        'Date Range': f"{df[time_col].min().strftime('%Y-%m-%d')} to {df[time_col].max().strftime('%Y-%m-%d')}",
        'Average Time Delta (days)': (df.groupby(unique_id_col)[time_col].agg(lambda x: (x.max() - x.min()).days)).mean()
    }
    
    # Calculate aggregated metrics for each numeric column
    for col in numeric_columns:
        # Calculate metrics for each client first
        client_values = []
        for client_id in df[unique_id_col].unique():
            client_data = df[df[unique_id_col] == client_id].sort_values(by=time_col)
            if not client_data.empty:
                first_valid = client_data[col].first_valid_index()
                last_valid = client_data[col].last_valid_index()
                if first_valid is not None and last_valid is not None:
                    start_val = client_data.loc[first_valid, col]
                    end_val = client_data.loc[last_valid, col]
                    client_values.append({
                        'start': start_val,
                        'end': end_val,
                        'change': end_val - start_val,
                        'days': (client_data[time_col].max() - client_data[time_col].min()).days,
                        'last_3': client_data[col].tail(3).mean()
                    })
        
        if client_values:
            # Calculate averages across all clients
            avg_start = np.mean([c['start'] for c in client_values])
            avg_end = np.mean([c['end'] for c in client_values])
            avg_change = np.mean([c['change'] for c in client_values])
            avg_change_per_day = np.mean([c['change']/c['days'] if c['days'] > 0 else 0 for c in client_values])
            avg_last_3 = np.mean([c['last_3'] for c in client_values])
            
            metrics.update({
                f"{col} Average Start Value": avg_start,
                f"{col} Average End Value": avg_end,
                f"{col} Average Total Change": avg_change,
                f"{col} Average Change per Day": avg_change_per_day,
                f"{col} Average of Last 3 Measurements": avg_last_3
            })
    
    # Convert to DataFrame
    df_metrics = pd.DataFrame([metrics])
    
    # Format column names
    df_metrics.columns = df_metrics.columns.astype(str).str.replace("_", " ").str.title()
    
    return df_metrics