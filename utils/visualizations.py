import streamlit as st
import plotly.express as px
import numpy as np
from scipy import stats
import pandas as pd

def create_analysis_plot(df, col, time_col, title):
    try:
        # Drop missing values and ensure numeric conversion
        df = df.dropna(subset=[col, time_col])
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

        # Check if there are enough unique data points for regression
        df = df.sort_values(time_col)
        days = (df[time_col] - df[time_col].min()).dt.days
        values = df[col].values

        if len(days) < 2 or len(values) < 2 or len(set(days)) < 2:
            st.warning(f"Insufficient unique data points for {col}.")
            return None  # Not enough data to perform regression

        # Calculate the trend line
        slope, intercept, _, _, _ = stats.linregress(days, values)
        trend_line = slope * days + intercept

        # Create scatter plot with trend line
        fig = px.scatter(df, x=time_col, y=col, title=title)
        fig.add_scatter(x=df[time_col], y=trend_line, name='Trend', mode='lines')

        return fig

    except ValueError as ve:
        st.warning(f"Error generating plot for {col}: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred while plotting {col}: {e}")
    
    return None  # Return None to indicate failure
