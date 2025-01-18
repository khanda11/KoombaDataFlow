import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy import stats
import re
from datetime import datetime

# Set up the Streamlit app
st.title("Koomba Data Analysis")
st.write("Upload your CSV files to analyze client progress over time.")

# Analysis type selector
analysis_type = st.radio(
    "Select Analysis Type",
    ["Before and After Test", "Data Summary"],
    horizontal=True
)

# Upload CSV file(s)
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

# Define required columns for validation
REQUIRED_COLUMNS = ["Unique ID", "Completed", "Client", "Client's Provider"]

def extract_numeric_value(s):
    """Extract numeric value from a string that may contain letters."""
    if isinstance(s, str):
        match = re.search(r"([-+]?\d+(\.\d+)?)", s)
        return float(match.group(1)) if match else None
    return s if isinstance(s, (int, float)) else None

def preprocess_data(df):
    """Preprocess the dataframe for analysis."""
    # Standardize column names
    df.columns = df.columns.str.strip().str.replace("\xa0", "")
    
    # Convert timestamps
    df["Completed"] = pd.to_datetime(df["Completed"].str.replace("EST", "").str.strip(), errors="coerce")
    
    # Process numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in df.columns:
        if col not in numeric_cols and col not in ["Unique ID", "Client", "Client's Provider", "Completed"]:
            df[col] = df[col].apply(extract_numeric_value)
            
    return df.dropna(subset=["Unique ID", "Completed"])

def calculate_group_metrics(df, col):
    """Calculate group-level metrics including start and end averages."""
    values = pd.to_numeric(df[col], errors='coerce')
    if len(values) == 0:
        return None
        
    metrics = {
        'current_avg': values.mean(),
        'min': values.min(),
        'max': values.max(),
    }
    
    # If we have multiple timepoints, calculate change metrics
    if len(df['Completed'].unique()) > 1:
        # Sort by date to get proper start/end values
        df_sorted = df.sort_values('Completed')
        
        # Calculate first and last month averages for each client
        client_start_ends = []
        for _, client_data in df_sorted.groupby('Unique ID'):
            client_data = client_data.sort_values('Completed')
            first_value = client_data[col].iloc[0]
            last_value = client_data[col].iloc[-1]
            client_start_ends.append({'start': first_value, 'end': last_value})
        
        # Calculate averages across all clients
        start_values = [item['start'] for item in client_start_ends]
        end_values = [item['end'] for item in client_start_ends]
        
        metrics.update({
            'start_avg': np.mean(start_values),
            'end_avg': np.mean(end_values),
            'has_change_data': True
        })
    else:
        metrics['has_change_data'] = False
        
    return metrics

def calculate_metrics(df, unique_id_col, time_col):
    """Calculate comprehensive metrics for each client."""
    df = df.sort_values(by=[unique_id_col, time_col])
    numeric_columns = df.select_dtypes(include=['number']).columns
    numeric_columns = [col for col in numeric_columns if col not in [unique_id_col, 'Client Phone Number']]
    
    results = []
    
    for unique_id, group in df.groupby(unique_id_col):
        group = group.sort_values(time_col)
        time_delta = (group[time_col].max() - group[time_col].min()).days
        
        metrics = {
            'Unique ID': unique_id,
            'Client': group['Client'].iloc[0],
            "Client's Provider": group["Client's Provider"].iloc[0],
            'Time Delta (days)': time_delta,
            'Measurements': len(group)
        }
        
        for col in numeric_columns:
            values = pd.to_numeric(group[col], errors='coerce')
            if len(values.dropna()) < 1:
                continue
                
            # Calculate metrics
            if len(values.dropna()) >= 2:
                total_change = values.iloc[-1] - values.iloc[0]
                days = (group[time_col] - group[time_col].min()).dt.days
                slope, _, _, _, _ = stats.linregress(days, values)
                correlation_matrix = np.corrcoef(days, values)
                r_squared = correlation_matrix[0, 1] ** 2 if correlation_matrix.size > 1 else 0
                last_3_avg = values.tail(3).mean()
                
                metrics.update({
                    f"{col}_total_change": f"{total_change:.2f} ({values.iloc[0]:.2f} → {values.iloc[-1]:.2f})",
                    f"{col}_change_per_day": f"{slope:.3f}",
                    f"{col}_consistency": f"{r_squared:.2f}",
                    f"{col}_last_3_avg": f"{last_3_avg:.2f}",
                    f"{col}_has_change_data": True,
                    f"{col}_start": values.iloc[0],
                    f"{col}_end": values.iloc[-1]
                })
            else:
                # For single timepoint data, just record the value
                metrics.update({
                    f"{col}_current_value": f"{values.iloc[0]:.2f}",
                    f"{col}_has_change_data": False
                })
            
        results.append(metrics)
    
    return pd.DataFrame(results)

def create_analysis_plot(df, col, time_col, title):
    """Create a simplified analysis plot with just the trend line."""
    if len(df) < 2:
        return None
        
    fig = px.scatter(df, x=time_col, y=col, title=title)
    
    # Add trend line
    days = (df[time_col] - df[time_col].min()).dt.days
    values = df[col]
    slope, intercept, _, _, _ = stats.linregress(days, values)
    trend_line = slope * days + intercept
    fig.add_scatter(x=df[time_col], y=trend_line, 
                   name='Trend', mode='lines')
    
    return fig

def create_group_average_plot(df, col, time_col, group_col, title):
    """Create a plot showing average values over time for different groups."""
    df['date'] = df[time_col].dt.date
    avg_data = df.groupby(['date', group_col])[col].mean().reset_index()
    
    fig = px.line(avg_data, x='date', y=col, color=group_col,
                  title=title, labels={col: f'Average {col}', 'date': 'Date'})
    return fig

def calculate_summary_statistics(df):
    """Calculate summary statistics for the data."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_cols = [col for col in numeric_cols if col not in ["Unique ID", "Client Phone Number"]]
    
    summary_stats = {}
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors='coerce')
        summary_stats[col] = {
            'Mean': values.mean(),
            'Median': values.median(),
            'Std Dev': values.std(),
            'Min': values.min(),
            'Max': values.max(),
            'Count': values.count()
        }
    return summary_stats

def generate_client_report(metrics_df, client_data):
    """Generate a detailed report for a single client."""
    client_metrics = metrics_df.iloc[0]
    
    # Basic client information
    report_data = {
        'Client': [client_metrics['Client']],
        'Provider': [client_metrics["Client's Provider"]],
        'Time Period (days)': [client_metrics['Time Delta (days)']],
        'Number of Measurements': [client_metrics['Measurements']]
    }
    
    # Add metrics for each measurement type
    numeric_cols = client_data.select_dtypes(include=['number']).columns
    numeric_cols = [col for col in numeric_cols if col not in ["Unique ID", "Client Phone Number"]]
    
    for col in numeric_cols:
        if f"{col}_total_change" in client_metrics:
            report_data.update({
                f"{col} Start Value": [client_metrics[f'{col}_start']],
                f"{col} End Value": [client_metrics[f'{col}_end']],
                f"{col} Total Change": [float(client_metrics[f'{col}_total_change'].split()[0])],
                f"{col} Change per Day": [float(client_metrics[f'{col}_change_per_day'])],
                f"{col} Consistency Score": [float(client_metrics[f'{col}_consistency'])],
                f"{col} Last 3 Average": [float(client_metrics[f'{col}_last_3_avg'])]
            })
    
    return pd.DataFrame(report_data)

def generate_group_report(metrics_df, filtered_data):
    """Generate a summary report for a group of clients."""
    numeric_cols = filtered_data.select_dtypes(include=['number']).columns
    numeric_cols = [col for col in numeric_cols if col not in ["Unique ID", "Client Phone Number"]]
    
    report_data = []
    
    for col in numeric_cols:
        group_metrics = calculate_group_metrics(filtered_data, col)
        if group_metrics and group_metrics['has_change_data']:
            avg_change = pd.to_numeric(metrics_df[f"{col}_change_per_day"], errors='coerce').mean()
            avg_consistency = pd.to_numeric(metrics_df[f"{col}_consistency"], errors='coerce').mean()
            
            report_data.append({
                'Metric': col,
                'Starting Average': group_metrics['start_avg'],
                'Ending Average': group_metrics['end_avg'],
                'Average Change per Day': avg_change,
                'Average Consistency Score': avg_consistency,
                'Minimum Value': group_metrics['min'],
                'Maximum Value': group_metrics['max']
            })
    
    return pd.DataFrame(report_data)

def generate_summary_report(filtered_data, provider=None):
    """Generate a summary statistics report."""
    summary_stats = calculate_summary_statistics(filtered_data)
    
    report_data = []
    for metric, stats in summary_stats.items():
        row = {
            'Metric': metric,
            'Mean': stats['Mean'],
            'Median': stats['Median'],
            'Standard Deviation': stats['Std Dev'],
            'Minimum': stats['Min'],
            'Maximum': stats['Max'],
            'Count': stats['Count']
        }
        report_data.append(row)
    
    # Add additional summary information
    client_count = filtered_data["Client"].nunique()
    measurement_count = len(filtered_data)
    time_span = (filtered_data["Completed"].max() - filtered_data["Completed"].min()).days
    
    summary_row = {
        'Metric': 'Summary Statistics',
        'Mean': f'Total Clients: {client_count}',
        'Median': f'Total Measurements: {measurement_count}',
        'Standard Deviation': f'Avg Measurements per Client: {measurement_count/client_count:.1f}',
        'Minimum': f'Time Span (days): {time_span}',
        'Maximum': provider if provider else 'All Providers',
        'Count': filtered_data["Client's Provider"].nunique()
    }
    report_data.append(summary_row)
    
    return pd.DataFrame(report_data)

# Main app logic
if uploaded_files:
    # Process all uploaded files
    all_data = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file, delimiter=";")
            
            # Validate required columns
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                st.error(f"File {file.name} is missing columns: {missing_cols}")
                continue
                
            df = preprocess_data(df)
            all_data.append(df)
            st.success(f"Successfully processed {file.name}")
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            continue
    
    if all_data:
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Create sidebar filters
        st.sidebar.header("Filters")
        
        # Date range filter
        min_date = pd.to_datetime(combined_data["Completed"].min())
        max_date = pd.to_datetime(combined_data["Completed"].max())
        date_range = st.sidebar.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        # Provider filter
        providers = ["All"] + sorted(combined_data["Client's Provider"].unique().tolist())
        selected_provider = st.sidebar.selectbox("Select Provider", providers)
        
        # Client filter
        clients = ["All"] + sorted(combined_data["Client"].unique().tolist())
        selected_client = st.sidebar.selectbox("Select Client", clients)
        
        # Apply filters
        filtered_data = combined_data.copy()
        
        if len(date_range) == 2:
            filtered_data = filtered_data[
                (filtered_data["Completed"].dt.date >= date_range[0]) &
                (filtered_data["Completed"].dt.date <= date_range[1])
            ]
            
        if selected_provider != "All":
            filtered_data = filtered_data[filtered_data["Client's Provider"] == selected_provider]
            
        if selected_client != "All":
            filtered_data = filtered_data[filtered_data["Client"] == selected_client]

        # Generate timestamp for file names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if analysis_type == "Before and After Test":
            # Calculate metrics
            metrics_df = calculate_metrics(filtered_data, "Unique ID", "Completed")
            
            # Display results
            st.header("Analysis Results")
            
            if selected_client != "All":
                st.subheader(f"Detailed Analysis for {selected_client}")
                
                # Generate and display client report
                client_report = generate_client_report(
                    metrics_df[metrics_df["Client"] == selected_client], 
                    filtered_data
                )
                
                # Add download button for client report
                st.download_button(
                    label="Download Client Report",
                    data=client_report.to_csv(index=False).encode('utf-8'),
                    file_name=f"client_report_{selected_client}_{timestamp}.csv",
                    mime="text/csv"
                )
                
                # Display metrics
                client_metrics = metrics_df[metrics_df["Client"] == selected_client].iloc[0]
                st.write("Summary Metrics:")
                st.write(f"- Time Period: {client_metrics['Time Delta (days)']} days")
                st.write(f"- Number of Measurements: {client_metrics['Measurements']}")
                
                # Create plots for numeric columns
                numeric_cols = filtered_data.select_dtypes(include=['number']).columns
                numeric_cols = [col for col in numeric_cols if col not in ["Unique ID", "Client Phone Number"]]
                
                for col in numeric_cols:
                    if f"{col}_total_change" in client_metrics:
                        st.write(f"\n{col}:")
                        st.write(f"- Total Change: {client_metrics[f'{col}_total_change']}")
                        st.write(f"- Change per Day: {client_metrics[f'{col}_change_per_day']}")
                        st.write(f"- Consistency Score: {client_metrics[f'{col}_consistency']}")
                        st.write(f"- Average of Last 3 Measurements: {client_metrics[f'{col}_last_3_avg']}")
                        
                        fig = create_analysis_plot(
                            filtered_data, 
                            col, 
                            "Completed", 
                            f"{col} Progress Over Time"
                        )
                        if fig:
                            st.plotly_chart(fig)
            
            else:
                st.subheader("Overview of All Clients")
                st.dataframe(metrics_df)
                
                # Add download button for all clients report
                st.download_button(
                    label="Download Complete Client Analysis",
                    data=metrics_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"all_clients_analysis_{timestamp}.csv",
                    mime="text/csv"
                )
                
                # Show average metrics and visualizations
                st.subheader("Average Metrics Across All Filtered Clients")
                
                # Generate and display group report
                group_report = generate_group_report(metrics_df, filtered_data)
                
                # Add download button for group report
                st.download_button(
                    label="Download Group Summary Report",
                    data=group_report.to_csv(index=False).encode('utf-8'),
                    file_name=f"group_summary_{timestamp}.csv",
                    mime="text/csv"
                )
                
                numeric_cols = filtered_data.select_dtypes(include=['number']).columns
                numeric_cols = [col for col in numeric_cols if col not in ["Unique ID", "Client Phone Number"]]
                
                for col in numeric_cols:
                    st.write(f"\n### {col}")
                    
                    # Calculate group metrics
                    group_metrics = calculate_group_metrics(filtered_data, col)
                    
                    if group_metrics:
                        if group_metrics['has_change_data']:
                            avg_change_col = f"{col}_change_per_day"
                            if avg_change_col in metrics_df.columns:
                                avg_change = pd.to_numeric(metrics_df[avg_change_col], errors='coerce').mean()
                                avg_consistency = pd.to_numeric(metrics_df[f"{col}_consistency"], errors='coerce').mean()
                                
                                st.write(f"Average Change Stats:")
                                st.write(f"* Starting Average: {group_metrics['start_avg']:.2f}")
                                st.write(f"* Ending Average: {group_metrics['end_avg']:.2f}")
                                st.write(f"* Average Change per Day: {avg_change:.3f}")
                                st.write(f"* Average Consistency Score: {avg_consistency:.2f}")
                        else:
                            st.write(f"Current Statistics:")
                            st.write(f"* Current Average: {group_metrics['current_avg']:.2f}")
                            st.write(f"* Range: {group_metrics['min']:.2f} to {group_metrics['max']:.2f}")
                        
                        # Add group average visualization
                        if selected_provider == "All" and group_metrics['has_change_data']:
                            fig = create_group_average_plot(
                                filtered_data,
                                col,
                                "Completed",
                                "Client's Provider",
                                f"Average {col} by Provider Over Time"
                            )
                            st.plotly_chart(fig)
        
        else:  # Data Summary view
            st.header("Data Summary")
            
            # Calculate and display summary statistics
            summary_stats = calculate_summary_statistics(filtered_data)
            
            # Display summary by provider if no specific provider is selected
            if selected_provider == "All":
                st.subheader("Summary by Provider")
                
                all_provider_summaries = []
                
                for provider in filtered_data["Client's Provider"].unique():
                    provider_data = filtered_data[filtered_data["Client's Provider"] == provider]
                    provider_report = generate_summary_report(provider_data, provider)
                    all_provider_summaries.append(provider_report)
                    
                    st.write(f"\n### {provider}")
                    st.dataframe(provider_report)
                    
                    # Download button for each provider's report
                    st.download_button(
                        label=f"Download {provider} Summary Report",
                        data=provider_report.to_csv(index=False).encode('utf-8'),
                        file_name=f"summary_report_{provider}_{timestamp}.csv",
                        mime="text/csv"
                    )
                
                # Combine all provider summaries
                combined_summary = pd.concat(all_provider_summaries)
                
                # Download button for complete summary report
                st.download_button(
                    label="Download Complete Summary Report",
                    data=combined_summary.to_csv(index=False).encode('utf-8'),
                    file_name=f"complete_summary_report_{timestamp}.csv",
                    mime="text/csv"
                )
            
            else:
                # Display summary for selected provider
                st.subheader(f"Summary for {selected_provider}")
                provider_report = generate_summary_report(filtered_data, selected_provider)
                st.dataframe(provider_report)
                
                # Download button for provider report
                st.download_button(
                    label=f"Download {selected_provider} Summary Report",
                    data=provider_report.to_csv(index=False).encode('utf-8'),
                    file_name=f"summary_report_{selected_provider}_{timestamp}.csv",
                    mime="text/csv"
                )

else:
    st.info("Please upload CSV files to begin analysis.")