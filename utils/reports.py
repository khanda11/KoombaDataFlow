import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

def is_numeric_column(df, column):
    """
    Check if a column contains numeric data by attempting to convert to float
    and checking if any valid numbers exist.
    """
    try:
        numeric_series = pd.to_numeric(df[column], errors='coerce')
        return numeric_series.notna().any()
    except:
        return False

def get_analyzable_columns(df):
    """
    Get list of columns that contain valid numeric data for analysis.
    """
    excluded_columns = ['Unique ID', 'Client Email', 'Client Phone Number', 
                       'First', 'Last', 'Email', 'Class Year', 'Completed']
    
    analyzable_cols = []
    for col in df.columns:
        if col not in excluded_columns and is_numeric_column(df, col):
            analyzable_cols.append(col)
    
    return analyzable_cols

def generate_time_trend_report(df, output_file=None):
    """
    Generate a comprehensive HTML report focusing on time trends and visualizations.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with client data
    output_file (str): Optional output file path for the report
    """
    # Initialize HTML content
    html_content = []
    
    # Add report header with CSS
    html_content.append("""
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { text-align: center; margin-bottom: 30px; }
            .section { margin-bottom: 40px; }
            .metric-card { 
                border: 1px solid #ddd; 
                padding: 20px; 
                margin: 10px 0; 
                border-radius: 5px;
            }
            .trend-positive { color: green; }
            .trend-negative { color: red; }
            .trend-neutral { color: gray; }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .stat-box {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
            }
            .stat-label {
                font-weight: bold;
                margin-bottom: 5px;
            }
            .stat-value {
                font-size: 1.2em;
            }
        </style>
    </head>
    <body>
    """)
    
    # Add report title
    html_content.append(f"""
    <div class="header">
        <h1>Time Trend Analysis Report</h1>
        <p>Generated on {datetime.now().strftime('%B %d, %Y')}</p>
    </div>
    """)
    
    # Date range info
    date_range = f"{df['Completed'].min().strftime('%B %d, %Y')} to {df['Completed'].max().strftime('%B %d, %Y')}"
    html_content.append(f"""
    <div class="metric-card">
        <p><strong>Analysis Period:</strong> {date_range}</p>
        <p><strong>Total Clients:</strong> {len(df['Unique ID'].unique())}</p>
    </div>
    """)
    
    # Get analyzable columns
    numeric_cols = get_analyzable_columns(df)
    
    if not numeric_cols:
        html_content.append("""
        <div class="metric-card">
            <p><strong>No numeric data found for analysis.</strong></p>
            <p>The provided dataset does not contain any questions with numeric responses that can be analyzed.</p>
        </div>
        """)
    else:
        for col in numeric_cols:
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            if numeric_data.notna().sum() < 2:  # Skip if less than 2 valid numbers
                continue
                
            html_content.append(f"<div class='section'><h2>{col} Analysis</h2>")
            
            # Single client or multiple clients logic
            is_single_client = len(df['Unique ID'].unique()) == 1
            
            if is_single_client:
                # Single client analysis
                client_data = df.sort_values('Completed')
                numeric_data = pd.to_numeric(client_data[col], errors='coerce')
                valid_data = client_data[numeric_data.notna()]
                
                if len(valid_data) >= 2:
                    start_val = float(valid_data.iloc[0][col])
                    end_val = float(valid_data.iloc[-1][col])
                    change = end_val - start_val
                    change_pct = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
                    
                    trend_class = 'trend-positive' if change < 0 else 'trend-negative' if change > 0 else 'trend-neutral'
                    trend_symbol = '↓' if change < 0 else '↑' if change > 0 else '→'
                    
                    html_content.append(f"""
                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="stat-label">Start Value</div>
                            <div class="stat-value">{start_val:.2f}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">End Value</div>
                            <div class="stat-value">{end_val:.2f}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Absolute Change</div>
                            <div class="stat-value {trend_class}">{change:.2f} {trend_symbol}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Percent Change</div>
                            <div class="stat-value {trend_class}">{change_pct:.1f}% {trend_symbol}</div>
                        </div>
                    </div>
                    """)
            else:
                # Multiple clients analysis
                client_metrics = []
                for client_id in df['Unique ID'].unique():
                    client_data = df[df['Unique ID'] == client_id].sort_values('Completed')
                    numeric_data = pd.to_numeric(client_data[col], errors='coerce')
                    valid_data = client_data[numeric_data.notna()]
                    
                    if len(valid_data) >= 2:
                        start_val = float(valid_data.iloc[0][col])
                        end_val = float(valid_data.iloc[-1][col])
                        change = end_val - start_val
                        change_pct = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
                        client_metrics.append({
                            'start': start_val,
                            'end': end_val,
                            'change': change,
                            'change_pct': change_pct
                        })
                
                if client_metrics:
                    avg_start = np.mean([m['start'] for m in client_metrics])
                    avg_end = np.mean([m['end'] for m in client_metrics])
                    avg_change = np.mean([m['change'] for m in client_metrics])
                    avg_change_pct = np.mean([m['change_pct'] for m in client_metrics])
                    
                    trend_class = 'trend-positive' if avg_change < 0 else 'trend-negative' if avg_change > 0 else 'trend-neutral'
                    trend_symbol = '↓' if avg_change < 0 else '↑' if avg_change > 0 else '→'
                    
                    html_content.append(f"""
                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="stat-label">Average Start Value</div>
                            <div class="stat-value">{avg_start:.2f}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Average End Value</div>
                            <div class="stat-value">{avg_end:.2f}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Average Absolute Change</div>
                            <div class="stat-value {trend_class}">{avg_change:.2f} {trend_symbol}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Average Percent Change</div>
                            <div class="stat-value {trend_class}">{avg_change_pct:.1f}% {trend_symbol}</div>
                        </div>
                    </div>
                    """)
            
            # Create visualization
            fig = go.Figure()
            
            # Add individual client lines (light gray)
            for client_id in df['Unique ID'].unique():
                client_data = df[df['Unique ID'] == client_id].sort_values('Completed')
                numeric_data = pd.to_numeric(client_data[col], errors='coerce')
                valid_data = client_data[numeric_data.notna()]
                
                if len(valid_data) >= 2:
                    fig.add_trace(go.Scatter(
                        x=valid_data['Completed'],
                        y=numeric_data[numeric_data.notna()],
                        mode='lines+markers',
                        line=dict(color='lightgray'),
                        name=f'Client {client_id}' if is_single_client else None,
                        showlegend=is_single_client
                    ))
            
            if not is_single_client:
                # Add average trend line for multiple clients
                avg_by_date = df.groupby('Completed')[col].apply(
                    lambda x: pd.to_numeric(x, errors='coerce').mean()
                ).dropna()
                
                if len(avg_by_date) >= 2:
                    fig.add_trace(go.Scatter(
                        x=avg_by_date.index,
                        y=avg_by_date.values,
                        mode='lines+markers',
                        name='Average',
                        line=dict(color='blue', width=3)
                    ))
            
            fig.update_layout(
                title=f'{col} Progress Over Time',
                xaxis_title='Date',
                yaxis_title='Score',
                height=400
            )
            
            html_content.append(fig.to_html(full_html=False))
            html_content.append("</div>")
    
    # Close HTML
    html_content.append("</body></html>")
    
    # Combine all HTML content
    full_html = "\n".join(html_content)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_html)
    
    return full_html

def generate_summary_report(df, output_file=None):
    """
    Generate a comprehensive HTML report focusing on summary statistics.
    """
    html_content = []
    
    # Add report header with CSS
    html_content.append("""
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { text-align: center; margin-bottom: 30px; }
            .section { margin-bottom: 40px; }
            .metric-card { 
                border: 1px solid #ddd; 
                padding: 20px; 
                margin: 10px 0; 
                border-radius: 5px;
            }
            table { 
                width: 100%; 
                border-collapse: collapse; 
                margin: 20px 0; 
            }
            th, td { 
                padding: 12px; 
                text-align: left; 
                border-bottom: 1px solid #ddd; 
            }
            th { background-color: #f8f9fa; }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .stat-box {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
            }
            .stat-label {
                font-weight: bold;
                margin-bottom: 5px;
            }
            .stat-value {
                font-size: 1.2em;
            }
        </style>
    </head>
    <body>
    """)
    
    # Add report title
    html_content.append(f"""
    <div class="header">
        <h1>Summary Statistics Report</h1>
        <p>Generated on {datetime.now().strftime('%B %d, %Y')}</p>
    </div>
    """)
    
    # Overall Statistics Section
    total_clients = len(df['Unique ID'].unique())  # Updated to use correct case
    total_measurements = len(df)
    measurements_per_client = df.groupby('Unique ID').size()  # Updated to use correct case
    
    html_content.append("""
    <div class="section">
        <h2>Overall Statistics</h2>
        <div class="stats-grid">""")
    
    # Basic statistics using stat-box layout
    html_content.append(f"""
            <div class="stat-box">
                <div class="stat-label">Total Clients</div>
                <div class="stat-value">{total_clients}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total Measurements</div>
                <div class="stat-value">{total_measurements}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Avg Measurements/Client</div>
                <div class="stat-value">{measurements_per_client.mean():.1f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Median Measurements/Client</div>
                <div class="stat-value">{measurements_per_client.median():.1f}</div>
            </div>
        </div>
        <div class="metric-card">
            <p><strong>Analysis Period:</strong> {df['Completed'].min().strftime('%B %d, %Y')} to {df['Completed'].max().strftime('%B %d, %Y')}</p>
        </div>
    </div>
    """)
    
    # Get analyzable columns
    numeric_cols = get_analyzable_columns(df)
    
    if numeric_cols:
        html_content.append("""
        <div class="section">
            <h2>Metric Summaries</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Valid Responses</th>
                </tr>
        """)
        
        for col in numeric_cols:
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            stats = numeric_data.describe()
            valid_count = numeric_data.notna().sum()
            html_content.append(f"""
                <tr>
                    <td>{col}</td>
                    <td>{stats['mean']:.2f}</td>
                    <td>{stats['50%']:.2f}</td>
                    <td>{stats['std']:.2f}</td>
                    <td>{stats['min']:.2f}</td>
                    <td>{stats['max']:.2f}</td>
                    <td>{valid_count}</td>
                </tr>
            """)
        
        html_content.append("</table></div>")
    else:
        html_content.append("""
        <div class="metric-card">
            <p><strong>No numeric data found for analysis.</strong></p>
            <p>The provided dataset does not contain any questions with numeric responses that can be analyzed.</p>
        </div>
        """)
    
    # Distribution plots for each metric
    for col in numeric_cols:
        numeric_data = pd.to_numeric(df[col], errors='coerce')
        data = numeric_data.dropna()
        
        if len(data) >= 2:
            fig = go.Figure()
            
            # Calculate optimal number of bins using Freedman-Diaconis rule
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            bin_width = 2 * iqr / (len(data) ** (1/3)) if iqr > 0 else 0.5
            num_bins = int((data.max() - data.min()) / bin_width) if bin_width > 0 else 30
            num_bins = min(max(10, num_bins), 50)
            
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=num_bins,
                name=col,
                showlegend=False,
                marker_color='rgb(55, 83, 109)'
            ))
            
            fig.update_layout(
                title=f'{col} Distribution',
                xaxis_title=col,
                yaxis_title='Count',
                height=400,
                showlegend=False,
                bargap=0.1,
                margin=dict(l=50, r=50, t=50, b=50),
                plot_bgcolor='white',
                xaxis=dict(
                    gridcolor='lightgrey',
                    showgrid=True
                ),
                yaxis=dict(
                    gridcolor='lightgrey',
                    showgrid=True
                )
            )
            
            html_content.append(f"<div class='section'><h2>{col} Distribution</h2>")
            html_content.append(fig.to_html(full_html=False))
            html_content.append("</div>")
    
    # Close HTML
    html_content.append("</body></html>")
    
    # Combine all HTML content
    full_html = "\n".join(html_content)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_html)
    
    return full_html

def save_report(df, output_file):
    """
    Generate and save the report to a file.
    """
    if "time_trend" in output_file.lower():
        report_html = generate_time_trend_report(df, output_file)
    else:
        report_html = generate_summary_report(df, output_file)
    return report_html
'''
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

def is_numeric_column(df, column):
    """
    Check if a column contains numeric data by attempting to convert to float
    and checking if any valid numbers exist.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to check
    
    Returns:
    bool: True if column contains valid numeric data, False otherwise
    """
    try:
        # Convert to numeric, coercing errors to NaN
        numeric_series = pd.to_numeric(df[column], errors='coerce')
        # Check if there are any non-NaN values
        return numeric_series.notna().any()
    except:
        return False

def get_analyzable_columns(df):
    """
    Get list of columns that contain valid numeric data for analysis.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    list: List of column names containing numeric data
    """
    # Columns to exclude from analysis
    excluded_columns = ['Unique ID', 'Client Email', 'Client Phone Number', 
                       'First', 'Last', 'Email', 'Class Year', 'Completed']
    
    # Check each column for numeric data
    analyzable_cols = []
    for col in df.columns:
        if col not in excluded_columns and is_numeric_column(df, col):
            analyzable_cols.append(col)
    
    return analyzable_cols

def generate_time_trend_report(df, output_file=None):
    """
    Generate a comprehensive HTML report focusing on time trends and visualizations.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with client data
    output_file (str): Optional output file path for the report
    """
    # Initialize HTML content
    html_content = []
    
    # Add report header with CSS
    html_content.append("""
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { text-align: center; margin-bottom: 30px; }
            .section { margin-bottom: 40px; }
            .metric-card { 
                border: 1px solid #ddd; 
                padding: 20px; 
                margin: 10px 0; 
                border-radius: 5px;
            }
            .trend-positive { color: green; }
            .trend-negative { color: red; }
            .trend-neutral { color: gray; }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .stat-box {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
            }
            .stat-label {
                font-weight: bold;
                margin-bottom: 5px;
            }
            .stat-value {
                font-size: 1.2em;
            }
        </style>
    </head>
    <body>
    """)
    
    # Add report title
    html_content.append(f"""
    <div class="header">
        <h1>Time Trend Analysis Report</h1>
        <p>Generated on {datetime.now().strftime('%B %d, %Y')}</p>
    </div>
    """)
    
    # Date range info
    date_range = f"{df['Completed'].min().strftime('%B %d, %Y')} to {df['Completed'].max().strftime('%B %d, %Y')}"
    html_content.append(f"""
    <div class="metric-card">
        <p><strong>Analysis Period:</strong> {date_range}</p>
        <p><strong>Total Clients:</strong> {len(df['Unique ID'].unique())}</p>
    </div>
    """)
    
    # Get analyzable columns
    numeric_cols = get_analyzable_columns(df)
    
    if not numeric_cols:
        html_content.append("""
        <div class="metric-card">
            <p><strong>No numeric data found for analysis.</strong></p>
            <p>The provided dataset does not contain any questions with numeric responses that can be analyzed.</p>
        </div>
        """)
    else:
        for col in numeric_cols:
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            if numeric_data.notna().sum() < 2:  # Skip if less than 2 valid numbers
                continue
                
            html_content.append(f"<div class='section'><h2>{col} Analysis</h2>")
            
            # Single client or multiple clients logic
            is_single_client = len(df['Unique ID'].unique()) == 1
            
            if is_single_client:
                # Single client analysis
                client_data = df.sort_values('Completed')
                numeric_data = pd.to_numeric(client_data[col], errors='coerce')
                valid_data = client_data[numeric_data.notna()]
                
                if len(valid_data) >= 2:
                    start_val = float(valid_data.iloc[0][col])
                    end_val = float(valid_data.iloc[-1][col])
                    change = end_val - start_val
                    change_pct = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
                    
                    trend_class = 'trend-positive' if change < 0 else 'trend-negative' if change > 0 else 'trend-neutral'
                    trend_symbol = '↓' if change < 0 else '↑' if change > 0 else '→'
                    
                    html_content.append(f"""
                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="stat-label">Start Value</div>
                            <div class="stat-value">{start_val:.2f}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">End Value</div>
                            <div class="stat-value">{end_val:.2f}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Absolute Change</div>
                            <div class="stat-value {trend_class}">{change:.2f} {trend_symbol}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Percent Change</div>
                            <div class="stat-value {trend_class}">{change_pct:.1f}% {trend_symbol}</div>
                        </div>
                    </div>
                    """)
            else:
                # Multiple clients analysis
                client_metrics = []
                for client_id in df['Unique ID'].unique():
                    client_data = df[df['Unique ID'] == client_id].sort_values('Completed')
                    numeric_data = pd.to_numeric(client_data[col], errors='coerce')
                    valid_data = client_data[numeric_data.notna()]
                    
                    if len(valid_data) >= 2:
                        start_val = float(valid_data.iloc[0][col])
                        end_val = float(valid_data.iloc[-1][col])
                        change = end_val - start_val
                        change_pct = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
                        client_metrics.append({
                            'start': start_val,
                            'end': end_val,
                            'change': change,
                            'change_pct': change_pct
                        })
                
                if client_metrics:
                    avg_start = np.mean([m['start'] for m in client_metrics])
                    avg_end = np.mean([m['end'] for m in client_metrics])
                    avg_change = np.mean([m['change'] for m in client_metrics])
                    avg_change_pct = np.mean([m['change_pct'] for m in client_metrics])
                    
                    trend_class = 'trend-positive' if avg_change < 0 else 'trend-negative' if avg_change > 0 else 'trend-neutral'
                    trend_symbol = '↓' if avg_change < 0 else '↑' if avg_change > 0 else '→'
                    
                    html_content.append(f"""
                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="stat-label">Average Start Value</div>
                            <div class="stat-value">{avg_start:.2f}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Average End Value</div>
                            <div class="stat-value">{avg_end:.2f}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Average Absolute Change</div>
                            <div class="stat-value {trend_class}">{avg_change:.2f} {trend_symbol}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Average Percent Change</div>
                            <div class="stat-value {trend_class}">{avg_change_pct:.1f}% {trend_symbol}</div>
                        </div>
                    </div>
                    """)
            
            # Create visualization
            fig = go.Figure()
            
            # Add individual client lines (light gray)
            for client_id in df['Unique ID'].unique():
                client_data = df[df['Unique ID'] == client_id].sort_values('Completed')
                numeric_data = pd.to_numeric(client_data[col], errors='coerce')
                valid_data = client_data[numeric_data.notna()]
                
                if len(valid_data) >= 2:
                    fig.add_trace(go.Scatter(
                        x=valid_data['Completed'],
                        y=numeric_data[numeric_data.notna()],
                        mode='lines+markers',
                        line=dict(color='lightgray'),
                        name=f'Client {client_id}' if is_single_client else None,
                        showlegend=is_single_client
                    ))
            
            if not is_single_client:
                # Add average trend line for multiple clients
                avg_by_date = df.groupby('Completed')[col].apply(
                    lambda x: pd.to_numeric(x, errors='coerce').mean()
                ).dropna()
                
                if len(avg_by_date) >= 2:
                    fig.add_trace(go.Scatter(
                        x=avg_by_date.index,
                        y=avg_by_date.values,
                        mode='lines+markers',
                        name='Average',
                        line=dict(color='blue', width=3)
                    ))
            
            fig.update_layout(
                title=f'{col} Progress Over Time',
                xaxis_title='Date',
                yaxis_title='Score',
                height=400
            )
            
            html_content.append(fig.to_html(full_html=False))
            html_content.append("</div>")
    
    # Close HTML
    html_content.append("</body></html>")
    
    # Combine all HTML content
    full_html = "\n".join(html_content)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_html)
    
    return full_html

def generate_summary_report(df, output_file=None):
    """
    Generate a comprehensive HTML report focusing on summary statistics.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with client data
    output_file (str): Optional output file path for the report
    """
    # Initialize HTML content
    html_content = []
    
    # Add report header with CSS
    html_content.append("""
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { text-align: center; margin-bottom: 30px; }
            .section { margin-bottom: 40px; }
            .metric-card { 
                border: 1px solid #ddd; 
                padding: 20px; 
                margin: 10px 0; 
                border-radius: 5px;
            }
            table { 
                width: 100%; 
                border-collapse: collapse; 
                margin: 20px 0; 
            }
            th, td { 
                padding: 12px; 
                text-align: left; 
                border-bottom: 1px solid #ddd; 
            }
            th { background-color: #f8f9fa; }
        </style>
    </head>
    <body>
    """)
    
    # Add report title
    html_content.append(f"""
    <div class="header">
        <h1>Summary Statistics Report</h1>
        <p>Generated on {datetime.now().strftime('%B %d, %Y')}</p>
    </div>
    """)
    
    # Overall Statistics Section
    total_clients = len(df['Unique ID'].unique())
    total_measurements = len(df)
    measurements_per_client = df.groupby('Unique ID').size()
    
    html_content.append("""
    <div class="section">
        <h2>Overall Statistics</h2>
        <div class="metric-card">""")
    
    # Basic statistics
    html_content.append(f"""
            <p><strong>Total Clients:</strong> {total_clients}</p>
            <p><strong>Total Measurements:</strong> {total_measurements}</p>
            <p><strong>Average Measurements per Client:</strong> {measurements_per_client.mean():.1f}</p>
            <p><strong>Median Measurements per Client:</strong> {measurements_per_client.median():.1f}</p>
            <p><strong>Date Range:</strong> {df['Completed'].min().strftime('%B %d, %Y')} to {df['Completed'].max().strftime('%B %d, %Y')}</p>
        </div>
    </div>
    """)
    
    # Get analyzable columns
    numeric_cols = get_analyzable_columns(df)
    
    if numeric_cols:
        html_content.append("""
        <div class="section">
            <h2>Metric Summaries</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
        """)
        
        for col in numeric_cols:
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            stats = numeric_data.describe()
            html_content.append(f"""
                <tr>
                    <td>{col}</td>
                    <td>{stats['mean']:.2f}</td>
                    <td>{stats['50%']:.2f}</td>
                    <td>{stats['std']:.2f}</td>
                    <td>{stats['min']:.2f}</td>
                    <td>{stats['max']:.2f}</td>
                </tr>
            """)
        
        html_content.append("</table></div>")
    else:
        html_content.append("""
        <div class="metric-card">
            <p><strong>No numeric data found for analysis.</strong></p>
            <p>The provided dataset does not contain any questions with numeric responses that can be analyzed.</p>
        </div>
        """)
    
    # Distribution plots for each metric with improved binning and layout
    for col in numeric_cols:
        numeric_data = pd.to_numeric(df[col], errors='coerce')
        data = numeric_data.dropna()
        
        if len(data) > 0:
            fig = go.Figure()
            
            # Calculate optimal number of bins using Freedman-Diaconis rule
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            bin_width = 2 * iqr / (len(data) ** (1/3)) if iqr > 0 else 0.5
            num_bins = int((data.max() - data.min()) / bin_width) if bin_width > 0 else 30
            num_bins = min(max(10, num_bins), 50)  # Keep bins between 10 and 50
            
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=num_bins,
                name=col,
                showlegend=False,
                marker_color='rgb(55, 83, 109)'
            ))
            
            fig.update_layout(
                title=f'{col} Distribution',
                xaxis_title=col,
                yaxis_title='Count',
                height=400,
                showlegend=False,
                bargap=0.1,
                margin=dict(l=50, r=50, t=50, b=50),
                plot_bgcolor='white',
                xaxis=dict(
                    gridcolor='lightgrey',
                    showgrid=True
                ),
                yaxis=dict(
                    gridcolor='lightgrey',
                    showgrid=True
                )
            )
            
            html_content.append(f"<div class='section'><h2>{col} Distribution</h2>")
            html_content.append(fig.to_html(full_html=False))
            html_content.append("</div>")
    
    # Close HTML
    html_content.append("</body></html>")
    
    # Combine all HTML content
    full_html = "\n".join(html_content)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_html)
    
    return full_html

def save_report(df, output_file):
    """
    Generate and save the report to a file.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with client data
    output_file (str): Output file path (should end with .html)
    """
    if "time_trend" in output_file.lower():
        report_html = generate_time_trend_report(df, output_file)
    else:
        report_html = generate_summary_report(df, output_file)
    return report_html



'''