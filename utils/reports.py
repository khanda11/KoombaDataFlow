import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

def generate_summary_report(df, output_file=None):
    """
    Generate a comprehensive HTML report with statistics and visualizations.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with client data
    output_file (str): Optional output file path for the report
    """
    # Initialize HTML content
    html_content = []
    
    # Add report header
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
            .trend-positive { color: green; }
            .trend-negative { color: red; }
            .trend-neutral { color: gray; }
        </style>
    </head>
    <body>
    """)
    
    # Add report title and date
    html_content.append(f"""
    <div class="header">
        <h1>Client Progress Report</h1>
        <p>Generated on {datetime.now().strftime('%B %d, %Y')}</p>
    </div>
    """)
    
    # Overall Statistics Section
    html_content.append("""
    <div class="section">
        <h2>Overall Statistics</h2>
    """)
    
    # Calculate overall metrics
    total_clients = len(df['Unique ID'].unique())
    total_measurements = len(df)
    avg_measurements = total_measurements / total_clients
    date_range = f"{df['Completed'].min().strftime('%B %d, %Y')} to {df['Completed'].max().strftime('%B %d, %Y')}"
    
    html_content.append(f"""
        <div class="metric-card">
            <p><strong>Total Clients:</strong> {total_clients}</p>
            <p><strong>Total Measurements:</strong> {total_measurements}</p>
            <p><strong>Average Measurements per Client:</strong> {avg_measurements:.1f}</p>
            <p><strong>Date Range:</strong> {date_range}</p>
        </div>
    """)
    
    # Progress Metrics Section
    html_content.append("""
    <div class="section">
        <h2>Progress Metrics</h2>
    """)
    
    # Process each numeric column
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_cols = [col for col in numeric_cols if col != 'Unique ID']
    
    for col in numeric_cols:
        # Calculate metrics per client
        client_metrics = []
        for client_id in df['Unique ID'].unique():
            client_data = df[df['Unique ID'] == client_id].sort_values('Completed')
            if len(client_data) >= 2:
                start_val = client_data.iloc[0][col]
                end_val = client_data.iloc[-1][col]
                change = end_val - start_val
                client_metrics.append({
                    'start': start_val,
                    'end': end_val,
                    'change': change
                })
        
        if client_metrics:
            avg_start = np.mean([m['start'] for m in client_metrics])
            avg_end = np.mean([m['end'] for m in client_metrics])
            avg_change = np.mean([m['change'] for m in client_metrics])
            
            # Create trend indicator
            trend_class = 'trend-positive' if avg_change < 0 else 'trend-negative' if avg_change > 0 else 'trend-neutral'
            trend_symbol = '↓' if avg_change < 0 else '↑' if avg_change > 0 else '→'
            
            html_content.append(f"""
            <div class="metric-card">
                <h3>{col}</h3>
                <p><strong>Average Start Value:</strong> {avg_start:.2f}</p>
                <p><strong>Average End Value:</strong> {avg_end:.2f}</p>
                <p><strong>Average Change:</strong> <span class="{trend_class}">{avg_change:.2f} {trend_symbol}</span></p>
            </div>
            """)
            
            # Create visualization
            fig = go.Figure()
            
            # Add individual client lines (light gray)
            for client_id in df['Unique ID'].unique():
                client_data = df[df['Unique ID'] == client_id].sort_values('Completed')
                fig.add_trace(go.Scatter(
                    x=client_data['Completed'],
                    y=client_data[col],
                    mode='lines',
                    line=dict(color='lightgray'),
                    showlegend=False
                ))
            
            # Add average trend line
            avg_by_date = df.groupby('Completed')[col].mean()
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
    
    # Close HTML
    html_content.append("""
    </div>
    </body>
    </html>
    """)
    
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
    report_html = generate_summary_report(df, output_file)
    return report_html