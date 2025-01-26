import re
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import streamlit as st
import plotly.io as pio
import base64

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
    """Get list of columns that contain valid numeric data for analysis."""
    excluded_columns = ['Unique ID', 'Client Email', 'Client Phone Number', 
                       'First', 'Last', 'Email', 'Class Year', 'Completed']
    
    # Special column for topic exploration
    topic_column = "9. Which of these topic areas would you like to explore with Koomba’s Care Team? (Please select all that apply)"
    
    analyzable_cols = []
    for col in df.columns:
        # Keep the existing numeric column logic
        if col not in excluded_columns and is_numeric_column(df, col):
            analyzable_cols.append(col)
    
    return analyzable_cols

def display_time_trend_analysis(df):
    """Display interactive time trend analysis directly in Streamlit."""
    numeric_cols = get_analyzable_columns(df)
    
    if not numeric_cols:
        st.warning("No numeric data found for analysis.")
        return None
    
    for col in numeric_cols:
        st.subheader(f"{col} Analysis")
        
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
                
                # Create metrics display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Start Value", f"{start_val:.2f}")
                with col2:
                    st.metric("End Value", f"{end_val:.2f}")
                with col3:
                    st.metric("Absolute Change", f"{change:.2f}", 
                             f"{'-' if change < 0 else '+' if change > 0 else ''}{abs(change):.2f}")
                with col4:
                    st.metric("Percent Change", f"{change_pct:.1f}%",
                             f"{'-' if change_pct < 0 else '+' if change_pct > 0 else ''}{abs(change_pct):.1f}%")
        
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
                
                # Create metrics display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Start Value", f"{avg_start:.2f}")
                with col2:
                    st.metric("Avg End Value", f"{avg_end:.2f}")
                with col3:
                    st.metric("Avg Absolute Change", f"{avg_change:.2f}",
                             f"{'-' if avg_change < 0 else '+' if avg_change > 0 else ''}{abs(avg_change):.2f}")
                with col4:
                    st.metric("Avg Percent Change", f"{avg_change_pct:.1f}%",
                             f"{'-' if avg_change_pct < 0 else '+' if avg_change_pct > 0 else ''}{abs(avg_change_pct):.1f}%")
        
        # Create visualization
        fig = go.Figure()
        
        # Add individual client lines
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
        
        st.plotly_chart(fig, use_container_width=True)

def display_summary_statistics(df):
    """Display interactive summary statistics directly in Streamlit."""
    # Overall Statistics Section
    st.header("Overall Statistics")
    
    total_clients = len(df['Unique ID'].unique())
    total_measurements = len(df)
    measurements_per_client = df.groupby('Unique ID').size()
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Clients", total_clients)
    with col2:
        st.metric("Total Measurements", total_measurements)
    with col3:
        st.metric("Avg Measurements/Client", f"{measurements_per_client.mean():.1f}")
    with col4:
        st.metric("Median Measurements/Client", f"{measurements_per_client.median():.1f}")
    
    st.write(f"Analysis Period: {df['Completed'].min().strftime('%B %d, %Y')} to {df['Completed'].max().strftime('%B %d, %Y')}")
    
    # Get analyzable numeric columns
    numeric_cols = get_analyzable_columns(df)
    
    # Specific handling for Topic Exploration
    topic_column = "9. Which of these topic areas would you like to explore with Koomba’s Care Team? (Please select all that apply)"
    if topic_column in df.columns:
        st.header("Topic Exploration")
        
        all_topics = []
        for topics in df[topic_column].dropna():
            split_topics = [
                topic.strip() 
                for line in str(topics).split('\n') 
                for topic in line.split(',')
            ]
            all_topics.extend([t.strip() for t in split_topics if t.strip()])
        
        if all_topics:
            topic_counts = pd.Series(all_topics).value_counts()
            total_responses = len(df[topic_column].dropna())
            topic_percentages = (topic_counts / total_responses * 100).round(2)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=topic_percentages.index, 
                    y=topic_percentages.values,
                    text=[f'{p}%' for p in topic_percentages.values],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title='Care Team Topic Exploration',
                xaxis_title='Topics',
                yaxis_title='Percentage of Participants',
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    
    if numeric_cols:
        st.header("Metric Summaries")
        
        # Create and display metrics table
        stats_data = []
        for col in numeric_cols:
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            stats = numeric_data.describe()
            valid_count = numeric_data.notna().sum()
            stats_data.append({
                "Metric": col,
                "Mean": f"{stats['mean']:.2f}",
                "Median": f"{stats['50%']:.2f}",
                "Std Dev": f"{stats['std']:.2f}",
                "Min": f"{stats['min']:.2f}",
                "Max": f"{stats['max']:.2f}",
                "Valid Responses": valid_count
            })
        
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        
        # Distribution plots
        st.header("Distribution Plots")
        for col in numeric_cols:
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            data = numeric_data.dropna()
            
            if len(data) >= 2:
                fig = go.Figure()
                
                # Calculate optimal number of bins
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
                    bargap=0.1
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric data found for analysis.")

def plot_to_html(fig):
    """Convert Plotly figure to base64 encoded image for embedding in HTML."""
    img_bytes = pio.to_image(fig, format='png', width=800, height=400)
    encoded_img = base64.b64encode(img_bytes).decode('utf-8')
    return f'<img src="data:image/png;base64,{encoded_img}" style="max-width:100%; height:auto;">'

def generate_report_header(df, analysis_type, filters):
    """Generate filter details for report header."""
    filter_details = []
    
    # Date Range Filter
    date_range = [df['Completed'].min().strftime('%B %d, %Y'), 
                  df['Completed'].max().strftime('%B %d, %Y')]
    filter_details.append(f"Date Range: {date_range[0]} to {date_range[1]}")
    
    # Add other applied filters
    if filters is not None:
        # Check each filter with a more robust approach
        filter_mapping = {
            'provider': "Client's Provider",
            'client': 'Client',
            'team': 'Team'
        }
        
        for key, column_name in filter_mapping.items():
            filter_value = filters.get(key)
            if filter_value and filter_value != 'All':
                filter_details.append(f"{column_name}: {filter_value}")
    
    # Create filter details HTML
    filter_html = '<div style="background-color: #f4f4f4; padding: 10px; margin-bottom: 20px;">'
    filter_html += '<h2>Report Filters</h2>'
    filter_html += '<ul>'
    for detail in filter_details:
        filter_html += f'<li>{detail}</li>'
    filter_html += '</ul></div>'
    
    return filter_html

def analyze_topic_exploration(df):
    """Analyze topic exploration with robust column handling."""
    text_columns = [
        "9. Which of these topic areas would you like to explore with Koomba’s Care Team? (Please select all that apply)",
    ]
    
    # Use a more flexible column matching approach
    matched_columns = [col for col in text_columns if col in df.columns]
    
    if not matched_columns:
        # No matching columns found
        return None
    
    col = matched_columns[0]
    
    # Split topics carefully
    all_topics = []
    for topics in df[col].dropna():
        split_topics = [
            topic.strip() 
            for line in str(topics).split('\n') 
            for topic in line.split(',')
        ]
        all_topics.extend([t.strip() for t in split_topics if t.strip()])
    
    if all_topics:
        topic_counts = pd.Series(all_topics).value_counts()
        total_responses = len(df[col].dropna())
        topic_percentages = (topic_counts / total_responses * 100).round(2)
        
        fig = go.Figure(data=[
            go.Bar(
                x=topic_percentages.index, 
                y=topic_percentages.values,
                text=[f'{p}%' for p in topic_percentages.values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Care Team Topic Exploration',
            xaxis_title='Topics',
            yaxis_title='Percentage of Participants',
            height=500,
            xaxis_tickangle=-45
        )
        
        # Convert to static image for downloadable report
        img_bytes = pio.to_image(fig, format='png', width=800, height=500)
        encoded_img = base64.b64encode(img_bytes).decode('utf-8')
        static_img_html = f'<img src="data:image/png;base64,{encoded_img}" style="max-width:100%; height:auto;">'
        
        return {
            'percentages': topic_percentages.to_dict(),
            'figure': pio.to_html(fig, full_html=False),  # For web
            'static_image': static_img_html  # For download
        }
    
    return None

def generate_time_trend_report(df, filters=None):
    """Generate an HTML report for time trend analysis."""
    filters = filters or {}
    report_content = []
    numeric_cols = get_analyzable_columns(df)
    
    for col in numeric_cols:
        is_single_client = len(df['Unique ID'].unique()) == 1
        
        # Create trend visualization for multiple clients
        trend_fig = go.Figure()
        
        if is_single_client:
            client_data = df.sort_values('Completed')
            numeric_data = pd.to_numeric(client_data[col], errors='coerce')
            valid_data = client_data[numeric_data.notna()]
            
            if len(valid_data) >= 2:
                start_val = float(valid_data.iloc[0][col])
                end_val = float(valid_data.iloc[-1][col])
                change = end_val - start_val
                change_pct = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
                
                # Single client trend visualization
                trend_fig.add_trace(go.Scatter(
                    x=valid_data['Completed'], 
                    y=numeric_data[numeric_data.notna()],
                    mode='lines+markers',
                    name=col
                ))
                
                report_content.append({
                    'Metric': col,
                    'Start Value': f"{start_val:.2f}",
                    'End Value': f"{end_val:.2f}",
                    'Absolute Change': f"{change:.2f}",
                    'Percent Change': f"{change_pct:.1f}%",
                    'Trend Plot': plot_to_html(trend_fig)
                })
        
        else:
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
                
                # Add client-specific trace
                trend_fig.add_trace(go.Scatter(
                    x=valid_data['Completed'], 
                    y=numeric_data[numeric_data.notna()],
                    mode='lines+markers',
                    name=f'Client {client_id}'
                ))
            
            # Add average trend line
            avg_by_date = df.groupby('Completed')[col].apply(
                lambda x: pd.to_numeric(x, errors='coerce').mean()
            ).dropna()
            
            if len(avg_by_date) >= 2:
                trend_fig.add_trace(go.Scatter(
                    x=avg_by_date.index,
                    y=avg_by_date.values,
                    mode='lines+markers',
                    name='Average',
                    line=dict(color='blue', width=3)
                ))
            
            if client_metrics:
                avg_start = np.mean([m['start'] for m in client_metrics])
                avg_end = np.mean([m['end'] for m in client_metrics])
                avg_change = np.mean([m['change'] for m in client_metrics])
                avg_change_pct = np.mean([m['change_pct'] for m in client_metrics])
                
                report_content.append({
                    'Metric': col,
                    'Average Start Value': f"{avg_start:.2f}",
                    'Average End Value': f"{avg_end:.2f}",
                    'Average Absolute Change': f"{avg_change:.2f}",
                    'Average Percent Change': f"{avg_change_pct:.1f}%",
                    'Trend Plot': plot_to_html(trend_fig)
                })
    
    # Convert report content to DataFrame for HTML display
    report_df = pd.DataFrame(report_content)
    
    # Create full HTML report
    html_report = f"""
    <html>
    <head>
        <title>Time Trend Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: auto; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Time Trend Analysis Report</h1>
        {generate_report_header(df, 'Time Trend Analysis', filters)}
        {report_df.to_html(index=False, escape=False)}
    </body>
    </html>
    """
    
    return html_report

def generate_summary_report(df, filters=None):
    """Generate an HTML report for summary statistics with visualizations."""
    try:
        filters = filters or {}
        
        total_clients = len(df['Unique ID'].unique())
        total_measurements = len(df)
        measurements_per_client = df.groupby('Unique ID').size()
        
        report_header = generate_report_header(df, 'Summary Statistics', filters)
        numeric_cols = get_analyzable_columns(df)
        
        stats_data = []
        distribution_plots = []
        
        for col in numeric_cols:
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            stats = numeric_data.describe()
            valid_count = numeric_data.notna().sum()
            
            stats_entry = {
                "Metric": col,
                "Mean": f"{stats['mean']:.2f}",
                "Median": f"{stats['50%']:.2f}",
                "Std Dev": f"{stats['std']:.2f}",
                "Min": f"{stats['min']:.2f}",
                "Max": f"{stats['max']:.2f}",
                "Valid Responses": valid_count
            }
            stats_data.append(stats_entry)
            
            # Create distribution plot
            if len(numeric_data.dropna()) >= 2:
                fig = go.Figure()
                
                # Calculate optimal number of bins
                q75, q25 = np.percentile(numeric_data.dropna(), [75, 25])
                iqr = q75 - q25
                bin_width = 2 * iqr / (len(numeric_data.dropna()) ** (1/3)) if iqr > 0 else 0.5
                num_bins = int((numeric_data.max() - numeric_data.min()) / bin_width) if bin_width > 0 else 30
                num_bins = min(max(10, num_bins), 50)
                
                fig.add_trace(go.Histogram(
                    x=numeric_data.dropna(),
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
                    bargap=0.1
                )
                
                # Convert plot to HTML
                distribution_plots.append(plot_to_html(fig))
        
        # Analyze topic exploration
        topic_analysis = analyze_topic_exploration(df)
        
        # HTML Report Generation
        html_report = f"""
        <html>
        <head>
            <title>Summary Statistics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: auto; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Summary Statistics Report</h1>
            {report_header}
            
            <h2>Overall Statistics</h2>
            <table>
                <tr>
                    <th>Total Clients</th>
                    <th>Total Measurements</th>
                    <th>Avg Measurements/Client</th>
                    <th>Median Measurements/Client</th>
                </tr>
                <tr>
                    <td>{total_clients}</td>
                    <td>{total_measurements}</td>
                    <td>{measurements_per_client.mean():.1f}</td>
                    <td>{measurements_per_client.median():.1f}</td>
                </tr>
            </table>
        """
        
        # Topic Exploration Section
        if topic_analysis and topic_analysis.get('static_image'):
            html_report += f"""
            <h2>Topic Exploration</h2>
            {topic_analysis.get('static_image', '')}
            <h3>Topic Percentages</h3>
            <table>
            <tr><th>Topic</th><th>Percentage</th></tr>
            {''.join([f'<tr><td>{k}</td><td>{v}%</td></tr>' for k, v in topic_analysis.get('percentages', {}).items()])}
            </table>
            """
        
        # Numeric Metrics Section
        if numeric_cols:
            html_report += """
            <h2>Detailed Metric Statistics</h2>
            <h3>Metrics Summary</h3>
            <table>
                <tr><th>Metric</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Max</th><th>Valid Responses</th></tr>
            """
            for entry in stats_data:
                html_report += f"""
                <tr>
                    <td>{entry['Metric']}</td>
                    <td>{entry['Mean']}</td>
                    <td>{entry['Median']}</td>
                    <td>{entry['Std Dev']}</td>
                    <td>{entry['Min']}</td>
                    <td>{entry['Max']}</td>
                    <td>{entry['Valid Responses']}</td>
                </tr>
                """
            html_report += "</table>"
            
            # Distribution Plots
            html_report += """
            <h3>Distribution Plots</h3>
            """
            for plot in distribution_plots:
                html_report += f"<div>{plot}</div>"
        
        html_report += """
        </body>
        </html>
        """
        
        return html_report
    
    except Exception as e:
        print(f"Error in summary report generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise

def generate_downloadable_report(df, analysis_type, filters=None):
    """Generate HTML report for download."""
    if analysis_type == "Time Trend Analysis":
        return generate_time_trend_report(df, filters)
    else:
        html_report = generate_summary_report(df, filters)
        return html_report