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

def normalize_column_name(column):
    """Remove suffixes like .1, .2 etc from column names and clean up the name."""
    # Remove .1, .2 etc suffixes
    base_name = re.sub(r'\.\d+$', '', column)
    return base_name

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
                       'First', 'Last', 'Email', 'Class Year', 'Completed', 'Visit_Number',
                       'GAD7_Severity']
    
    topic_column = "9. Which of these topic areas would you like to explore with Koomba's Care Team? (Please select all that apply)"
    
    score_columns = []
    if 'GAD7_Total_Score' in df.columns:
        score_columns.append('GAD7_Total_Score')
    if 'SMHAT_Total_Score' in df.columns:
        score_columns.append('SMHAT_Total_Score')
    
    column_mapping = {}
    for col in df.columns:
        if col not in excluded_columns and col not in score_columns:
            normalized_name = normalize_column_name(col)
            if is_numeric_column(df, col):
                if normalized_name in column_mapping:
                    if '.1' not in col:
                        column_mapping[normalized_name] = col
                else:
                    column_mapping[normalized_name] = col
    
    return score_columns + list(column_mapping.values())

def create_assessment_visualization(df, assessment_type, view_type="By Date"):
    """Create visualization for assessment scores with support for date and visit number views."""
    try:
        score_column = f"{assessment_type}_Total_Score"
        if score_column not in df.columns:
            return None
            
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Sort data based on view type
        if view_type == "By Date":
            df_sorted = df.sort_values('Completed')
            x_axis = 'Completed'
            x_title = 'Date'
        else:
            df_sorted = df.sort_values(['Unique ID', 'Visit_Number'])
            x_axis = 'Visit_Number'
            x_title = 'Visit Number'

        # Check if we're dealing with single or multiple clients
        is_single_client = len(df['Unique ID'].unique()) == 1
        metrics_html = ""
        
        if is_single_client:
            # Single client visualization
            valid_data = df_sorted[df_sorted[score_column].notna()]
            if len(valid_data) < 2:
                return None

            start_val = float(valid_data.iloc[0][score_column])
            end_val = float(valid_data.iloc[-1][score_column])
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
                         f"{'+' if change > 0 else ''}{change:.2f}")
            with col4:
                st.metric("Percent Change", f"{change_pct:.1f}%",
                         f"{'+' if change_pct > 0 else ''}{change_pct:.1f}%")

            fig.add_trace(
                go.Scatter(
                    x=valid_data[x_axis],
                    y=valid_data[score_column],
                    name=f"{assessment_type} Score",
                    mode='lines+markers',
                    text=valid_data['Visit_Number'] if view_type == "By Date" else None
                ),
                secondary_y=False
            )
        else:
            # Multiple clients visualization
            valid_clients = []
            client_metrics = []
            
            for client_id in df['Unique ID'].unique():
                client_data = df_sorted[
                    (df_sorted['Unique ID'] == client_id) & 
                    (df_sorted[score_column].notna())
                ]
                if len(client_data) >= 2:
                    start_val = float(client_data.iloc[0][score_column])
                    end_val = float(client_data.iloc[-1][score_column])
                    change = end_val - start_val
                    change_pct = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
                    client_metrics.append({
                        'start': start_val,
                        'end': end_val,
                        'change': change,
                        'change_pct': change_pct
                    })
                    valid_clients.append(client_id)
                    fig.add_trace(
                        go.Scatter(
                            x=client_data[x_axis],
                            y=client_data[score_column],
                            name=f"Client {client_id}",
                            mode='lines+markers',
                            line=dict(color='lightgray'),
                            text=client_data['Visit_Number'] if view_type == "By Date" else None
                        ),
                        secondary_y=False
                    )
            
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
                             f"{'+' if avg_change > 0 else ''}{avg_change:.2f}")
                with col4:
                    st.metric("Avg Percent Change", f"{avg_change_pct:.1f}%",
                             f"{'+' if avg_change_pct > 0 else ''}{avg_change_pct:.1f}%")

                if valid_clients:
                    valid_data = df_sorted[
                        (df_sorted['Unique ID'].isin(valid_clients)) & 
                        (df_sorted[score_column].notna())
                    ]
                    
                    if view_type == "By Date":
                        avg_by_date = valid_data.groupby('Completed')[score_column].mean()
                        x_vals = avg_by_date.index
                        y_vals = avg_by_date.values
                    else:
                        avg_by_visit = valid_data.groupby('Visit_Number')[score_column].mean()
                        x_vals = avg_by_visit.index
                        y_vals = avg_by_visit.values

                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            name='Average',
                            mode='lines+markers',
                            line=dict(color='blue', width=3)
                        ),
                        secondary_y=False
                    )
            else:
                return None
        
        # Update layout based on view type
        fig.update_layout(
            title=f"{assessment_type} Assessment Scores Over {x_title}",
            xaxis_title=x_title,
            yaxis_title=f"{assessment_type} Score",
            height=400
        )

        if view_type == "By Date":
            fig.update_layout(
                xaxis=dict(
                    type='date',
                    tickformat='%b %d\n%Y'
                )
            )
        else:
            fig.update_layout(
                xaxis=dict(
                    tickmode='linear',
                    dtick=1
                )
            )
        
        return fig
    except Exception as e:
        st.error(f"Error creating {assessment_type} visualization: {str(e)}")
        return None
def display_time_trend_analysis(df):
    """Display interactive time trend analysis including assessment scores."""
    # Add view type selector
    view_type = st.radio("View Data By:", ["By Date", "By Visit Number"])
    
    # Get numeric columns for analysis
    numeric_cols = get_analyzable_columns(df)
    
    # Check for and display assessment scores first
    if 'GAD7_Total_Score' in df.columns:
        st.subheader("GAD-7 Assessment Analysis")
        gad7_fig = create_assessment_visualization(df, 'GAD7', view_type)  # Pass view_type here
        if gad7_fig:
            st.plotly_chart(gad7_fig, use_container_width=True)
            
            
    
    if 'SMHAT_Total_Score' in df.columns:
        st.subheader("SMHAT Assessment Analysis")
        smhat_fig = create_assessment_visualization(df, 'SMHAT', view_type)  # Pass view_type here
        if smhat_fig:
            st.plotly_chart(smhat_fig, use_container_width=True)
            
    
    # Continue with analysis for other numeric columns
    for col in numeric_cols:
        if not col.endswith('_Score') and not col.endswith('_Severity'):
            display_name = normalize_column_name(col)
            st.subheader(f"{display_name} Analysis")
            
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            valid_data = df[numeric_data.notna()]
            
            # Sort data based on view type
            if view_type == "By Date":
                valid_data = valid_data.sort_values('Completed')
                x_axis = 'Completed'
                x_title = 'Date'
            else:
                valid_data = valid_data.sort_values(['Unique ID', 'Visit_Number'])
                x_axis = 'Visit_Number'
                x_title = 'Visit Number'
            
            if len(valid_data) < 2:
                st.warning(f"Insufficient data points for {display_name} analysis. Need at least 2 valid measurements.")
                continue
            
            # Single client or multiple clients logic
            is_single_client = len(df['Unique ID'].unique()) == 1
            
            if is_single_client:
                # Single client analysis
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
                valid_clients = []
                
                for client_id in df['Unique ID'].unique():
                    client_data = valid_data[valid_data['Unique ID'] == client_id]
                    
                    if len(client_data) >= 2:
                        start_val = float(client_data.iloc[0][col])
                        end_val = float(client_data.iloc[-1][col])
                        change = end_val - start_val
                        change_pct = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
                        client_metrics.append({
                            'start': start_val,
                            'end': end_val,
                            'change': change,
                            'change_pct': change_pct
                        })
                        valid_clients.append(client_id)
                
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
            has_valid_data = is_single_client and len(valid_data) >= 2 or not is_single_client and valid_clients
            
            if has_valid_data:
                fig = go.Figure()
                
                if is_single_client:
                    fig.add_trace(go.Scatter(
                        x=valid_data[x_axis],
                        y=numeric_data[valid_data.index],
                        mode='lines+markers',
                        name=display_name,
                        text=valid_data['Visit_Number'] if view_type == "By Date" else None
                    ))
                else:
                    # Add individual client lines
                    for client_id in valid_clients:
                        client_data = valid_data[valid_data['Unique ID'] == client_id]
                        fig.add_trace(go.Scatter(
                            x=client_data[x_axis],
                            y=pd.to_numeric(client_data[col], errors='coerce'),
                            mode='lines+markers',
                            line=dict(color='lightgray'),
                            name=f'Client {client_id}',
                            text=client_data['Visit_Number'] if view_type == "By Date" else None,
                            showlegend=True
                        ))
                    
                    # Add average trend line
                    if view_type == "By Date":
                        avg_by_date = valid_data.groupby('Completed')[col].apply(
                            lambda x: pd.to_numeric(x, errors='coerce').mean()
                        ).sort_index()
                        x_vals = avg_by_date.index
                        y_vals = avg_by_date.values
                    else:
                        avg_by_visit = valid_data.groupby('Visit_Number')[col].apply(
                            lambda x: pd.to_numeric(x, errors='coerce').mean()
                        ).sort_index()
                        x_vals = avg_by_visit.index
                        y_vals = avg_by_visit.values
                    
                    if len(x_vals) >= 2:
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode='lines+markers',
                            name='Average',
                            line=dict(color='blue', width=3)
                        ))
                
                # Update layout based on view type
                fig.update_layout(
                    title=f'{display_name} Progress Over {x_title}',
                    xaxis_title=x_title,
                    yaxis_title='Score',
                    height=400
                )
                
                if view_type == "By Date":
                    fig.update_layout(
                        xaxis=dict(
                            type='date',
                            tickformat='%b %d\n%Y'
                        )
                    )
                else:
                    fig.update_layout(
                        xaxis=dict(
                            tickmode='linear',
                            dtick=1
                        )
                    )
                
                st.plotly_chart(fig, use_container_width=True)
def display_summary_statistics(df):
    """Display interactive summary statistics including assessment scores and topic analysis."""
    st.header("Overall Statistics")
    
    # Display assessment statistics first if available
    if 'GAD7_Total_Score' in df.columns:
        st.subheader("GAD-7 Assessment Summary")
        gad7_stats = df['GAD7_Total_Score'].describe()
        st.write("GAD-7 Total Score Statistics:")
        st.write(pd.DataFrame(gad7_stats).transpose())
    
    if 'SMHAT_Total_Score' in df.columns:
        st.subheader("SMHAT Assessment Summary")
        smhat_stats = df['SMHAT_Total_Score'].describe()
        st.write("SMHAT Total Score Statistics:")
        st.write(pd.DataFrame(smhat_stats).transpose())
    
    total_clients = len(df['Unique ID'].unique())
    total_measurements = len(df)
    measurements_per_client = df.groupby('Unique ID').size()
    
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
    
    # Add Topic Analysis Section
    topic_column = "9. Which of these topic areas would you like to explore with Koomba's Care Team? (Please select all that apply)"
    if topic_column in df.columns:
        st.header("Topic Exploration Analysis")
        
        # Process topic data
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
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=topic_percentages.index,
                    y=topic_percentages.values,
                    text=[f'{p:.1f}%' for p in topic_percentages.values],
                    textposition='auto',
                    marker_color='rgb(55, 83, 109)'
                )
            ])
            
            fig.update_layout(
                title='Care Team Topic Preferences',
                xaxis_title='Topics',
                yaxis_title='Percentage of Participants',
                height=500,
                xaxis_tickangle=-45,
                showlegend=False,
                margin=dict(b=100)  # Add bottom margin for rotated labels
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display topic percentages in a table
            st.subheader("Topic Selection Breakdown")
            topic_df = pd.DataFrame({
                'Topic': topic_percentages.index,
                'Percentage': [f'{p:.1f}%' for p in topic_percentages.values],
                'Count': topic_counts.values
            })
            st.dataframe(topic_df, use_container_width=True)
    
    # Get analyzable numeric columns
    numeric_cols = get_analyzable_columns(df)
    
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
        
        # Create the Plotly figure
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
            xaxis_tickangle=-45,
            xaxis_tickfont_size=8,  # Increase font size for x-axis labels
            bargap=0.1  # Adjust the gap between bars
        )
        
        # Render the figure as a static image
        img_bytes = pio.to_image(fig, format='png', width=1200, height=600)
        encoded_img = base64.b64encode(img_bytes).decode('utf-8')
        static_img_html = f'<img src="data:image/png;base64,{encoded_img}" style="max-width:100%; height:auto;">'
        
        return {
            'percentages': topic_percentages.to_dict(),
            'static_image': static_img_html
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