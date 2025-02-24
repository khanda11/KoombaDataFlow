import streamlit as st
import pandas as pd
from datetime import datetime
from utils import data_processing
from utils.reports import (
    display_time_trend_analysis,
    display_summary_statistics,
    generate_downloadable_report
)


st.set_page_config(
    page_title="Koomba Data Analysis",
    page_icon="📊",
    layout="wide"
)


st.title("Koomba Data Analysis")
st.write("Upload your CSV files to analyze client progress.")


analysis_type = st.radio(
    "Select Analysis Type",
    ["Time Trend Analysis", "Summary Statistics"],
    help="Choose between analyzing trends over time or viewing summary statistics"
)

# Define required columns for validation
REQUIRED_COLUMNS = ["Unique ID", "Team"]

# Initialize empty DataFrame for combined data
combined_data = pd.DataFrame(columns=REQUIRED_COLUMNS)

# File upload section
uploaded_files = st.file_uploader(
    "Upload CSV files", 
    type=["csv"], 
    accept_multiple_files=True,
    help="Upload one or more CSV files containing client data"
)

if uploaded_files:
    all_data = []

    for file in uploaded_files:
        try:
            # Read CSV file
            df = pd.read_csv(file, delimiter=";")
            st.success(f"File {file.name} uploaded successfully.")

            # Validate required columns
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                st.error(f"File {file.name} is missing required columns: {missing_cols}")
                continue

            # Preprocess data
            df = data_processing.preprocess_data(df)
            all_data.append(df)

        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)

# Ensure combined_data has content before processing
if not combined_data.empty:
    # Add download button for full unfiltered data
    st.write("---")
    st.subheader("Download Full Dataset")
    csv_data = combined_data.to_csv(index=False)
    st.download_button(
        label="Download Complete Dataset (CSV)",
        data=csv_data,
        file_name=f"complete_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Convert the "Completed" column to datetime
    combined_data["Completed"] = pd.to_datetime(combined_data["Completed"], errors='coerce')

    # Calculate visit numbers for the entire dataset
    combined_data['Visit_Number'] = combined_data.groupby('Unique ID')['Completed'].rank(method='dense')
    max_visits = int(combined_data['Visit_Number'].max())

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        
        # Create tabs for different filter categories
        date_tab, client_tab, visit_tab = st.tabs(["Date Filters", "Client Filters", "Visit Filters"])
        
        with date_tab:
            # Date Range Filter
            min_date = combined_data["Completed"].min().date()
            max_date = combined_data["Completed"].max().date()
            date_range = st.date_input(
                "Select Date Range",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date,
                help="Filter data by date range"
            )

        with client_tab:
            # Provider Filter
            providers = ["All"] + sorted(combined_data["Client's Provider"].dropna().unique().tolist())
            selected_provider = st.selectbox(
                "Select Provider",
                providers,
                help="Filter by healthcare provider"
            )

            # Client Filter
            clients = ["All"] + sorted(combined_data["Client"].dropna().unique().tolist())
            selected_client = st.selectbox(
                "Select Client",
                clients,
                help="Filter by specific client"
            )

            # Team Filter
            teams = ["All"] + sorted(combined_data["Team"].dropna().unique().tolist())
            selected_team = st.selectbox(
                "Select Team",
                teams,
                help="Filter by team"
            )

        with visit_tab:
            # Visit Number Filter
            enable_visit_filter = st.checkbox(
                "Filter by Visit Number",
                False,
                help="Enable filtering by visit sequence number"
            )
            
            if enable_visit_filter:
                visit_range = st.slider(
                    "Select Visit Range",
                    min_value=1,
                    max_value=max_visits,
                    value=(1, max_visits),
                    help="Filter data to show only selected visit numbers"
                )
                
                # Option to show only clients with complete data for selected range
                complete_data_only = st.checkbox(
                    "Show only clients with complete data for selected range",
                    False,
                    help="Filter out clients who don't have data for all visits in the selected range"
                )

    # Apply filters
    filtered_data = combined_data.copy()

    # Date filter
    if len(date_range) == 2:
        filtered_data = filtered_data[
            (filtered_data["Completed"].dt.date >= date_range[0]) &
            (filtered_data["Completed"].dt.date <= date_range[1])
        ]

    # Client filters
    if selected_provider != "All":
        filtered_data = filtered_data[filtered_data["Client's Provider"] == selected_provider]
    if selected_client != "All":
        filtered_data = filtered_data[filtered_data["Client"] == selected_client]
    if selected_team != "All":
        filtered_data = filtered_data[filtered_data["Team"] == selected_team]

    # Visit number filter
    if enable_visit_filter:
        filtered_data = filtered_data[
            filtered_data['Visit_Number'].between(visit_range[0], visit_range[1])
        ]
        
        if complete_data_only:
            # Get clients with complete data for the selected range
            visit_counts = filtered_data.groupby('Unique ID')['Visit_Number'].agg(['min', 'max', 'count'])
            complete_clients = visit_counts[
                (visit_counts['min'] <= visit_range[0]) &
                (visit_counts['max'] >= visit_range[1]) &
                (visit_counts['count'] >= (visit_range[1] - visit_range[0] + 1))
            ].index
            filtered_data = filtered_data[filtered_data['Unique ID'].isin(complete_clients)]

    # Check if data remains after filtering
    if filtered_data.empty:
        st.warning("No data available after applying filters. Please adjust the filters and try again.")
    else:
        # Display filter summary
        st.write("---")
        st.subheader("Applied Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"Date Range: {date_range[0]} to {date_range[1]}")
        with col2:
            st.write(f"Provider: {selected_provider}")
            st.write(f"Team: {selected_team}")
        with col3:
            st.write(f"Clients: {len(filtered_data['Unique ID'].unique())}")
            if enable_visit_filter:
                st.write(f"Visit Range: {visit_range[0]} to {visit_range[1]}")

        # Display interactive analysis based on selected type
        if analysis_type == "Time Trend Analysis":
            st.header("Time Trend Analysis")
            display_time_trend_analysis(filtered_data)
        else:
            st.header("Summary Statistics")
            display_summary_statistics(filtered_data)

        # Add download button for HTML report
        st.write("---")
        st.subheader("Download Report")
        
        # Prepare filters for report generation
        filters = {
            'provider': selected_provider,
            'client': selected_client,
            'team': selected_team,
            'date_range': date_range,
            'visit_range': visit_range if enable_visit_filter else None
        }
        
        report_html = generate_downloadable_report(filtered_data, analysis_type, filters)
        output_file = f"{analysis_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        st.download_button(
            label=f"Download {analysis_type} Report",
            data=report_html,
            file_name=output_file,
            mime="text/html"
        )
else:
    st.info("👆 Please upload your CSV files to begin analysis.")