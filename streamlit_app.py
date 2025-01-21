import streamlit as st
import pandas as pd
from datetime import datetime
from utils import data_processing
from utils.reports import generate_summary_report, generate_time_trend_report, save_report

# Set up the Streamlit app
st.title("Koomba Data Analysis")
st.write("Upload your CSV files to analyze client progress.")

# Upload CSV file(s)
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

# Define required columns for validation
REQUIRED_COLUMNS = ["Unique ID", "Completed", "Client", "Client's Provider", "Team"]

# Add analysis type selection
analysis_type = st.radio(
    "Select Analysis Type",
    ["Time Trend Analysis", "Summary Statistics"],
    help="Choose between analyzing trends over time or viewing summary statistics"
)

# Initialize empty DataFrame for combined data
combined_data = pd.DataFrame(columns=REQUIRED_COLUMNS)

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
    # Convert the "Completed" column to datetime
    combined_data["Completed"] = pd.to_datetime(combined_data["Completed"], errors='coerce')

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date Range Filter
    min_date = combined_data["Completed"].min().date()
    max_date = combined_data["Completed"].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date
    )

    # Provider Filter
    providers = ["All"] + sorted(combined_data["Client's Provider"].dropna().unique().tolist())
    selected_provider = st.sidebar.selectbox("Select Provider", providers)

    # Client Filter
    clients = ["All"] + sorted(combined_data["Client"].dropna().unique().tolist())
    selected_client = st.sidebar.selectbox("Select Client", clients)

    # Team Filter
    teams = ["All"] + sorted(combined_data["Team"].dropna().unique().tolist())
    selected_team = st.sidebar.selectbox("Select Team", teams)

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
    if selected_team != "All":
        filtered_data = filtered_data[filtered_data["Team"] == selected_team]

    # Check if data remains after filtering
    if filtered_data.empty:
        st.warning("No data available after applying filters. Adjust the filters and try again.")
    else:
        # Generate appropriate report based on analysis type
        if analysis_type == "Time Trend Analysis":
            st.subheader("Time Trend Analysis")
            report_html = generate_time_trend_report(filtered_data)
            output_file = f"time_trend_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        else:  # Summary Statistics
            st.subheader("Summary Statistics")
            report_html = generate_summary_report(filtered_data)
            output_file = f"summary_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        # Display the report
        st.components.v1.html(report_html, height=800, scrolling=True)

        # Download button for the report
        st.download_button(
            label=f"Download {analysis_type} Report",
            data=report_html,
            file_name=output_file,
            mime="text/html"
        )
else:
    st.warning("No valid data available. Please upload valid CSV files.")

