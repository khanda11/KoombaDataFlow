import streamlit as st
import pandas as pd
import plotly.express as px
import re

# Set up the Streamlit app
st.title("Koomba Data Analysis")
st.write("Upload your CSV files to calculate rate of change for each person.")

# Upload CSV file(s)
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

# Define required columns for validation
REQUIRED_COLUMNS = ["Unique ID", "Completed", "Client"]

def extract_numeric_value(s):
    """
    Extract numeric value from a string that may contain letters.
    """
    if isinstance(s, str):
        match = re.search(r"(\d+(\.\d+)?)", s)  # Match numeric part
        if match:
            return float(match.group(1))
    return None

def preprocess_numeric_columns(df):
    """
    Preprocess numeric columns to extract numeric values from mixed-content fields.
    """
    for col in df.columns:
        if df[col].dtype == "object":
            # Extract numeric values and coerce the rest to NaN
            df[col] = df[col].apply(extract_numeric_value)
    return df

def validate_and_clean(file):
    try:
        # Load the CSV with the correct delimiter
        df = pd.read_csv(file, delimiter=";")
        st.write(f"**File {file.name} loaded successfully!**")
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
        return None

    # Check for missing columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        st.error(f"File {file.name} is missing columns: {missing_columns}")
        return None
    else:
        st.success(f"File {file.name} passed validation!")

    # Handle missing data
    df.dropna(subset=["Unique ID", "Completed"], inplace=True)  # Drop rows missing critical fields

    # Standardize column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("\xa0", "", regex=True)

    # Fix timezone issues and format "Completed" column
    try:
        df["Completed"] = pd.to_datetime(
            df["Completed"].str.replace("EST", "").str.strip(), errors="coerce"
        )
    except Exception as e:
        st.warning(f"Unable to parse 'Completed' timestamps in {file.name}: {e}")

    return df

def calculate_total_change(df, unique_id_col, time_col):
    """
    Calculate the total change for numeric columns grouped by Unique ID.
    """
    df = df.sort_values(by=[unique_id_col, time_col])
    df = preprocess_numeric_columns(df)
    numeric_columns = [
        col for col in df.select_dtypes(include=["number"]).columns 
        if col not in ["Unique ID", "Client Phone Number"]
    ]

    results = []
    for unique_id, group in df.groupby(unique_id_col):
        earliest_row = group.iloc[0]
        latest_row = group.iloc[-1]
        time_delta = latest_row[time_col] - earliest_row[time_col]
        time_delta_days = time_delta.days if isinstance(time_delta, pd.Timedelta) else 0

        result = {
            "Unique ID": unique_id,
            "Time Delta (days)": time_delta_days,
        }

        for col in numeric_columns:
            try:
                start_value = pd.to_numeric(earliest_row[col], errors="coerce")
                end_value = pd.to_numeric(latest_row[col], errors="coerce")

                if pd.notna(start_value) and pd.notna(end_value):
                    change = end_value - start_value
                    result[col] = f"{change:.2f} ({start_value:.2f} → {end_value:.2f})"
                else:
                    result[col] = "N/A"
            except Exception as e:
                result[col] = "N/A"

        results.append(result)

    return pd.DataFrame(results)

def generate_plotly_line_graph(df, unique_id_col, time_col, client_mapping):
    """
    Generate line graphs for numeric fields for each unique client.
    """
    df = preprocess_numeric_columns(df)
    numeric_columns = [
        col for col in df.select_dtypes(include=["number"]).columns 
        if col not in ["Unique ID", "Client Phone Number"]
    ]

    for unique_id, group in df.groupby(unique_id_col):
        if len(group) <= 1:
            st.warning(f"Not enough data points to generate a graph for Client ID {unique_id}.")
            continue

        client_name = client_mapping.get(unique_id, "Unknown Client")
        st.subheader(f"Client: {client_name}")

        for col in numeric_columns:
            fig = px.line(
                group.sort_values(time_col),
                x=time_col,
                y=col,
                title=f"{col} Trend for {client_name}",
                labels={time_col: "Timestamp", col: "Value"},
                markers=True
            )
            st.plotly_chart(fig)

        csv_data = group.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {client_name} Data",
            data=csv_data,
            file_name=f"{client_name}_data.csv",
            mime="text/csv",
            key=f"{unique_id}-download"
        )

def display_total_change_table(total_change_df, combined_data):
    """
    Display total change data in a tabular format using Streamlit's dataframe.
    """
    st.header("Total Change Summary (Table Format)")

    client_mapping = combined_data.set_index("Unique ID")["Client"].to_dict()
    total_change_df["Client"] = total_change_df["Unique ID"].map(client_mapping)

    filtered_columns = [col for col in total_change_df.columns if not total_change_df[col].eq("N/A").all()]
    total_change_df = total_change_df[filtered_columns]

    columns = ["Client", "Time Delta (days)"] + [
        col for col in total_change_df.columns 
        if col not in ["Client", "Time Delta (days)"]
    ]
    total_change_df = total_change_df[columns]

    st.dataframe(total_change_df, use_container_width=True)

# Main App Logic
if uploaded_files:
    all_data = []
    for file in uploaded_files:
        cleaned_data = validate_and_clean(file)
        if cleaned_data is not None:
            all_data.append(cleaned_data)

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        st.write("**Combined Data from All Files:**")
        st.dataframe(combined_data)

        # Ensure required columns are present
        if "Client" not in combined_data.columns or "Unique ID" not in combined_data.columns:
            st.error("The dataset must include 'Client' and 'Unique ID' columns.")
        elif "Client's Provider" not in combined_data.columns:
            st.error("The dataset must include a 'Client's Provider' column.")
        else:
            # Create client mapping before using it
            client_mapping = combined_data.set_index("Unique ID")["Client"].to_dict()

            # Filter Section
            st.sidebar.header("Filters")

            # Date Filter
            if "Completed" in combined_data.columns:
                min_date = pd.to_datetime(combined_data["Completed"]).min()
                max_date = pd.to_datetime(combined_data["Completed"]).max()
                date_range = st.sidebar.date_input(
                    "Select Date Range",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )
            else:
                date_range = None

            # Provider Filter
            actual_providers = combined_data["Client's Provider"].dropna().unique()
            selected_provider = st.sidebar.selectbox(
                "Select Provider (one at a time)",
                options=["All"] + list(actual_providers),
                index=0
            )

            # Client Filter using the created client_mapping
            unique_ids = combined_data["Unique ID"].dropna().unique()
            client_names = [client_mapping.get(uid, "Unknown") for uid in unique_ids]
            client_name_to_id = {client_mapping.get(uid, "Unknown"): uid for uid in unique_ids}
            selected_client = st.sidebar.selectbox(
                "Select Athlete (one at a time)",
                options=["All"] + client_names,
                index=0
            )

            # Apply Filters
            filtered_data = combined_data.copy()

            if date_range:
                filtered_data = filtered_data[
                    filtered_data["Completed"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
                ]

            if selected_provider != "All":
                filtered_data = filtered_data[filtered_data["Client's Provider"] == selected_provider]

            if selected_client != "All":
                filtered_data = filtered_data[filtered_data["Unique ID"] == client_name_to_id[selected_client]]

            if not filtered_data.empty:
                st.write("**Filtered Data:**")
                st.dataframe(filtered_data)

                total_change_df = calculate_total_change(filtered_data, "Unique ID", "Completed")
                display_total_change_table(total_change_df, filtered_data)
                generate_plotly_line_graph(filtered_data, "Unique ID", "Completed", client_mapping)
            else:
                st.warning("No data available for the selected filters.")