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

    # Log detected column names
   

    # Check for missing columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        st.error(f"File {file.name} is missing columns: {missing_columns}")
        return None
    else:
        st.success(f"File {file.name} passed validation!")

    # Handle missing data
    
    df.dropna(subset=["Unique ID", "Completed"], inplace=True)  # Drop rows missing critical fields

    # Standardize column names (strip spaces and remove non-breaking spaces)
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
    df = preprocess_numeric_columns(df)  # Ensure numeric columns are preprocessed
    numeric_columns = [
        col for col in df.select_dtypes(include=["number"]).columns if col not in ["Unique ID", "Client Phone Number"]
    ]

    results = []
    for unique_id, group in df.groupby(unique_id_col):
        earliest_row = group.iloc[0]
        latest_row = group.iloc[-1]
        time_delta = latest_row[time_col] - earliest_row[time_col]
        time_delta_days = time_delta.days if isinstance(time_delta, pd.Timedelta) else 0  # Ensure timedelta is handled

        result = {
            "Unique ID": unique_id,
            "Time Delta (days)": time_delta_days,
        }

        for col in numeric_columns:
            try:
                # Ensure values are numeric
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
    df = preprocess_numeric_columns(df)  # Preprocess the data to include numeric values
    numeric_columns = [
        col for col in df.select_dtypes(include=["number"]).columns if col not in ["Unique ID", "Client Phone Number"]
    ]

    for unique_id, group in df.groupby(unique_id_col):
        if len(group) <= 1:  # Skip graph generation if one or fewer data points
            st.warning(f"Not enough data points to generate a graph for Client ID {unique_id}.")
            continue

        client_name = client_mapping.get(unique_id, "Unknown Client")  # Use mapping to get the name
        st.subheader(f"Client: {client_name}")

        # Generate line graph for each numeric column
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

        # Allow CSV download of this client's data with unique keys
        csv_data = group.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {client_name} Data",
            data=csv_data,
            file_name=f"{client_name}_data.csv",
            mime="text/csv",
            key=f"{unique_id}-download"  # Ensure unique key for each button
        )

def display_total_change_table(total_change_df, combined_data):
    """
    Display total change data in a tabular format using Streamlit's dataframe.
    """
    st.header("Total Change Summary (Table Format)")

    # Map Unique ID to Client Name
    client_mapping = combined_data.set_index("Unique ID")["Client"].to_dict()
    total_change_df["Client"] = total_change_df["Unique ID"].map(client_mapping)

    # Drop columns where all values are "N/A"
    filtered_columns = [col for col in total_change_df.columns if not total_change_df[col].eq("N/A").all()]
    total_change_df = total_change_df[filtered_columns]

    # Reorganize columns
    columns = ["Client", "Time Delta (days)"] + [col for col in total_change_df.columns if col not in ["Client", "Time Delta (days)"]]
    total_change_df = total_change_df[columns]

    # Display as a table
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

        # Create client mapping once and use it throughout
        client_mapping = combined_data.set_index("Unique ID")["Client"].to_dict()

        # Calculate total change
        total_change_df = calculate_total_change(combined_data, "Unique ID", "Completed")
        # Display total change data as a table
        display_total_change_table(total_change_df, combined_data)

        # Generate line graphs for each unique client
        generate_plotly_line_graph(combined_data, "Unique ID", "Completed", client_mapping)
