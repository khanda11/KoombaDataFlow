import streamlit as st
import pandas as pd

# Set up the Streamlit app
st.title("Koomba Data Analysis")
st.write("Upload your CSV files to calculate total change for each person.")


uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)


REQUIRED_COLUMNS = ["Unique ID", "Completed"]

def validate_and_clean(file):
    try:
        # Load the CSV with the correct delimiter
        df = pd.read_csv(file, delimiter=';')
        st.write(f"**File {file.name} loaded successfully!**")
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
        return None
    

    st.write(f"**Detected columns in {file.name}:** {df.columns.tolist()}")

   
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        st.error(f"File {file.name} is missing columns: {missing_columns}")
        return None
    else:
        st.success(f"File {file.name} passed validation!")
    
    # Handle missing data
    st.write(f"Processing {file.name}...")
    df.dropna(subset=["Unique ID", "Completed"], inplace=True)  # Drop rows missing critical fields

    # Standardize column names (strip spaces and remove non-breaking spaces)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("\xa0", "", regex=True)

    # Format the "Completed" column to datetime
    try:
        df["Completed"] = pd.to_datetime(df["Completed"], errors="coerce")
        if df["Completed"].isna().any():
            st.warning(f"Some timestamps in {file.name} could not be parsed and were set to NaT.")
    except Exception as e:
        st.warning(f"Unable to parse 'Completed' timestamps in {file.name}: {e}")
    
    return df

def calculate_total_change(df, unique_id_col, time_col):
    """
    Calculate the total change for numeric columns grouped by Unique ID.

    Args:
        df (pd.DataFrame): The cleaned dataset.
        unique_id_col (str): Column representing unique IDs (e.g., "Unique ID").
        time_col (str): Column representing timestamps (e.g., "Completed").

    Returns:
        pd.DataFrame: A DataFrame containing Unique ID and total change for each numeric column.
    """
   
    df = df.sort_values(by=[unique_id_col, time_col])


    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    # Group by Unique ID
    results = []
    for unique_id, group in df.groupby(unique_id_col):
        # Get the earliest and latest rows
        earliest_row = group.iloc[0]
        latest_row = group.iloc[-1]
        
        # Calculate total change
        total_change = latest_row[numeric_columns] - earliest_row[numeric_columns]
        
        # Add Unique ID and time delta
        total_change["Unique ID"] = unique_id
        total_change["Time Delta (days)"] = (latest_row[time_col] - earliest_row[time_col]).days
        
        # Append to results
        results.append(total_change)

 
    total_changes_df = pd.DataFrame(results)


    cols = ["Unique ID"] + [col for col in total_changes_df.columns if col != "Unique ID"]
    total_changes_df = total_changes_df[cols]

    return total_changes_df

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


        total_change_df = calculate_total_change(combined_data, "Unique ID", "Completed")
        st.write("**Total Change for Each Unique ID:**")
        st.dataframe(total_change_df)

        csv = total_change_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Total Change Data",
            data=csv,
            file_name="total_change_data.csv",
            mime="text/csv",
        )
