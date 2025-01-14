import streamlit as st
import pandas as pd

# Set up the Streamlit app
st.title("Koomba Data Analysis")
st.write("Upload your CSV files to validate and clean data.")

# Upload CSV file(s)
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

# Define required columns for validation
REQUIRED_COLUMNS = ["Unique ID", "Completed", "Survey Question 1", "Survey Question 2", "Response Category"]

def validate_and_clean(file):
    # Load the CSV into a DataFrame
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
        return None
    
    # Check for missing columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        st.error(f"File {file.name} is missing columns: {missing_columns}")
        return None

    # Handle missing data
    st.write(f"Processing {file.name}...")
    # Drop rows where critical fields are missing
    df.dropna(subset=["Unique ID", "Completed"], inplace=True)
    
    # Fill missing survey responses
    for col in ["Survey Question 1", "Survey Question 2"]:
        if col in df.columns:
            df[col].fillna("Not Answered", inplace=True)
    
    # Standardize categorical responses
    if "Response Category" in df.columns:
        category_map = {
            "Never": 0,
            "Several Days": 1,
            "Most Days": 2,
            "Every Day": 3
        }
        df["Response Category"] = df["Response Category"].map(category_map).fillna(-1)

    # Format timestamps
    try:
        df["Completed"] = pd.to_datetime(df["Completed"], errors="coerce")
    except Exception as e:
        st.warning(f"Unable to parse 'Completed' timestamps in {file.name}: {e}")

    # Show cleaned data preview
    st.write("Preview of cleaned data:")
    st.dataframe(df.head())
    return df

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        cleaned_data = validate_and_clean(file)
        if cleaned_data is not None:
            all_data.append(cleaned_data)
    
    # Combine all cleaned data if multiple files are uploaded
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        st.write("Combined Data from All Files:")
        st.dataframe(combined_data)

        # Allow download of cleaned data
        csv = combined_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned Data",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
