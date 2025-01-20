import pandas as pd
import re

def extract_numeric_value(s):
    if isinstance(s, str):
        match = re.search(r"([-+]?\d+(\.\d+)?)", s)
        return float(match.group(1)) if match else None
    return s if isinstance(s, (int, float)) else None

def preprocess_data(df):
    df.columns = df.columns.str.strip().str.replace("\xa0", "")
    df["Completed"] = pd.to_datetime(df["Completed"].str.replace("EST", "").str.strip(), errors="coerce")

    for col in df.columns:
        if col not in ["Team", "Unique ID", "Client", "Client's Provider", "Completed"]:
            df[col] = df[col].apply(extract_numeric_value)

    return df.dropna(subset=["Unique ID", "Completed"])
