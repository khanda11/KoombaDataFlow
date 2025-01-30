import pandas as pd
import re
from plotly.subplots import make_subplots
import plotly.graph_objects as go
view_type = "By Date"

def extract_numeric_value(s):
    if isinstance(s, str):
        match = re.search(r"([-+]?\d+(\.\d+)?)", s)
        return float(match.group(1)) if match else None
    return s if isinstance(s, (int, float)) else None

def get_gad7_questions():
    """Return the exact column names for GAD-7 questions."""
    return [
        "Feeling Nervous, Anxious, or on Edge?",
        "Not Being Able to Stop or Control Worrying?",
        "Worrying Too Much About Different Things?",
        "Trouble Relaxing?",
        "Being So Restless That It Is Hard To Sit Still?",
        "Becoming Easily Annoyed or Irritable?",
        "Feeling Afraid, As If Something Awful Might Happen?"
    ]

def get_smhat_questions():
    """Return the exact column names for SMHAT questions."""
    return [
        "It was difficult to be around teammates.",
        "I found it difficult to do what I needed to do.",
        "I was less motivated.",
        "I was irritable, angry, or aggressive",
        "I could not stop worrying about injury or my performance.",
        "I found training more stressful.",
        "I found it hard to cope with selection pressures.",
        "I worried about life after sport.",
        "I needed alcohol or other substances to relax.",
        "I took unusual risks off-field."
    ]

def calculate_assessment_scores(df):
    """
    Calculate total GAD-7 and SMHAT scores for each response.
    Returns DataFrame with new columns for total scores and severity levels.
    """
    df_with_scores = df.copy()
    
    # Get questions for each assessment
    gad7_questions = get_gad7_questions()
    smhat_questions = get_smhat_questions()
    
    # Find matching columns in the dataframe using flexible matching
    gad7_cols = []
    smhat_cols = []
    
    # More robust column matching that handles suffixes
    for col in df.columns:
        # Remove any .1, .2 etc suffixes for matching
        base_col = re.sub(r'\.\d+$', '', col)
        
        # Convert to lowercase for case-insensitive matching
        base_col_lower = base_col.lower()
        
        # Match GAD-7 questions
        for q in gad7_questions:
            if q.lower() == base_col_lower:
                gad7_cols.append(col)
                break
                
        # Match SMHAT questions
        for q in smhat_questions:
            if q.lower() == base_col_lower:
                smhat_cols.append(col)
                break
    
    # Group columns by their base names (without suffixes)
    gad7_base_cols = {}
    for col in gad7_cols:
        base_name = re.sub(r'\.\d+$', '', col)
        if base_name not in gad7_base_cols:
            gad7_base_cols[base_name] = []
        gad7_base_cols[base_name].append(col)
    
    smhat_base_cols = {}
    for col in smhat_cols:
        base_name = re.sub(r'\.\d+$', '', col)
        if base_name not in smhat_base_cols:
            smhat_base_cols[base_name] = []
        smhat_base_cols[base_name].append(col)
    
    # Calculate scores for each set of columns
    for cols in gad7_base_cols.values():
        # Convert columns to numeric
        for col in cols:
            df_with_scores[col] = pd.to_numeric(df_with_scores[col], errors='coerce')
            
            # Calculate GAD-7 total score for this column set
            if len(cols) > 0:
                score_col = f"GAD7_Total_Score{'.1' if '.' in col else ''}"
                df_with_scores.loc[df_with_scores[col].notna(), score_col] = df_with_scores[cols].sum(axis=1)
                
                
    # Do the same for SMHAT
    for cols in smhat_base_cols.values():
        for col in cols:
            df_with_scores[col] = pd.to_numeric(df_with_scores[col], errors='coerce')
            
            # Calculate SMHAT total score for this column set
            if len(cols) > 0:
                score_col = f"SMHAT_Total_Score{'.1' if '.' in col else ''}"
                df_with_scores.loc[df_with_scores[col].notna(), score_col] = df_with_scores[cols].sum(axis=1)
    
    # Debug information
    print(f"Found GAD-7 column sets: {gad7_base_cols}")
    print(f"Found SMHAT column sets: {smhat_base_cols}")
    
    return df_with_scores
def preprocess_data(df):
    """
    Enhanced preprocessing function that includes assessment score calculation.
    """
    # Existing preprocessing
    df.columns = df.columns.str.strip().str.replace("\xa0", "")
    df["Completed"] = pd.to_datetime(df["Completed"].str.replace("EST", "").str.strip(), errors="coerce")
    
    text_columns = [
        "9. Which of these topic areas would you like to explore with Koomba's Care Team? (Please select all that apply)",
    ]
    
    # Convert numeric values
    for col in df.columns:
        if col not in text_columns and col not in ["Team", "Unique ID", "Client", "Client's Provider", "Completed"]:
            df[col] = df[col].apply(extract_numeric_value)
    
    # Calculate assessment scores
    df = calculate_assessment_scores(df)
    
    return df.dropna(subset=["Unique ID", "Completed"])