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
    
    # Robust column matching
    def match_columns(assessment_questions):
        matched_cols = []
        for base_question in assessment_questions:
            # Normalize base question
            base_norm = base_question.lower().replace('?', '').strip()
            
            # Find matching columns with at least one non-null value
            matching_cols = [
                col for col in df.columns 
                if (base_norm in re.sub(r'\.\d+$', '', col).lower().replace('?', '').strip()) 
                and df[col].notna().any()
            ]
            
            # If no match found, print debug info
            if not matching_cols:
                print(f"No match found for: {base_question}")
                return []
            
            matched_cols.append(matching_cols[0])
        
        return matched_cols
    
    # Calculate scores only if ALL required questions are present
    # GAD-7 Processing
    gad7_cols = match_columns(gad7_questions)
    if len(gad7_cols) == len(gad7_questions):
        for col in gad7_cols:
            df_with_scores[col] = pd.to_numeric(df_with_scores[col], errors='coerce')
        
        # Calculate GAD-7 total score
        df_with_scores['GAD7_Total_Score'] = df_with_scores[gad7_cols].sum(axis=1)
        
        # Severity categorization
        def categorize_gad7_severity(score):
            if score < 5:
                return 'Minimal Anxiety'
            elif score < 10:
                return 'Mild Anxiety'
            elif score < 15:
                return 'Moderate Anxiety'
            else:
                return 'Severe Anxiety'
        
        df_with_scores['GAD7_Severity'] = df_with_scores['GAD7_Total_Score'].apply(categorize_gad7_severity)
    
    # SMHAT Processing
    smhat_cols = match_columns(smhat_questions)
    if len(smhat_cols) == len(smhat_questions):
        for col in smhat_cols:
            df_with_scores[col] = pd.to_numeric(df_with_scores[col], errors='coerce')
        
        # Calculate SMHAT total score
        df_with_scores['SMHAT_Total_Score'] = df_with_scores[smhat_cols].sum(axis=1)
    
    return df_with_scores
def preprocess_data(df):
    """
    Enhanced preprocessing function that includes assessment score calculation.
    """
    # Existing preprocessing
    df.columns = df.columns.str.strip().str.replace("\xa0", "")
    df["Completed"] = pd.to_datetime(df["Completed"].str.replace("EST", "").str.strip(), errors="coerce")
    
    text_columns = [
        "9. Which of these topic areas would you like to explore with Koomba’s Care Team? (Please select all that apply)",
    ]
    
    # Convert numeric values
    for col in df.columns:
        if col not in text_columns and col not in ["Team", "Unique ID", "Client", "Client's Provider", "Completed"]:
            df[col] = df[col].apply(extract_numeric_value)
    
    # Calculate assessment scores
    df = calculate_assessment_scores(df)
    
    return df.dropna(subset=["Unique ID", "Completed"])