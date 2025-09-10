import numpy as np

def handle_nans(df):
    """
    Handle NaN values using linear interpolation (Requirement b)
    
    This function addresses missing data which is common in stock data due to:
    - Market holidays when exchanges are closed
    - Data feed interruptions
    - Newly listed stocks with limited history
    
    Linear interpolation is chosen because:
    - Stock prices have temporal continuity
    - Forward/backward fill preserves realistic price movements
    - More robust than dropping rows which loses valuable data
    """
    if df.isnull().any().any():
        print("NaN values detected, applying linear interpolation")
        # Only interpolate numerical columns to avoid issues with date/string columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(
            method='linear',           # Linear interpolation between known values
            limit_direction='both'     # Fill gaps in both forward and backward directions
        )
        print(f"Interpolated NaN values in columns: {numeric_cols.tolist()}")
    else:
        print("No NaN values detected")
    return df