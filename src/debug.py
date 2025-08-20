import pandas as pd

# Debug the date arithmetic issue

file_path = '/data/projects/punim0401/Arpan/medical_chatbot_v3/data/structured/INSPIRE_cleaned.csv'
df = pd.read_csv(file_path, low_memory=False)
# Step 1: Check what types your date columns actually contain
print("=== DATA TYPE INVESTIGATION ===")
onset_sample = df["Onset Date"].dropna().head(5)
arrival_sample = df["Date of Arrival at Receiving Hospital"].dropna().head(5)

print("Onset Date samples:")
for i, val in enumerate(onset_sample):
    print(f"  {i}: {val} (type: {type(val)})")

print("\nArrival Date samples:")
for i, val in enumerate(arrival_sample):
    print(f"  {i}: {val} (type: {type(val)})")

# Step 2: Test the filtering
valid_rows = df[(df["Onset Date"].notna()) & (df["Date of Arrival at Receiving Hospital"].notna())]
print(f"\nValid rows count: {len(valid_rows)}")

if len(valid_rows) > 0:
    # Step 3: Check a few sample calculations
    print("\n=== SAMPLE CALCULATIONS ===")
    sample_data = valid_rows[["Onset Date", "Date of Arrival at Receiving Hospital"]].head(5)
    
    for idx, row in sample_data.iterrows():
        onset = row["Onset Date"]
        arrival = row["Date of Arrival at Receiving Hospital"]
        
        print(f"\nRow {idx}:")
        print(f"  Onset: {onset} (type: {type(onset)})")
        print(f"  Arrival: {arrival} (type: {type(arrival)})")
        
        # Try direct subtraction (for date objects)
        try:
            if hasattr(onset, 'year') and hasattr(arrival, 'year'):  # Both are date objects
                diff_direct = arrival - onset
                print(f"  Direct subtraction: {diff_direct} (type: {type(diff_direct)})")
                print(f"  Days difference: {diff_direct.days}")
                print(f"  Seconds difference: {diff_direct.total_seconds()}")
            else:
                print("  Not date objects, trying pd.to_datetime conversion...")
                onset_dt = pd.to_datetime(onset)
                arrival_dt = pd.to_datetime(arrival)
                diff_pd = arrival_dt - onset_dt
                print(f"  pd.to_datetime difference: {diff_pd}")
                print(f"  Seconds: {diff_pd.total_seconds()}")
        except Exception as e:
            print(f"  Error in calculation: {e}")

# Step 4: Corrected calculation approach
print("\n=== CORRECTED CALCULATION ===")
try:
    # If your columns contain date objects, use direct arithmetic
    result_df = df[(df["Onset Date"].notna()) & (df["Date of Arrival at Receiving Hospital"].notna())].copy()
    
    # Method 1: Direct subtraction (if date objects)
    result_df['Time Difference'] = result_df['Date of Arrival at Receiving Hospital'] - result_df['Onset Date']
    
    # Convert timedelta to seconds
    result_df['Time Difference Seconds'] = result_df['Time Difference'].apply(
        lambda x: x.total_seconds() if hasattr(x, 'total_seconds') else None
    )
    
    median_seconds = result_df['Time Difference Seconds'].median()
    median_days = median_seconds / (24 * 3600) if median_seconds is not None else None
    median_hours = median_seconds / 3600 if median_seconds is not None else None
    
    print(f"Median time difference:")
    print(f"  Seconds: {median_seconds}")
    print(f"  Hours: {median_hours}")
    print(f"  Days: {median_days}")
    
except Exception as e:
    print(f"Error in corrected calculation: {e}")
    
    # Method 2: Force conversion to datetime
    try:
        print("\nTrying forced datetime conversion...")
        result_df = df[(df["Onset Date"].notna()) & (df["Date of Arrival at Receiving Hospital"].notna())].copy()
        result_df['Onset_DT'] = pd.to_datetime(result_df['Onset Date'], errors='coerce')
        result_df['Arrival_DT'] = pd.to_datetime(result_df['Date of Arrival at Receiving Hospital'], errors='coerce')
        result_df['Time_Diff'] = result_df['Arrival_DT'] - result_df['Onset_DT']
        result_df['Seconds'] = result_df['Time_Diff'].dt.total_seconds()
        
        median_seconds_v2 = result_df['Seconds'].median()
        print(f"Median (method 2): {median_seconds_v2} seconds")
        print(f"Median (method 2): {median_seconds_v2 / 3600} hours")
        
    except Exception as e2:
        print(f"Method 2 also failed: {e2}")