import pandas as pd
import numpy as np


def analyze_data_types(df):
    print("=" * 80)
    print("TIPET E TË DHËNAVE (DATA TYPES)")
    print("=" * 80)
    print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    print("Data Types Summary:")
    print("-" * 80)
    data_types_summary = df.dtypes.value_counts()
    for dtype, count in data_types_summary.items():
        print(f"  {dtype}: {count} column(s)")
    
    print("\nDetailed Column Information:")
    print("-" * 80)
    for col in df.columns:
        print(f"  {col:25} -> {df[col].dtype}")
    
    print("\nColumn Descriptions:")
    print("-" * 80)
    column_descriptions = {
        'timestamp': 'Timestamp of the traffic record (object)',
        'origin': 'Starting location (object)',
        'destination': 'Destination location (object)',
        'distance_km': 'Distance in kilometers (float64)',
        'duration_normal_min': 'Normal travel duration in minutes (float64)',
        'duration_traffic_min': 'Actual travel duration with traffic in minutes (float64)',
        'delay_min': 'Delay caused by traffic in minutes (float64)',
        'hour': 'Hour of the day (int64)',
        'day_of_week': 'Day of week (0=Monday, 6=Sunday) (int64)',
        'is_weekend': 'Binary indicator if weekend (int64)',
        'temperature': 'Temperature in Celsius (float64)',
        'wind': 'Wind speed in km/h (float64)',
        'rain': 'Rainfall amount (float64)'
    }
    
    for col, desc in column_descriptions.items():
        print(f"  {col:25} - {desc}")


def analyze_data_quality(df):
    """
    Task 2: Analyze and assess data quality
    """
    print("\n\n")
    print("=" * 80)
    print("TASK 2: KUALITETI I TË DHËNAVE (DATA QUALITY)")
    print("=" * 80)
    
    data_types_summary = df.dtypes.value_counts()
    
    print("\n1. MISSING VALUES (NULL VALUES):")
    print("-" * 80)
    missing = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    for col in df.columns:
        null_count = missing[col]
        null_pct = missing_percentage[col]
        if null_count > 0:
            print(f"  {col:25} - {null_count:6} missing values ({null_pct:.2f}%)")
        else:
            print(f"  {col:25} - OK (no missing values)")
    
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    print(f"\n  Total missing values: {total_missing} out of {total_cells} cells ({(total_missing/total_cells)*100:.3f}%)")
    
    print("\n2. DUPLICATE ROWS:")
    print("-" * 80)
    duplicate_count = df.duplicated().sum()
    print(f"  Total duplicate rows: {duplicate_count}")
    if duplicate_count == 0:
        print("  Status: ✓ Dataset is clean (no duplicates)")
    
    print("\n3. DATA COMPLETENESS:")
    print("-" * 80)
    completeness = ((len(df) - df.isnull().sum()) / len(df)) * 100
    avg_completeness = completeness.mean()
    print(f"  Average completeness: {avg_completeness:.2f}%")
    for col in df.columns:
        pct = completeness[col]
        status = "✓" if pct == 100 else "⚠"
        print(f"  {status} {col:25} - {pct:.2f}% complete")
    
    print("\n4. UNIQUE VALUES:")
    print("-" * 80)
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_pct = (unique_count / len(df)) * 100
        print(f"  {col:25} - {unique_count:6} unique values ({unique_pct:.2f}% cardinality)")
    
    print("\n5. STATISTICAL SUMMARY (Numerical Columns):")
    print("-" * 80)
    print(df.describe().to_string())
    
    print("\n6. QUALITY ASSESSMENT SUMMARY:")
    print("-" * 80)
    quality_score = 100 - ((total_missing / total_cells) * 100)
    print(f"  Overall Data Quality Score: {quality_score:.2f}%")
    print(f"  Status: {'✓ EXCELLENT' if quality_score > 99 else '✓ GOOD' if quality_score > 95 else '⚠ ACCEPTABLE'}")
    print(f"\n  Summary:")
    print(f"    • Records: {len(df):,}")
    print(f"    • Columns: {len(df.columns)}")
    print(f"    • Missing values: {total_missing} ({(total_missing/total_cells)*100:.3f}%)")
    print(f"    • Duplicates: {duplicate_count}")
    print(f"    • Data types: {df.dtypes.nunique()} types ({data_types_summary.to_dict()})")


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('traffic_dataset.csv')
    
    # Perform analyses
    analyze_data_types(df)
    analyze_data_quality(df)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
