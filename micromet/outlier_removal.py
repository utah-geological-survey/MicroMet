import pandas as pd
import numpy as np
from typing import Union, List, Dict
from datetime import datetime


def detect_extreme_variations(
        df: pd.DataFrame,
        fields: Union[str, List[str]] = None,
        frequency: str = 'D',
        variation_threshold: float = 3.0,
        null_value: Union[float, int] = -9999,
        min_periods: int = 2
) -> Dict[str, pd.DataFrame]:
    """
    Detect extreme variations in specified fields of a datetime-indexed DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with datetime index
    fields : str or list of str, optional
        Column names to analyze. If None, analyzes all numeric columns
    frequency : str, default 'D'
        Frequency to analyze variations over ('D' for daily, 'h' for hourly, etc.)
    variation_threshold : float, default 3.0
        Number of standard deviations beyond which a variation is considered extreme
    null_value : float or int, default -9999
        Value to be treated as null
    min_periods : int, default 2
        Minimum number of valid observations required to calculate variation

    Returns:
    --------
    dict
        Dictionary containing:
        - 'variations': DataFrame with calculated variations
        - 'extreme_points': DataFrame with flagged extreme variations
        - 'summary': DataFrame with summary statistics
    """
    # Validate input DataFrame
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index")

    # Create copy of DataFrame
    df_copy = df.copy()

    # Replace null values
    df_copy = df_copy.replace(null_value, np.nan)

    # Select fields to analyze
    if fields is None:
        fields = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(fields, str):
        fields = [fields]

    # Initialize results
    variations = pd.DataFrame(index=df_copy.index)
    extreme_points = pd.DataFrame(index=df_copy.index)
    summary_stats = []

    # Calculate variations and detect extremes for each field
    for field in fields:
        # Group by frequency and calculate statistics
        grouped = df_copy[field].groupby(pd.Grouper(freq=frequency))

        # Calculate variation metrics
        field_var = f"{field}_variation"
        variations[field_var] = grouped.transform(lambda x: np.abs(x - x.mean()) / x.std()
        if len(x.dropna()) >= min_periods else np.nan)

        # Flag extreme variations
        extreme_points[f"{field}_extreme"] = variations[field_var] > variation_threshold

        # Calculate summary statistics
        field_summary = {
            'field': field,
            'total_observations': len(df_copy[field].dropna()),
            'extreme_variations': extreme_points[f"{field}_extreme"].sum(),
            'mean_variation': variations[field_var].mean(),
            'max_variation': variations[field_var].max(),
            'std_variation': variations[field_var].std()
        }
        summary_stats.append(field_summary)

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_stats)

    return {
        'variations': variations,
        'extreme_points': extreme_points,
        'summary': summary_df
    }


def clean_extreme_variations(
        df: pd.DataFrame,
        fields: Union[str, List[str]] = None,
        frequency: str = 'D',
        variation_threshold: float = 3.0,
        null_value: Union[float, int] = -9999,
        min_periods: int = 2,
        replacement_method: str = 'nan'
) -> Dict[str, Union[pd.DataFrame, pd.DataFrame]]:
    """
    Clean extreme variations from specified fields in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with datetime index
    fields : str or list of str, optional
        Column names to clean. If None, processes all numeric columns
    frequency : str, default 'D'
        Frequency to analyze variations over ('D' for daily, 'H' for hourly, etc.)
    variation_threshold : float, default 3.0
        Number of standard deviations beyond which a variation is considered extreme
    null_value : float or int, default -9999
        Value to be treated as null
    min_periods : int, default 2
        Minimum number of valid observations required to calculate variation
    replacement_method : str, default 'nan'
        Method to handle extreme values:
        - 'nan': Replace with NaN
        - 'interpolate': Linear interpolation
        - 'mean': Replace with frequency mean
        - 'median': Replace with frequency median

    Returns:
    --------
    dict
        Dictionary containing:
        - 'cleaned_data': DataFrame with cleaned data
        - 'cleaning_summary': DataFrame summarizing the cleaning process
        - 'removed_points': DataFrame containing the removed values
    """
    # Validate replacement method
    valid_methods = ['nan', 'interpolate', 'mean', 'median']
    if replacement_method not in valid_methods:
        raise ValueError(f"replacement_method must be one of {valid_methods}")

    # Detect extreme variations
    variation_results = detect_extreme_variations(
        df=df,
        fields=fields,
        frequency=frequency,
        variation_threshold=variation_threshold,
        null_value=null_value,
        min_periods=min_periods
    )

    # Create copy of input DataFrame
    cleaned_df = df.copy()

    # Initialize summary statistics
    cleaning_summary = []
    removed_points = pd.DataFrame(index=df.index)

    # Process each field
    if fields is None:
        fields = df.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(fields, str):
        fields = [fields]

    for field in fields:
        # Get extreme points for this field
        extreme_mask = variation_results['extreme_points'][f"{field}_extreme"]

        # Store removed values
        removed_points[field] = np.where(extreme_mask, cleaned_df[field], np.nan)

        # Apply replacement method
        if replacement_method == 'nan':
            cleaned_df.loc[extreme_mask, field] = np.nan

        elif replacement_method == 'interpolate':
            temp_series = cleaned_df[field].copy()
            temp_series[extreme_mask] = np.nan
            cleaned_df[field] = temp_series.interpolate(method='time')

        elif replacement_method in ['mean', 'median']:
            grouped = cleaned_df[field].groupby(pd.Grouper(freq=frequency))
            if replacement_method == 'mean':
                replacements = grouped.transform('mean')
            else:
                replacements = grouped.transform('median')
            cleaned_df.loc[extreme_mask, field] = replacements[extreme_mask]

        # Calculate cleaning summary
        cleaning_stats = {
            'field': field,
            'points_removed': extreme_mask.sum(),
            'percent_removed': (extreme_mask.sum() / len(df)) * 100,
            'replacement_method': replacement_method
        }
        cleaning_summary.append(cleaning_stats)

    return cleaned_df[fields]

def replace_flat_values(data, column_name,
                        flat_threshold=0.01,
                        window_size=10,
                        replacement_value=np.nan,
                        null_value=-9999,
                        inplace=False):
    """
    Detects and replaces flat-line anomalies in a time series.

    Parameters:
        data (pd.DataFrame): DataFrame containing the time series.
        column_name (str): Column name with the time series values.
        flat_threshold (float): Minimum change to consider not flat.
        window_size (int): Number of consecutive points to check for flatness.
        replacement_value (float or int): Value to replace anomalies with (e.g., NaN or -9999).

    Returns:
        pd.DataFrame: Updated DataFrame with anomalies replaced.
    """
    if not inplace:
        df = data.copy()
    else:
        df = data

    # Treat -9999 as NaN
    df[column_name] = df[column_name].replace(null_value, np.nan)

    # Compute rolling difference
    df['rolling_diff'] = df[column_name].diff().abs()

    # Flag flat lines
    df['is_flat'] = (
        df['rolling_diff'].rolling(window=window_size, min_periods=1,center=True).max() <= flat_threshold
    )

    # Replace flat-line anomalies with the specified replacement value
    df.loc[df['is_flat'], column_name] = replacement_value

    # Drop helper columns
    df.drop(columns=['rolling_diff', 'is_flat'], inplace=True)

    return df[column_name]


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    np.random.seed(42)

    sample_data = pd.DataFrame({
        'temperature': np.random.normal(20, 5, len(dates)) + \
                       10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24),  # Daily cycle
        'humidity': np.random.normal(60, 10, len(dates)),
        'pressure': np.random.normal(1013, 2, len(dates))
    }, index=dates)

    # Add some extreme variations
    sample_data.loc['2024-01-15', 'temperature'] = 45  # Extreme temperature
    sample_data.loc['2024-01-20', 'humidity'] = -9999  # Null value

    # Clean extreme variations
    cleaning_results = clean_extreme_variations(
        df=sample_data,
        fields=['temperature', 'humidity', 'pressure'],
        frequency='D',
        variation_threshold=3.0,
        null_value=-9999,
        replacement_method='interpolate'
    )

    # Print results
    print("\nCleaning Summary:")
    print(cleaning_results['cleaning_summary'])

    print("\nRemoved Points:")
    removed = cleaning_results['removed_points'].dropna(how='all')
    print(removed)