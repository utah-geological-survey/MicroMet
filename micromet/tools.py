import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from datetime import datetime
import plotly.graph_objects as go


def find_irr_dates(df,
                   swc_col="SWC_1_1_1",
                   do_plot=False,
                   dist=20,
                   height=30,
                   prom=0.6
):
    """
    Finds irrigation dates within a DataFrame.

    :param df: A pandas DataFrame containing the data.
    :param swc_col: String. The column name in 'df' containing the soil water content data. Should be in units of percent and not a decimal; Default is 'SWC_1_1_1'.
    :param do_plot: Boolean. Whether to plot the irrigation dates on a graph. Default is False.
    :param dist: Integer. The minimum number of time steps between peaks in 'swc_col'. Default is 20.
    :param height: Integer. The minimum height (vertical distance) of the peaks in 'swc_col'. Default is 30(%).
    :param prom: Float. The minimum prominence of the peaks in 'swc_col'. Default is 0.6.

    :return: A tuple containing the irrigation dates and the corresponding soil water content values.
    """
    df_irr_season = df[df.index.month.isin([4, 5, 6, 7, 8, 9, 10])]
    peaks, _ = find_peaks(
        df_irr_season[swc_col], distance=dist, height=height, prominence=(prom, None)
    )
    dates_of_irr = df_irr_season.iloc[peaks].index
    swc_during_irr = df_irr_season[swc_col].iloc[peaks]
    if do_plot:
        plt.plot(df.index, df[swc_col])
        plt.plot(dates_of_irr, swc_during_irr, "x")
        plt.show()
    return dates_of_irr, swc_during_irr


def find_gaps(df, columns, missing_value=-9999, min_gap_periods=1):
    """
    Find gaps in time series data where values are either NaN or equal to missing_value
    for longer than min_gap_periods.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with a time series index of regular frequency
    columns : str or list of str
        Column(s) to check for gaps
    missing_value : numeric, default -9999
        Value to consider as missing data alongside NaN
    min_gap_periods : int, default 1
        Minimum number of consecutive missing periods to be considered a gap

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing gap information with columns:
        - gap_start: start datetime of gap
        - gap_end: end datetime of gap
        - duration_hours: duration of gap in hours
        - missing_records: number of missing records in gap
        - column: name of column where gap was found
    """
    if isinstance(columns, str):
        columns = [columns]

    # Initialize list to store gap information
    gaps = []

    for col in columns:
        # Create boolean mask for missing values
        is_missing = df[col].isna() | (df[col] == missing_value)

        # Get the frequency of the time series as a pandas timedelta
        freq = pd.tseries.frequencies.to_offset(pd.infer_freq(df.index))

        # Find runs of missing values
        missing_runs = (is_missing != is_missing.shift()).cumsum()[is_missing]

        if len(missing_runs) == 0:
            continue

        # Group consecutive missing values
        for run_id in missing_runs.unique():
            run_mask = missing_runs == run_id
            run_indices = missing_runs[run_mask].index

            # Only consider runs longer than min_gap_periods
            if len(run_indices) > min_gap_periods:
                gap_start = run_indices[0]
                gap_end = run_indices[-1]

                # Calculate duration in hours
                duration_hours = (gap_end - gap_start).total_seconds() / 3600

                gaps.append({
                    'gap_start': gap_start,
                    'gap_end': gap_end,
                    'duration_hours': duration_hours,
                    'missing_records': len(run_indices),
                    'column': col
                })

    if not gaps:
        return pd.DataFrame(columns=['gap_start', 'gap_end', 'duration_hours',
                                     'missing_records', 'column'])

    return pd.DataFrame(gaps).sort_values('gap_start').reset_index(drop=True)


def plot_gaps(gaps_df, title="Time Series Data Gaps"):
    """
    Create a Gantt chart visualization of gaps in time series data.

    Parameters:
    -----------
    gaps_df : pandas.DataFrame
        DataFrame containing gap information as returned by find_gaps()
    title : str, default "Time Series Data Gaps"
        Title for the plot

    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive Plotly figure showing gaps as a Gantt chart
    """
    if len(gaps_df) == 0:
        print("No gaps found to plot.")
        return None

    # Create figure
    fig = go.Figure()

    # Get unique columns and assign colors
    unique_columns = gaps_df['column'].unique()
    # Define a set of colors manually
    colors = ['rgb(166,206,227)', 'rgb(31,120,180)', 'rgb(178,223,138)',
              'rgb(51,160,44)', 'rgb(251,154,153)', 'rgb(227,26,28)',
              'rgb(253,191,111)', 'rgb(255,127,0)', 'rgb(202,178,214)']
    # Cycle through colors if more variables than colors
    color_map = dict(zip(unique_columns, [colors[i % len(colors)] for i in range(len(unique_columns))]))

    # Add gaps as horizontal bars
    for idx, row in gaps_df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['duration_hours']],
            y=[row['column']],
            orientation='h',
            base=[(row['gap_start'] - pd.Timestamp.min).total_seconds() / 3600],
            marker_color=color_map[row['column']],
            name=row['column'],
            showlegend=False,
            hovertemplate=(
                    f"Column: {row['column']}<br>" +
                    f"Start: {row['gap_start']}<br>" +
                    f"End: {row['gap_end']}<br>" +
                    f"Duration: {row['duration_hours']:.1f} hours<br>" +
                    f"Missing Records: {row['missing_records']}"
            )
        ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Variables",
        barmode='overlay',
        height=max(200, 100 * len(unique_columns)),
        showlegend=False,
        xaxis=dict(
            tickformat='%Y-%m-%d %H:%M',
            type='date'
        )
    )

    return fig


# Example usage:
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='30min')
    df = pd.DataFrame(index=dates)

    # Create multiple columns with gaps
    df['temperature'] = 20 + np.random.randn(len(dates))
    df['humidity'] = 60 + np.random.randn(len(dates))
    df['pressure'] = 1013 + np.random.randn(len(dates))

    # Insert some gaps
    df.loc['2024-01-01 10:00':'2024-01-01 12:00', 'temperature'] = -9999
    df.loc['2024-01-01 15:00':'2024-01-01 16:00', 'humidity'] = np.nan
    df.loc['2024-01-01 18:00':'2024-01-01 20:00', 'pressure'] = -9999

    # Find gaps
    gaps_df = find_gaps(df, ['temperature', 'humidity', 'pressure'], min_gap_periods=1)

    # Create and show plot
    fig = plot_gaps(gaps_df, "Sample Data Gaps")
    fig.show()
