import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

def energy_sankey(df, date_text="2024-06-19 12:00"):
    """
    Create a Sankey diagram representing the energy balance for a specific date and time.

    This function generates a Sankey diagram to visualize the flow of energy in a system,
    typically used in meteorological or environmental studies. It calculates various
    energy balance components and creates a diagram showing their relationships.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing time series data with columns for different energy
        components (SW_IN, LW_IN, SW_OUT, LW_OUT, NETRAD, G, LE, H).
    date_text : str, optional
        A string representing the date and time for which to create the diagram.
        Default is "2024-06-19 12:00".

    Returns:
    --------
    plotly.graph_objs._figure.Figure
        A Plotly Figure object containing the Sankey diagram.

    Notes:
    ------
    - The function assumes that the DataFrame index is a DatetimeIndex.
    - Energy balance components are extracted from the DataFrame for the specified date.
    - The energy balance ratio (EBR) is calculated as (H + LE) / (NETRAD - G).
    - The residual term is calculated as NETRAD - (G + H + LE).
    - The Sankey diagram visualizes the flow of energy between different components.

    Energy Balance Components:
    --------------------------
    - SW_IN: Incoming Shortwave Radiation
    - LW_IN: Incoming Longwave Radiation
    - SW_OUT: Outgoing Shortwave Radiation
    - LW_OUT: Outgoing Longwave Radiation
    - NETRAD: Net Radiation
    - G: Ground Heat Flux
    - LE: Latent Heat
    - H: Sensible Heat

    Example:
    --------
    >>> import pandas as pd
    >>> import plotly.graph_objs
    >>> # Assume 'df' is a properly formatted DataFrame with energy balance data
    >>> fig = energy_sankey(df, "2024-06-19 12:00")
    >>> fig.show()

    Dependencies:
    -------------
    - pandas
    - plotly.graph_objs

    See Also:
    ---------
    plotly.graph_objs.Sankey : For more information on creating Sankey diagrams with Plotly
    """
    select_date = pd.to_datetime(date_text)
    swi = df.loc[select_date, "SW_IN"]
    lwi = df.loc[select_date, "LW_IN"]
    swo = df.loc[select_date, "SW_OUT"]
    lwo = df.loc[select_date, "LW_OUT"]
    nr = df.loc[select_date, "NETRAD"]
    shf = df.loc[select_date, "G"]
    le = df.loc[select_date, "LE"]
    h = df.loc[select_date, "H"]

    # Define the energy balance terms and their indices
    labels = [
        "Incoming Shortwave Radiation",
        "Incoming Longwave Radiation",
        "Total Incoming Radiation",
        "Outgoing Shortwave Radiation",
        "Outgoing Longwave Radiation",
        "Net Radiation",
        "Ground Heat Flux",
        "Sensible Heat",
        "Latent Heat",
        "Residual",
    ]

    print(h)
    rem = nr - (shf + h + le)

    ebr = (h + le) / (nr - shf)

    # Define the source and target nodes and the corresponding values for the energy flow
    source = [0, 1, 2, 2, 2, 5, 5, 5, 5]  # Indices of the source nodes
    target = [2, 2, 5, 3, 4, 6, 7, 8, 9]  # Indices of the target nodes

    # Define the source and target nodes and the corresponding values for the energy flow
    # source = [0, 1, 2, 2, 2, 5, 5, 5, 5]  # Indices of the source nodes
    # target = [2, 2, 5, 3, 4, 6, 7, 8, 9]  # Indices of the target nodes
    values = [lwi, swi, nr, swo, lwo, shf, h, le, rem]  # Values of the energy flow

    # Create the Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=values,
                ),
            )
        ]
    )

    # Update layout and title
    fig.update_layout(
        title_text=f"Energy Balance {ebr:0.2f} on {select_date:%Y-%m-%d}", font_size=10
    )

    # Show the figure
    # fig.show()
    return fig


def scatterplot_instrument_comparison(edmet, compare_dict, station):
    """
    Generate a scatterplot comparing two instrument measurements with regression analysis.

    This function compares two instruments specified in `compare_dict` by plotting
    their 1-hour mean values, interpolating missing data, and performing linear regression.
    It includes a 1:1 reference line and displays key regression statistics.

    Parameters
    ----------
    edmet : pandas.DataFrame
        DataFrame containing environmental measurements, indexed by datetime.
    compare_dict : dict
        Dictionary where keys are instrument column names and values are tuples of
        metadata in the format (variable, instrument label, units).
    station : str
        Identifier for the station to annotate the plot title.

    Returns
    -------
    slope : float
        Slope of the linear regression line.
    intercept : float
        Intercept of the linear regression line.
    r_squared : float
        Coefficient of determination ($R^2$) from the regression.
    p_value : float
        Two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero.
    std_err : float
        Standard error of the estimated slope.
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object containing the plot.

    Notes
    -----
    - Missing values (-9999) are replaced with NaN before processing.
    - The data is resampled to hourly frequency and linearly interpolated.
    - The regression line and 1:1 line are plotted for visual comparison.
    - X and Y axes are labeled using metadata from `compare_dict`.
    - The function both displays the plot and prints regression statistics.
    """
    # Compare two instruments
    instruments = list(compare_dict.keys())
    df = edmet[instruments].replace(-9999, np.nan).dropna()
    df = df.resample("1h").mean().interpolate(method="linear")
    df = df.dropna()

    x = df[instruments[0]]
    y = df[instruments[1]]

    xinfo = compare_dict[instruments[0]]
    yinfo = compare_dict[instruments[1]]

    # one to one line
    xline = np.arange(df.min().min(), df.max().max(), 0.1)
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    # Predict y values
    y_pred = slope * x + intercept
    # R-squared
    r_squared = r_value**2

    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot
    ax.scatter(x, y, alpha=0.5, s=1, label="Data points")
    ax.set_title(f"{xinfo[1]} Comparison: {station}")
    ax.plot(xline, xline, label="1:1 line", color="green", linestyle="--")
    ax.plot(
        x,
        y_pred,
        color="red",
        label=f"Fit: y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_squared:.3f}",
    )
    plt.legend()
    plt.grid(True)

    ax.set_xlabel(f"{xinfo[0]} {xinfo[1]} ({xinfo[2]})")
    ax.set_ylabel(f"{yinfo[0]} {yinfo[1]} ({yinfo[2]})")

    plt.show()

    # Print results
    print(f"Slope: {slope:.3f}")
    print(f"Intercept: {intercept:.3f}")
    print(f"R-squared: {r_squared:.3f}")
    return slope, intercept, r_squared, p_value, std_err, fig, ax


def bland_alt_plot(edmet, compare_dict, station, alpha=0.5):
    """
    Create a Bland–Altman plot to assess agreement between two instruments.

    This function compares two instruments specified in `compare_dict` by plotting the
    mean of their measurements against their difference. It reports the root mean square error
    (RMSE), bias, and spread, and visualizes the limits of agreement.

    Parameters
    ----------
    edmet : pandas.DataFrame
        DataFrame containing environmental measurements, indexed by datetime.
    compare_dict : dict
        Dictionary where keys are instrument column names and values are tuples of
        metadata in the format (variable, instrument label, units).
    station : str
        Name or identifier for the measurement station, used in plot titles.
    alpha : float, optional
        Transparency of the plot elements. Default is 0.5.

    Returns
    -------
    f : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    ax : matplotlib.axes.Axes
        The matplotlib axes object for the Bland–Altman plot.

    Notes
    -----
    - Missing values (-9999) are replaced with NaN before processing.
    - Data is resampled to hourly frequency and linearly interpolated.
    - Bias is calculated as the mean difference between instruments.
    - Spread is the standard deviation of the difference.
    - Limits of agreement are shown at ±1.96 × standard deviation.
    - The `statsmodels.graphics.mean_diff_plot` utility is used for plotting.
    """
    # Compare two instruments
    instruments = list(compare_dict.keys())
    df = edmet[instruments].replace(-9999, np.nan).dropna()
    df = df.resample("1h").mean().interpolate(method="linear")
    df = df.dropna()

    x = df[instruments[0]]
    y = df[instruments[1]]
    rmse = np.sqrt(mean_squared_error(x, y))
    print(f"RMSE: {rmse:.3f}")

    mean_vals = df[instruments].mean(axis=1)
    diff_vals = x - y
    bias = diff_vals.mean()
    spread = diff_vals.std()
    print(f"Bias = {bias:.3f}, Spread = {spread:.3f}")
    top = diff_vals.mean() + 1.96 * diff_vals.std()
    bottom = diff_vals.mean() - 1.96 * diff_vals.std()

    f, ax = plt.subplots(1, figsize=(8, 5), alpha=alpha)
    sm.graphics.mean_diff_plot(x, y, ax=ax)
    ax.text(
        mean_vals.mean(),
        top,
        s=compare_dict[instruments[0]][0],
        verticalalignment="top",
        fontweight="bold",
    )
    ax.text(
        mean_vals.mean(),
        bottom,
        s=compare_dict[instruments[1]][0],
        verticalalignment="bottom",
        fontweight="bold",
    )
    ax.set_title(
        f"{compare_dict[instruments[0]][0]} vs {compare_dict[instruments[1]][0]} at {station}"
    )
    ax.set_xlabel(
        f"Mean {compare_dict[instruments[0]][1]} ({compare_dict[instruments[0]][2]})"
    )
    ax.set_ylabel(
        f"Difference ({compare_dict[instruments[0]][2]})\n({compare_dict[instruments[0]][0]} - {compare_dict[instruments[1]][0]})",
        fontsize=10,
    )

    return f, ax


# Example of filtering by date range
def plot_timeseries_daterange(
    input_df, selected_station, selected_field, start_date, end_date
):
    """
    Plot a time series for a specific station and variable within a date range.

    This function extracts and visualizes a single time series from a multi-station
    DataFrame, limited to a specified date range and filtered by station ID and
    data field. It handles missing values and generates a time series plot.

    Parameters
    ----------
    input_df : pandas.DataFrame
        DataFrame containing a multi-index (station, timestamp) with time series data.
    selected_station : str
        Station identifier to select from the DataFrame.
    selected_field : str
        Column name of the variable to plot (e.g., temperature, humidity).
    start_date : str or pandas.Timestamp
        Start date of the time range to include in the plot.
    end_date : str or pandas.Timestamp
        End date of the time range to include in the plot.

    Returns
    -------
    None
        Displays a matplotlib plot but does not return any value.

    Notes
    -----
    - Assumes `input_df` is indexed by station and datetime (MultiIndex).
    - Missing values equal to -9999 are replaced with NaN prior to plotting.
    - The function sets global variables `fig` and `ax`, which may overwrite
      other plots in an interactive session.
    """
    global fig, ax
    # ax.clear()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter data by date range
    filtered_df = input_df.loc[selected_station].loc[start_date:end_date]
    filtered_df = filtered_df.loc[:, selected_field].replace(-9999, np.nan)

    # Plot each selected category
    ax.plot(filtered_df.index, filtered_df, label=selected_station, linewidth=2)

    plt.title(f"{selected_station} {selected_field}\n{start_date} to {end_date}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def save_plot(b):
    """
    Saves plot for an interactive notebook button function
    """
    # This line saves the plot as a .png file. Change it to .pdf to save as pdf.
    fig.savefig("plot.png")
