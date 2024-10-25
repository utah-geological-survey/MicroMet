import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        "Residual"
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
    fig = go.Figure(data=[go.Sankey(
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
        )
    )])

    # Update layout and title
    fig.update_layout(title_text=f"Energy Balance {ebr:0.2f} on {select_date:%Y-%m-%d}", font_size=10)

    # Show the figure
    # fig.show()
    return fig

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    CI_low    = md - 1.96*sd
    CI_high   = md + 1.96*sd

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='black', linestyle='-')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    return md, sd, mean, CI_low, CI_high
