
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def find_irr_dates(
    df, swc_col="SWC_1_1_1", do_plot=False, dist=20, height=30, prom=0.6
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
