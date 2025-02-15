import numpy as np
import datetime as dt
import datetime

# Constants
STEFAN_BOLTZMANN = 5.67e-8  # W m^-2 K^-4
SOLAR_CONSTANT = 1361  # W/m² (Extraterrestrial solar radiation)

# Reasonable net radiation ranges
DAYTIME_RANGE = (50, 1000)  # Expected range in W/m²
NIGHTTIME_RANGE = (-200, 100)  # Expected range in W/m²

def solar_declination(day_of_year):
    """Calculate the solar declination angle (in radians) using the day of the year."""
    return 23.44 * np.cos(np.radians((360 / 365) * (day_of_year + 10))) * np.pi / 180

def solar_hour_angle(longitude, time_utc):
    """Calculate the solar hour angle (in radians)."""
    # Convert time to fractional hours
    time_decimal = time_utc.hour + time_utc.minute / 60 + time_utc.second / 3600
    # Solar time correction (simplified, assumes standard meridian)
    lstm = 15 * round(longitude / 15)  # Standard meridian correction
    time_offset = 4 * (longitude - lstm)  # Minutes offset
    solar_time = time_decimal + time_offset / 60  # Adjusted time
    return np.radians(15 * (solar_time - 12))  # Convert to radians

def solar_zenith_angle(latitude, declination, hour_angle):
    """Calculate the solar zenith angle (in radians)."""
    latitude_rad = np.radians(latitude)
    return np.arccos(
        np.sin(latitude_rad) * np.sin(declination) +
        np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    )

def extraterrestrial_solar_radiation(day_of_year):
    """Calculate extraterrestrial solar radiation (W/m²)."""
    solar_constant = 1361  # W/m²
    eccentricity_correction = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
    return solar_constant * eccentricity_correction

def atmospheric_transmissivity(zenith_angle):
    """Estimate atmospheric transmissivity based on a simple empirical model."""
    if zenith_angle > np.pi / 2:  # Sun below the horizon
        return 0
    return max(0.75 * np.exp(-0.15 / np.cos(zenith_angle)), 0)  # Simple attenuation model

def incoming_solar_radiation(latitude, longitude, datetime_utc):
    """Estimate incoming solar radiation at a given latitude and datetime."""
    day_of_year = datetime_utc.timetuple().tm_yday
    declination = solar_declination(day_of_year)
    hour_angle = solar_hour_angle(longitude, datetime_utc)
    zenith_angle = solar_zenith_angle(latitude, declination, hour_angle)

    extra_terrestrial = extraterrestrial_solar_radiation(day_of_year)
    transmissivity = atmospheric_transmissivity(zenith_angle)

    return extra_terrestrial * transmissivity



def is_daytime(hour, sunrise, sunset):
    """Check if the given hour is within the daytime period."""
    return sunrise <= hour <= sunset

def estimate_clear_sky_radiation(lat, lon, timestamp):
    """
    Estimates clear-sky shortwave radiation based on solar angle and atmospheric attenuation.
    Uses a simplified solar position model.
    """
    # Approximate solar declination angle (valid for general validation purposes)
    day_of_year = timestamp.timetuple().tm_yday
    declination = 23.45 * np.sin(np.radians((360 / 365) * (day_of_year - 81)))
    
    # Approximate solar hour angle
    hour_angle = (timestamp.hour - 12) * 15  # Degrees (15° per hour from solar noon)
    
    # Approximate solar zenith angle
    latitude_rad = np.radians(lat)
    declination_rad = np.radians(declination)
    hour_angle_rad = np.radians(hour_angle)
    
    cos_theta = (np.sin(latitude_rad) * np.sin(declination_rad) +
                 np.cos(latitude_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad))
    
    if cos_theta <= 0:  # Sun is below the horizon
        return 0
    
    # Estimate atmospheric attenuation (simplified)
    air_mass = 1 / cos_theta if cos_theta > 0 else np.inf
    transmittance = np.exp(-0.1 * air_mass)  # Approximate atmospheric absorption
    
    # Compute clear-sky shortwave radiation
    clear_sky_radiation = SOLAR_CONSTANT * transmittance * cos_theta
    return clear_sky_radiation

def estimate_max_net_radiation(temp_c):
    """
    Estimates the maximum possible net radiation based on Stefan-Boltzmann law.
    """
    temp_k = temp_c + 273.15
    return STEFAN_BOLTZMANN * temp_k**4  # W/m²

def validate_net_radiation(radiation_values, timestamps, temp_values, lat, lon, sunrise=6, sunset=18):
    """
    Validate net radiation values based on expected physical ranges and additional metrics.
    
    Parameters:
    - radiation_values: List of net radiation values (W/m²)
    - timestamps: Corresponding timestamps (datetime objects)
    - temp_values: List of air temperatures (°C) at the same timestamps
    - lat: Latitude
    - lon: Longitude
    - sunrise: Hour when the sun rises (default 6 AM)
    - sunset: Hour when the sun sets (default 6 PM)
    
    Returns:
    - List of (timestamp, value, status, reason) tuples
    """
    results = []
    
    for i, value in enumerate(radiation_values):
        timestamp = timestamps[i]
        temp_c = temp_values[i]
        hour = timestamp.hour
        reason = ""
        
        # Check general expected range
        if is_daytime(hour, sunrise, sunset):
            valid_range = DAYTIME_RANGE
        else:
            valid_range = NIGHTTIME_RANGE
        
        if not (valid_range[0] <= value <= valid_range[1]):
            reason = "Outside expected range"

        # Check against estimated clear-sky shortwave radiation
        clear_sky_rad = estimate_clear_sky_radiation(lat, lon, timestamp)
        if value > clear_sky_rad + 200:  # Allow some margin for atmospheric effects
            reason += ", Exceeds clear-sky estimate"

        # Check against maximum theoretical net radiation
        max_theoretical_rad = estimate_max_net_radiation(temp_c)
        if value > max_theoretical_rad:
            reason += ", Exceeds theoretical max"

        # Assign validity status
        status = "Valid" if reason == "" else "Invalid"
        
        results.append((timestamp, value, status, reason.strip(", ")))
    
    return results



