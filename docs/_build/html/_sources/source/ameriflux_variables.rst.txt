AmeriFlux Data Variables
========================

This page outlines standard AmeriFlux variable naming conventions for flux and meteorological data.

Timestamp Variables
-------------------
- ``TIMESTAMP_START``: Start of the measurement interval (e.g., ``202407010130``)
- ``TIMESTAMP_END``: End of the interval

Core Base Names
---------------
- ``LE``: Latent Heat Flux (W m⁻²)
- ``H``: Sensible Heat Flux (W m⁻²)
- ``FC``: CO₂ Flux (µmolCO₂ m⁻² s⁻¹)
- ``TA``: Air Temperature (°C)
- ``RH``: Relative Humidity (%)
- ``SW_IN``: Incoming Shortwave Radiation (W m⁻²)
- ``SW_OUT``: Outgoing Shortwave Radiation (W m⁻²)
- ``LW_IN``: Incoming Longwave Radiation (W m⁻²)
- ``LW_OUT``: Outgoing Longwave Radiation (W m⁻²)
- ``G``: Soil Heat Flux (W m⁻²)
- ``WS``: Wind Speed (m s⁻¹)
- ``WD``: Wind Direction (°)

Qualifier Structure
-------------------
Variables may include a **positional qualifier** in the form ``_H_V_R``:
- H = horizontal position
- V = vertical level (1 = top)
- R = replicate number

Example: ``TA_2_1_1`` refers to Air Temperature at horizontal position 2, vertical level 1, replicate 1.
