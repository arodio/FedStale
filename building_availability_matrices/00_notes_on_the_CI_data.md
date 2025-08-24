# Notes on CI data

The 2022 Cabon Intensity (CI) data comes from *Electricity Maps*: csv files for different countries can be freely downloaded (https://www.electricitymaps.com/data-portal).
Electricity maps also proposes a paid plan providing access, through an API, to historical, real-time and **forecasted (over the next 24 hours)** data.

Description of the data:
- The granularity at which this data is available is one value per hour.
- The CI is expressed in gram of CO2 equivalents per Watt-hour, or gCO2eq/kWh.

Some information from the data source (https://www.electricitymaps.com/methodology?utm_source=app.electricitymaps.com&utm_medium=referral):
- To address missing data over a short period of time, they use the *Time Slicer Average* (TSA) estimation method. This estimation method uses available data to fill in the gap. Each missing point in the gap is filled by the average of the available data points that belong to the same time period but on different days in the given month. The estimation is then aligned in order to ensure the continuity of the estimation points with the bounds of the gap observed.
- To deal with countries that don't provide real-time data (but rather monthly/yearly aggregates delayed by periods of 1-12 months and up to more than a year), they use the *Construct Breakdown* method. This method is based on estimating a total hourly production with a simple model, calibrated to match the available aggregated data. Total production is then broken down into a production mix using a static breakdown estimated from historical aggregates.
- When only the total energy production of a country is available (but not the production mode), they apply the *Reconstruct Breakdown* estimation model. First, they compute or estimate renewable energy sources, by training (on historical data) an estimation model for renewable energy sources considering weather parameters such as wind speed, wind direction, and solar irradiation. Once these production values have been estimated, they estimate the rest of the production by applying a static breakdown over the remaining production (total - variable renewables). The breakdown is again computed by using the available historical data for each production source.
 - If the above three methods are not applicable, they also have methods specific to the country. More information is available here: https://github.com/electricitymaps/electricitymaps-contrib/wiki/Estimation-methods

