# Problem definition

Our aim is to do **energy-mix aware** cross-silo Federated Learning (FL): 
- each client is in a different location, with its own energy-mix
- the availability of the clients depends on the energy-mix of their location, in the sense that the global loss and the clients' loss will depend on the energy-mix of the clients.

## Introduction

Various FL training algorithm exist. They have been built to answer to different settings in terms of clients availability, data heterogeneity, and so on. Each comes with specific advantages and drawbacks in terms of bias and convergence of the algorithm.

The FL training is a succession of training rounds during which a certain group of clients (possibly all of them) is available to training collaboratively a global model based on their data.

Here, the avaialbility of the clients for each round will depend on their energy-mix, i.e., the carbon intensity (CI) of their location. Moreover, the datasets of the clients are not necessarily iid.

## Our approach

On the one hand, we will design appropriate availability sequences for all clients, based on the estimated future CI data for these clients. 
Each availability sequence is a sequences of 0 and 1, where each number refers to one round of the FL training, and 0 means that whe client is not available for this round while 1 means that the client is available.
On the other hand, we will choose an appropriate FL algortihm to train the ML model based on these availability sequences.


The FL training plan (choice of clients at specific times) will need to make a balance between the environmental cost, the bias, the forgetting.

**How are are going in this direction:**

We are analyzing historical CI data provided by electricity maps: graph of raw data, graph of mean data over certain period, spectral analysis, seasonal decomposition, 2 by 2 comparison of countries over certains period of time. More analysis is foreseen.

We are creating different types of availability sequences. (1) For each country, the country is unavailable if the CI is above the country mean for this time period, and is available otherwise. (2) A global threshod is chosen (the mean CI over all countries and over the time period) and the client is unavailable if the CI is above this threshold and is available otherwise. (3) For countries with a large mean CI over the time period, further increase the threshold so that these countries are only available for a certain fraction of the time period (e.g., 10% of the time).

We are testing various FL algorthms with these availability sequences, for the training of a ML model on mnist, where the clients' datasets are not iid.


<!-- ### Charasterisitics of the problem.

**About the CI data that will be available to our algorithm:**
- Is the CI time-varying or constant over time? 
- If it is time varying, what is the time-granularity of the CI data (raw hourly values, daily mean value, monthly mean value, etc.)?
- Over which period do we have access to the CI data? 
- Is the CI data deterministic or stochastic?

**About the data distribution over the clients:**
- Do the clients have the same data or a subset of the data?
- If each client has a subset of the data, what are the characteritics of this data?

**About the cost/loss:** it includes
- environmental cost of the computations
- duration cost
- environmental cost of transfering weights/gradients

**About the FL algorithm:** What FL algorithm do we use?

### To do

Questions about the data:
- Are the CI or the different countries correlated?

Ideas: 
- Seasonal-Trend decomposition using LOESS (STL) -->



## Appendix

### CI data

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

