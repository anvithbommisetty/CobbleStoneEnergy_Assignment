import numpy as np

'''
This module provides functions to simulate a synthetic price series with concept drift, seasonality, and anomalies.
Functions:
- concept_drift(t): Determines the expected return (mu) and volatility (sigma) based on the phase of the time period.
- add_anomalies(price): Introduces anomalies into a given price series at random intervals.
- generate_price_series(): Generates a synthetic price series with seasonality, concept drift, and anomalies.
Functions:
    concept_drift(t):
    add_anomalies(price):
    generate_price_series():
'''


# Concept drift: Sudden changes in drift and volatility
def concept_drift(t):
    """
    Determines the expected return (mu) and volatility (sigma) based on the phase of the time period.

    Parameters:
    t (float): A time value between 0 and 1 representing the phase of the period.

    Returns:
    tuple: A tuple containing the expected return (mu) and volatility (sigma).

    Phases:
    - Initial phase (t < 0.3): Low expected return and low volatility.
    - Mid phase (0.3 <= t < 0.6): Higher expected return and higher volatility.
    - Later phase (t >= 0.6): Moderate expected return and moderate volatility.
    """
    if t < 0.3:   # Initial phase
        mu = 0.03  # Low expected return
        sigma = 0.15  # Low volatility
    elif t < 0.6:  # Mid phase (higher drift)
        mu = 0.07
        sigma = 0.25
    else:         # Later phase (moderate drift)
        mu = 0.04
        sigma = 0.18
    return mu, sigma

# Introduce anomalies randomly every 25-30 points
def add_anomalies(price):
    """
    Introduces anomalies into a given price series.
    This function modifies the input price series by introducing anomalies at random intervals.
    Anomalies can be either spikes or drops in the price, with a random magnitude between 0.2 and 0.7.
    The intervals between anomalies are randomly chosen between 25 and 30 time points.
    Parameters:
    price (list or numpy array): The input price series to be modified.
    Returns:
    list or numpy array: The modified price series with anomalies introduced.
    """
    num_points = len(price)
    next_anomaly = np.random.randint(25, 30)  # Random interval between 25 and 30

    for t in range(1, num_points):
        if t == next_anomaly:
            anomaly_magnitude = 0.2 + np.random.rand()*0.5  # Random magnitude between 0.2 and 0.7
            if np.random.rand() < 0.5:  # 50% chance for spike
                price[t] *= (1 + anomaly_magnitude)  # Spike
            else:
                price[t] *= (1 - anomaly_magnitude)  # Drop
            next_anomaly += np.random.randint(25, 30)

    return price

# Simulating price series with concept drift, seasonality, and anomalies
def generate_price_series():
    """
    Generates a synthetic price series with seasonality, concept drift, and anomalies.

    The function simulates a price series over time using a geometric Brownian motion model 
    with added seasonal components and concept drift. Anomalies are also introduced to the 
    generated price series.

    Returns:
        tuple: A tuple containing:
            - price (numpy.ndarray): The generated price series.
            - time (numpy.ndarray): The time vector corresponding to the price series.
    """
    # Parameters for the simulation
    T = 5.0        # Time in years
    dt = 1/252     # Time step (daily, assuming 252 trading days per year)
    N = int(T/dt)  # Number of time steps
    S0 = 100       # Initial stock price
    seasonality_amplitude = 0.02  # Amplitude of seasonality
    price = np.zeros(N)
    time = np.linspace(0, T, N)
    price[0] = S0
    # Random shocks for each time step
    random_shocks = np.random.normal(0, 1, N)
    for t in range(1, N):
        seasonal_component = seasonality_amplitude * np.sin(2 * np.pi * time[t] * 4)  # 4 cycles in a year
        mu_t, sigma_t = concept_drift(time[t])
        price[t] = price[t-1] * np.exp((mu_t - 0.5 * sigma_t**2) * dt + seasonal_component + sigma_t * np.sqrt(dt) * random_shocks[t])  
    # add anomalies   
    price = add_anomalies(price)
    return price,time