import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import logging

# Import settings for constants like START_DATE, END_DATE, HOLIDAYS
from config.settings import START_DATE, END_DATE, HOLIDAYS

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_sales_data(
    base_sales: int,
    weekly_peak_factor: float,
    summer_factor: float,
    winter_factor: float,
    spike_probability: float,
    spike_multiplier: float,
    seed: int = 42,
    noise_level: float = 0.1
) -> pd.DataFrame:
    """
    Generates synthetic sales data with various patterns and noise.

    Args:
        base_sales (int): Average daily sales.
        weekly_peak_factor (float): Multiplier for weekend sales (e.g., 1.2 for 20% increase).
        summer_factor (float): Multiplier for sales during summer months (June, July, Aug).
        winter_factor (float): Multiplier for sales during winter months (Dec, Jan, Feb).
        spike_probability (float): Probability of a random, short-term sales spike (0.0 to 1.0).
        spike_multiplier (float): Multiplier for sales during a spike event.
        seed (int): Random seed for reproducibility.
        noise_level (float): Std dev of daily noise as a fraction of base sales.

    Returns:
        pd.DataFrame: A DataFrame with 'ds' (date) and 'y' (sales) columns,
                      along with various features and a 'spike_effect' column.
    """
    random.seed(seed)
    np.random.seed(seed)

    start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE)
    
    dates = pd.date_range(start=start, end=end, freq='D')
    df = pd.DataFrame({'ds': dates})

    df['day_of_week'] = df['ds'].dt.dayofweek # Monday=0, Sunday=6
    df['is_weekend'] = ((df['day_of_week'] == 5) | (df['day_of_week'] == 6)).astype(int)
    df['month'] = df['ds'].dt.month
    df['day_of_year'] = df['ds'].dt.dayofyear

    # Base sales with weekly seasonality
    df['y'] = base_sales * (1 + df['is_weekend'] * (weekly_peak_factor - 1))

    # Monthly seasonality (simplistic: apply factors to specific months)
    df['season_factor'] = 1.0
    df.loc[df['month'].isin([6, 7, 8]), 'season_factor'] = summer_factor
    df.loc[df['month'].isin([12, 1, 2]), 'season_factor'] = winter_factor
    df['y'] = df['y'] * df['season_factor']

    # Add holidays effect (simple reduction for now)
    holiday_dates = [pd.to_datetime(h) for h in HOLIDAYS]
    df['is_holiday'] = df['ds'].isin(holiday_dates).astype(int)
    # For now, let's assume holidays increase sales (e.g., 1.2x on holiday itself)
    # If holidays cause a dip, this can be adjusted.
    df['y'] = df['y'] * (1 + df['is_holiday'] * 0.2) # Simple holiday boost

    # Add random noise
    df['noise'] = np.random.normal(0, base_sales * noise_level, len(df))
    df['y'] = df['y'] + df['noise']

    # Add sudden spike effect
    df['spike_effect'] = 0.0
    num_spikes = int(len(df) * spike_probability) # Number of days with potential spikes
    spike_days = random.sample(range(len(df)), num_spikes)
    for day_idx in spike_days:
        # A spike can last for 1-2 days. For simplicity, let's make it 1-day spike for now.
        # The `spike_effect` column will be the regressor, actual sales will be calculated.
        df.loc[day_idx, 'spike_effect'] = spike_multiplier - 1.0 # This is the *additional* factor
        df.loc[day_idx, 'y'] = df.loc[day_idx, 'y'] * spike_multiplier # Apply to actual sales

    df['y'] = df['y'].round().astype(int) # Sales must be integers
    df['y'] = df['y'].apply(lambda x: max(0, x)) # Ensure no negative sales

    # Add Prophet-friendly features (already there from the original generator)
    df['day_of_week_encoded'] = df['day_of_week'].map({0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6}) # 0-6
    df['month_encoded'] = df['month'] # 1-12
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    # Simulate temperature or other external factors
    # For simplicity, let's add a random temperature-like regressor
    df['temp_c_scaled'] = np.random.uniform(0, 1, len(df)) # Scaled temperature between 0 and 1

    return df[['ds', 'y', 'day_of_week', 'is_weekend', 'month', 'is_holiday',
               'day_of_week_encoded', 'month_encoded', 'day_of_year_sin', 
               'day_of_year_cos', 'temp_c_scaled', 'spike_effect']]