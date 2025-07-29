import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import holidays # We'll need to install this library

def generate_temperature(dates, base_temp=15, annual_amplitude=10, daily_noise=2):
    """Simulates daily temperature with annual seasonality."""
    temps = []
    for date in dates:
        day_of_year = date.timetuple().tm_yday
        # Simple sine wave for seasonality (peak around July/August)
        seasonal_temp = base_temp + annual_amplitude * np.sin(2 * np.pi * (day_of_year - 90) / 365.25)
        temps.append(seasonal_temp + np.random.normal(0, daily_noise))
    return np.array(temps)

def generate_rainfall(dates, rain_prob=0.3, max_rain=10):
    """Simulates daily rainfall (binary and amount)."""
    rain_amounts = []
    is_rainy_day = []
    for _ in dates:
        if np.random.rand() < rain_prob:
            rain = np.random.uniform(0, max_rain)
            rain_amounts.append(rain)
            is_rainy_day.append(1)
        else:
            rain_amounts.append(0)
            is_rainy_day.append(0)
    return np.array(rain_amounts), np.array(is_rainy_day)

def generate_promotions(dates, promo_frequency_days=60, promo_duration_days=3):
    """Simulates periodic promotions."""
    is_promo = np.zeros(len(dates), dtype=int)
    for i in range(0, len(dates), promo_frequency_days):
        for j in range(promo_duration_days):
            if i + j < len(dates):
                is_promo[i + j] = 1
    return is_promo

def generate_sales_data_enriched(
    product_name, store_id, start_date, days, base_sales,
    seasonality_factor=0.2, trend_factor=0.0005, holiday_impact=0.3,
    temp_sensitivity=3, rain_sensitivity=-0.1, promo_boost=0.2
):
    data = []
    dates = [start_date + timedelta(days=i) for i in range(days)]

    temps = generate_temperature(dates, base_temp=15, annual_amplitude=10, daily_noise=2)
    rain_amounts, is_rainy_day = generate_rainfall(dates, rain_prob=0.25, max_rain=8)
    # Using US holidays as an example; adjust for Belgium or relevant country
    my_holidays = holidays.country_holidays('US', years=range(start_date.year, start_date.year + (days // 365) + 2))
    is_holiday = [1 if date in my_holidays else 0 for date in dates]
    promotions = generate_promotions(dates, promo_frequency_days=45, promo_duration_days=4)


    for i, current_date in enumerate(dates):
        daily_sales = base_sales + (i * trend_factor * base_sales) # Linear trend

        # Weekend seasonality: higher sales on weekends
        if current_date.weekday() in [5, 6]: # Saturday (5) or Sunday (6)
            daily_sales *= (1 + seasonality_factor)
        else:
            daily_sales *= (1 - seasonality_factor * 0.2) # Slightly lower on weekdays

        # Apply holiday impact
        if is_holiday[i]:
            daily_sales *= (1 + holiday_impact)

        # Apply temperature impact (e.g., warmer == more cold drinks)
        # Assuming a sweet spot around 20C for cold drinks, decreasing for hot.
        # This is a simplification and would need more nuanced modeling.
        temp_deviation = temps[i] - 15 # Deviation from a comfortable average temp
        daily_sales *= (1 + temp_deviation * temp_sensitivity / 100) # Percentage change per degree

        # Apply rainfall impact (e.g., rain == lower foot traffic)
        if is_rainy_day[i]:
            daily_sales *= (1 + rain_amounts[i] * rain_sensitivity / 100) # Negative impact per mm of rain

        # Apply promotion boost
        if promotions[i]:
            daily_sales *= (1 + promo_boost)

        # Add some random noise
        daily_sales = max(0, daily_sales + np.random.normal(0, base_sales * 0.08))

        data.append({
            'date': current_date,
            'store_id': store_id,
            'product_name': product_name,
            'sales': int(round(daily_sales)),
            'avg_temp_c': round(temps[i], 1),
            'rainfall_mm': round(rain_amounts[i], 1),
            'is_holiday': is_holiday[i],
            'is_promotion': promotions[i]
        })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Define simulation parameters
    product = "Latte" # Or "Cold Brew" if we want to emphasize temp sensitivity
    store = "Store_A_Downtown"
    start = datetime(2023, 1, 1) # Start date for historical data
    duration_days = 365 * 3 # Three years of data
    avg_daily_sales = 180 # Average daily sales

    print(f"Generating enriched sales data for '{product}' at '{store}'...")
    sales_df = generate_sales_data_enriched(
        product_name=product,
        store_id=store,
        start_date=start,
        days=duration_days,
        base_sales=avg_daily_sales
    )

    # Save to a CSV file (our simulated POS data)
    file_path = f"data/{store}_{product}_enriched_sales_history.csv"
    import os
    os.makedirs('data', exist_ok=True) # Create data directory if it doesn't exist
    sales_df.to_csv(file_path, index=False)
    print(f"Enriched simulated data saved to: {file_path}")
    print("\nFirst 5 rows of generated data:")
    print(sales_df.head())
    print(f"\nTotal simulated sales records: {len(sales_df)}")
    print("\nColumns and their data types:")
    print(sales_df.info())