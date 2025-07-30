# synergy_brew_agent/src/agent/forecaster.py

import pandas as pd
import numpy as np
from scipy.stats import norm

def calculate_moving_average_forecast(sales_history: pd.Series, window_size: int) -> float:
    """
    Calculates the simple moving average forecast for the next period.

    Args:
        sales_history (pd.Series): A pandas Series of historical sales data (daily sales).
                                   The most recent sales should be at the end.
        window_size (int): The number of past periods to include in the moving average.

    Returns:
        float: The forecasted sales for the next period. Returns 0 if not enough data.
    """
    if len(sales_history) < window_size:
        # Not enough data to calculate MA for the given window
        return sales_history.mean() if not sales_history.empty else 0.0
    else:
        # Calculate moving average from the most recent 'window_size' days
        return sales_history.tail(window_size).mean()

def calculate_safety_stock(
    historical_demand: pd.Series,
    service_level: float,
    lead_time_days: int
) -> float:
    """
    Calculates the safety stock needed to achieve a desired service level.

    Formula: Safety Stock = Z * StdDev_LT
    Where:
        Z = Z-score corresponding to the desired service level.
        StdDev_LT = Standard deviation of demand during lead time.

    Args:
        historical_demand (pd.Series): Series of past daily demand values.
        service_level (float): Desired service level (e.g., 0.95 for 95%).
        lead_time_days (int): The lead time for delivery in days.

    Returns:
        float: The calculated safety stock in units.
    """
    if len(historical_demand) < 2: # Need at least 2 points to calculate std dev
        return 0 # Cannot calculate meaningful safety stock

    # Calculate standard deviation of daily demand
    std_dev_daily_demand = historical_demand.std()

    if np.isnan(std_dev_daily_demand) or std_dev_daily_demand == 0:
        return 0

    # Calculate Z-score for the desired service level
    # norm.ppf(service_level) gives the Z-score for a given cumulative probability
    z_score = norm.ppf(service_level)

    # Standard deviation of demand during lead time
    # Assuming demand during lead time follows a normal distribution,
    # StdDev_LT = StdDev_DailyDemand * sqrt(LeadTime)
    std_dev_lead_time_demand = std_dev_daily_demand * np.sqrt(lead_time_days)

    safety_stock = z_score * std_dev_lead_time_demand

    # Safety stock should not be negative
    return max(0, safety_stock)

def calculate_reorder_point(
    average_daily_demand_forecast: float,
    lead_time_days: int,
    safety_stock: float
) -> float:
    """
    Calculates the reorder point.

    Formula: Reorder Point = (Average Daily Demand Forecast * Lead Time) + Safety Stock

    Args:
        average_daily_demand_forecast (float): The forecasted average daily demand.
        lead_time_days (int): The lead time for delivery in days.
        safety_stock (float): The calculated safety stock in units.

    Returns:
        float: The calculated reorder point in units.
    """
    demand_during_lead_time = average_daily_demand_forecast * lead_time_days
    reorder_point = demand_during_lead_time + safety_stock
    return reorder_point

def calculate_order_quantity_by_target_level(
    current_inventory: float,
    on_order: float,
    target_inventory_level: float
) -> int:
    """
    Calculates the order quantity to reach a target inventory level.
    This is often used in (R, s, S) or (R, S) policies where S is the target level.

    Args:
        current_inventory (float): Current physical inventory on hand.
        on_order (float): Inventory currently on order and not yet received.
        target_inventory_level (float): The desired maximum inventory level to reach after placing an order.

    Returns:
        int: The quantity to order.
    """
    needed_to_reach_target = target_inventory_level - (current_inventory + on_order)
    return max(0, int(np.ceil(needed_to_reach_target)))