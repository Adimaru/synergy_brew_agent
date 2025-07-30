import pandas as pd
import numpy as np
import logging
from scipy.stats import norm

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_inventory_recommendation(
    current_stock: int,
    forecast_df: pd.DataFrame,
    lead_time_days: int,
    safety_stock_factor: float, # Retained for fallback if dynamic calculation not possible
    min_order_quantity: int,
    service_level: float, # Desired service level (e.g., 0.95)
    demand_forecast_error_std_dev: float, # Standard deviation of forecast errors
    debug_mode: bool = False
) -> dict:
    """
    Generates an inventory order recommendation based on current stock, forecast,
    lead time, and either a safety stock factor or dynamic safety stock from service level.

    Args:
        current_stock (int): Current inventory level.
        forecast_df (pd.DataFrame): DataFrame with 'ds' and 'yhat' columns for future sales.
        lead_time_days (int): Days until an order arrives.
        safety_stock_factor (float): Multiplier for safety stock if dynamic calculation isn't used.
        min_order_quantity (int): Minimum order quantity.
        service_level (float): Desired service level for dynamic safety stock (e.g., 0.95).
        demand_forecast_error_std_dev (float): Standard deviation of forecast errors, for dynamic safety stock.
        debug_mode (bool): If True, enables verbose logging.

    Returns:
        dict: Contains order_quantity, reorder_point, target_stock_level, and calculated_safety_stock.
    """
    if debug_mode:
        logging.info(f"  Inventory Recommendation Called: Current Stock={current_stock}, Lead Time={lead_time_days} days, Service Level={service_level*100:.1f}%, Error Std Dev={demand_forecast_error_std_dev:.2f}")

    # Ensure forecast_df 'ds' column is datetime and timezone-naive
    if 'ds' in forecast_df.columns and forecast_df['ds'].dt.tz is not None:
        forecast_df['ds'] = forecast_df['ds'].dt.tz_localize(None)

    # Calculate demand during lead time (sum of forecasted sales for lead_time_days)
    # The forecast_df's first 'yhat' is for the next day after the sales history,
    # so we need to select 'lead_time_days' from the beginning of the forecast.
    
    # Filter forecast to only include the lead time period
    forecast_for_lead_time = forecast_df.head(lead_time_days)
    
    if forecast_for_lead_time.empty:
        logging.warning("No forecast data available for the lead time. Cannot generate recommendation.")
        return {
            'order_quantity': 0,
            'reorder_point': 0,
            'target_stock_level': 0,
            'calculated_safety_stock': 0
        }

    lead_time_demand_forecast = forecast_for_lead_time['yhat'].sum()
    # Ensure lead_time_demand_forecast is non-negative
    lead_time_demand_forecast = max(0, lead_time_demand_forecast)

    if debug_mode:
        logging.info(f"  Lead Time Demand Forecast ({lead_time_days} days): {lead_time_demand_forecast:.2f}")

    calculated_safety_stock = 0
    # --- Dynamic Safety Stock Calculation ---
    # Only use dynamic safety stock if service_level is meaningful and there's sufficient error data
    if service_level > 0 and demand_forecast_error_std_dev > 0.01: # Small threshold to avoid near-zero std dev issues
        try:
            # Z-score for desired service level (one-tailed)
            z_score = norm.ppf(service_level)
            
            # Safety Stock = Z-score * Standard Deviation of Lead Time Demand Forecast Error
            # Assuming demand_forecast_error_std_dev is the STD DEV OF DAILY FORECAST ERRORS:
            calculated_safety_stock = z_score * demand_forecast_error_std_dev * np.sqrt(lead_time_days)
            
            if debug_mode:
                logging.info(f"  Dynamic Safety Stock: Z-score={z_score:.2f} * Error Std Dev={demand_forecast_error_std_dev:.2f} * sqrt(Lead Time)={np.sqrt(lead_time_days):.2f} = {calculated_safety_stock:.2f}")

        except Exception as e:
            logging.error(f"Error calculating dynamic safety stock: {e}. Falling back to factor-based safety stock.")
            calculated_safety_stock = lead_time_demand_forecast * safety_stock_factor
    else:
        # Fallback to simple factor-based safety stock if dynamic calculation not possible/desired
        calculated_safety_stock = lead_time_demand_forecast * safety_stock_factor
        if debug_mode:
            logging.info(f"  Using Factor-based Safety Stock: {lead_time_demand_forecast:.2f} * {safety_stock_factor:.2f} = {calculated_safety_stock:.2f}")
    
    # Ensure safety stock is not negative and is an integer
    calculated_safety_stock = max(0, int(np.ceil(calculated_safety_stock)))

    # Reorder Point (s in (s,S) policy)
    # Reorder Point = Expected Demand During Lead Time + Safety Stock
    reorder_point = int(np.ceil(lead_time_demand_forecast + calculated_safety_stock))

    # Target Stock Level (S in (s,S) policy)
    # This is the "order up to" level. When stock drops below ROP, we order enough
    # to bring it up to this target.
    # A common approach: Target Stock = ROP + Average_Lead_Time_Demand
    # This means when we order, we order enough to satisfy the ROP and cover *another*
    # lead time's worth of demand.
    target_stock_level = int(np.ceil(reorder_point + lead_time_demand_forecast))
    
    # Ensure target_stock_level is at least ROP
    target_stock_level = max(reorder_point, target_stock_level)


    if debug_mode:
        logging.info(f"  Reorder Point: {reorder_point}, Target Stock Level: {target_stock_level}, Calculated Safety Stock: {calculated_safety_stock}")

    # Determine order quantity
    order_quantity = 0
    # Only place an order if total available stock (on hand + in transit - not handled here) falls below ROP.
    # Here, we only consider current_stock. `core.py` will handle `on_order` quantities.
    if current_stock < reorder_point:
        # Calculate quantity needed to reach the target stock level
        qty_needed = target_stock_level - current_stock
        order_quantity = max(min_order_quantity, int(np.ceil(qty_needed))) # Ensure minimum order quantity
        
        # Ensure order quantity is non-negative
        order_quantity = max(0, order_quantity)

        if debug_mode:
            logging.info(f"  Current stock ({current_stock}) is below reorder point ({reorder_point}). Ordering {order_quantity} units to reach target {target_stock_level}.")
    else:
        if debug_mode:
            logging.info(f"  Current stock ({current_stock}) is above reorder point ({reorder_point}). No order placed.")

    return {
        'order_quantity': order_quantity,
        'reorder_point': reorder_point,
        'target_stock_level': target_stock_level,
        'calculated_safety_stock': calculated_safety_stock
    }