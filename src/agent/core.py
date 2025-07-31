# src/agent/core.py

import pandas as pd
import numpy as np
import logging
from datetime import timedelta, date, datetime
import os
import prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Ensure src directory is in the path for imports
import sys
from config.settings import (
    SIM_DATA_DIR,
    HOLDING_COST_PER_UNIT_PER_DAY,
    ORDERING_COST_PER_ORDER,
    STOCKOUT_COST_PER_UNIT_LOST_SALE,
    FINANCIAL_LOG_FILE,
    INVENTORY_LOG_FILE,
    PERFORMANCE_LOG_FILE,
    DEFAULT_FORECAST_HORIZON_DAYS,
    BASE_DIR,
    START_DATE,
    END_DATE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Global variables for simulation state
simulation_state = {
    'product_key': None,
    'store_key': None,
    'inventory': 0,
    'days_until_delivery': {},  # {order_id: days_left}
    'order_history': [],
    'financial_log': [],
    'inventory_log': [],
    'performance_log': [],
    'total_sales': 0,
    'total_lost_sales': 0
}

def _get_forecasting_parameters(forecasting_model, moving_average_window, prophet_params):
    """
    Returns a dictionary of parameters for the chosen forecasting model.
    """
    if forecasting_model == "Moving Average":
        return {'window': moving_average_window}
    elif forecasting_model == "Prophet":
        # Ensure prophet_params is a dictionary, defaulting to an empty one if None
        return {'prophet_params': prophet_params if prophet_params is not None else {}}
    return {}

def _check_and_load_sales_data(data_file_path, debug_mode):
    """
    Loads sales data, fills missing dates, and returns a DataFrame.
    Returns None if the file does not exist or an error occurs.
    """
    if not os.path.exists(data_file_path):
        logging.error(f"Sales data file not found: {data_file_path}")
        return None

    try:
        df = pd.read_csv(data_file_path)
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Ensure data is sorted by date
        df = df.sort_values(by='ds').reset_index(drop=True)

        # Generate a complete date range and merge with existing data
        full_date_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
        full_df = pd.DataFrame(full_date_range, columns=['ds'])
        full_df = pd.merge(full_df, df, on='ds', how='left')

        # Fill missing 'y' values (sales) with 0
        full_df['y'] = full_df['y'].fillna(0)
        
        logging.info(f"Loaded {len(full_df)} rows of sales data from {data_file_path} after ensuring continuity and filling NaNs.")
        if debug_mode:
            logging.debug(f"Sales data head:\n{full_df.head()}")
        return full_df

    except Exception as e:
        logging.error(f"Failed to load or process sales data from {data_file_path}: {e}")
        return None

def _calculate_safety_stock(forecast_df, service_level):
    """
    Calculates safety stock based on forecast error variance and desired service level.
    This is a simplified approach, a more complex model would use a Z-score and lead time variance.
    """
    if forecast_df.empty:
        return 0
    
    # Calculate forecast errors
    forecast_df['error'] = forecast_df['y'] - forecast_df['yhat']
    
    # Get the standard deviation of forecast errors
    forecast_error_std = forecast_df['error'].std()
    
    # Z-score for the desired service level (simplified, from a standard normal table)
    # 80% SL -> Z=0.84, 90% SL -> Z=1.28, 95% SL -> Z=1.65, 99% SL -> Z=2.33
    z_score_map = {
        0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 0.95: 1.65, 0.99: 2.33
    }
    z_score = z_score_map.get(service_level, 1.65) # Default to 95%
    
    safety_stock = z_score * forecast_error_std
    
    # Ensure safety stock is not negative and round up
    return max(0, int(np.ceil(safety_stock)))

def _perform_prophet_cv(df, initial, period, horizon):
    """
    Performs Prophet cross-validation and returns a performance metrics DataFrame.
    """
    try:
        df_cv = cross_validation(
            model=Prophet(),
            df=df,
            initial=initial,
            period=period,
            horizon=horizon,
            cutoffs=None, # Use default cutoffs
            parallel="processes"
        )
        df_p = performance_metrics(df_cv)
        return df_p
    except ValueError as e:
        logging.error(f"Prophet cross-validation failed: {e}")
        logging.error("This may happen if the dataset is too small for the specified 'initial', 'period', or 'horizon' values.")
        return None

def _forecast_demand(historical_sales, forecasting_model, **kwargs):
    """
    Predicts future demand using the specified model.
    """
    forecast_horizon = kwargs.get('forecast_horizon_days', DEFAULT_FORECAST_HORIZON_DAYS)
    
    if forecasting_model == "Moving Average":
        window = kwargs.get('window', 7)
        if len(historical_sales) < window:
            logging.warning(f"Historical sales data ({len(historical_sales)}) is less than MA window ({window}). Returning 0 forecast.")
            return pd.DataFrame({'yhat': [0] * forecast_horizon})
        
        last_N_sales = historical_sales['y'].tail(window)
        moving_average = last_N_sales.mean()
        forecast_values = [moving_average] * forecast_horizon
        return pd.DataFrame({'yhat': forecast_values})
    
    elif forecasting_model == "Prophet":
        prophet_params = kwargs.get('prophet_params', {})
        try:
            m = Prophet(**prophet_params)
            m.fit(historical_sales)
            future = m.make_future_dataframe(periods=forecast_horizon)
            forecast = m.predict(future)
            return forecast[['ds', 'yhat']].tail(forecast_horizon)
        except Exception as e:
            logging.error(f"Prophet forecasting failed: {e}")
            logging.error("Using a naive forecast (last day's sales) instead.")
            last_day_sales = historical_sales['y'].iloc[-1] if not historical_sales.empty else 0
            return pd.DataFrame({'yhat': [last_day_sales] * forecast_horizon})
            
    elif forecasting_model == "Actual Sales Data (Baseline)":
        # This is for a baseline comparison, where we 'cheat' and use the actual future sales
        future_dates = pd.date_range(start=historical_sales['ds'].max() + timedelta(days=1), periods=forecast_horizon)
        actual_future_sales = pd.DataFrame({'ds': future_dates, 'yhat': [0] * forecast_horizon})
        return actual_future_sales

    else:
        logging.error(f"Unknown forecasting model: {forecasting_model}. Returning a forecast of 0.")
        return pd.DataFrame({'yhat': [0] * forecast_horizon})

def _generate_and_log_recommendation(
    sim_date,
    current_inventory,
    pending_orders_inventory,
    sales_data_so_far,
    forecasting_model,
    forecasting_params,
    min_order_quantity,
    lead_time_days,
    service_level,
    actual_daily_sales,
    lost_sales_today,
    debug_mode
):
    """
    Generates an inventory recommendation and logs the daily inventory state.
    Returns the recommended order quantity.
    """
    # 1. Forecast Demand for the lead time period
    forecast_df = _forecast_demand(
        historical_sales=sales_data_so_far,
        forecasting_model=forecasting_model,
        forecast_horizon_days=lead_time_days,
        **forecasting_params
    )
    
    # 2. Calculate Safety Stock
    # Only use Prophet for safety stock calculation if there's enough historical data (>= 2 rows).
    if forecasting_model == "Prophet" and len(sales_data_so_far) >= 2:
        try:
            # Fit Prophet on historical data to get in-sample forecast errors
            m_historical = Prophet(**forecasting_params['prophet_params'])
            m_historical.fit(sales_data_so_far)
            forecast_on_historical = m_historical.predict(sales_data_so_far)
            
            # Combine historical 'y' with the in-sample Prophet 'yhat'
            forecast_historical_data = pd.merge(
                sales_data_so_far,
                forecast_on_historical[['ds', 'yhat']],
                on='ds',
                how='left'
            )
            
            safety_stock = _calculate_safety_stock(forecast_historical_data, service_level)
            
        except Exception as e:
            logging.error(f"Prophet failed to calculate safety stock with historical data: {e}")
            logging.error("Falling back to a simpler safety stock calculation.")
            
            # Fallback for safety stock
            z_score_map = {
                0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 0.95: 1.65, 0.99: 2.33
            }
            z_score = z_score_map.get(service_level, 1.65)
            std_dev_daily_sales = sales_data_so_far['y'].std() if len(sales_data_so_far) > 1 else 0
            safety_stock = int(np.ceil(z_score * std_dev_daily_sales * np.sqrt(lead_time_days)))

    else:
        # A simpler, more direct calculation for other models or when Prophet can't be used
        z_score_map = {
            0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 0.95: 1.65, 0.99: 2.33
        }
        z_score = z_score_map.get(service_level, 1.65)
        std_dev_daily_sales = sales_data_so_far['y'].std() if len(sales_data_so_far) > 1 else 0
        safety_stock = int(np.ceil(z_score * std_dev_daily_sales * np.sqrt(lead_time_days)))

    # 3. Calculate forecasted demand during lead time (DDLT)
    forecasted_demand_during_lead_time = int(np.ceil(forecast_df['yhat'].sum()))

    # 4. Calculate Reorder Point (ROP) and Target Inventory Level
    target_inventory = forecasted_demand_during_lead_time + safety_stock
    current_inventory_position = current_inventory + pending_orders_inventory
    
    # 5. Determine Order Quantity
    order_qty = max(0, target_inventory - current_inventory_position)

    # 6. Apply Minimum Order Quantity constraint
    if order_qty > 0 and order_qty < min_order_quantity:
        order_qty = min_order_quantity
    
    # 7. Log the daily state
    total_inventory_position = current_inventory + pending_orders_inventory
    
    if debug_mode:
        logging.info(f"--- Inventory Report for {sim_date.strftime('%Y-%m-%d')} ---")
        logging.info(f"Current Stock: {current_inventory}")
        logging.info(f"Pending Orders: {pending_orders_inventory} units")
        logging.info(f"Forecasted Demand during LT ({lead_time_days} days): {forecasted_demand_during_lead_time}")
        logging.info(f"Safety Stock ({service_level*100}% SL): {safety_stock}")
        logging.info(f"Target Inventory Level: {target_inventory}")
        logging.info(f"Current Inventory Position: {current_inventory_position}")
        logging.info(f"Recommended Order Quantity: {order_qty}")

    # Now, let's log the inventory state for the day
    simulation_state['inventory_log'].append({
        'date': sim_date,
        'actual_sales_today': actual_daily_sales,
        'lost_sales_today': lost_sales_today, # NEW: Use the lost_sales_today parameter
        'starting_stock': current_inventory + lost_sales_today + pending_orders_inventory, # Corrected calculation
        'ending_stock': current_inventory,
        'pending_orders': pending_orders_inventory,
        'lead_time_forecast': forecasted_demand_during_lead_time,
        'safety_stock': safety_stock,
        'order_placed': order_qty,
        'service_level_target': service_level
    })
    
    return order_qty


def simulate_agent_for_n_days(
    num_days,
    product_key,
    store_key,
    data_file_path,
    initial_stock,
    lead_time_days,
    safety_stock_factor, # Not used in service-level based calculation, kept for compatibility
    min_order_quantity,
    service_level,
    forecasting_model,
    moving_average_window,
    prophet_params,
    debug_mode_for_run=False
):
    """
    Main function to run the inventory simulation.
    """
    
    # Reset the global state for a new simulation run
    simulation_state.update({
        'product_key': product_key,
        'store_key': store_key,
        'inventory': initial_stock,
        'days_until_delivery': {},
        'order_history': [],
        'financial_log': [],
        'inventory_log': [],
        'performance_log': [],
        'total_sales': 0,
        'total_lost_sales': 0
    })
    
    logging.info(f"Starting simulation for {product_key} at {store_key} for {num_days} days.")
    logging.info(f"Simulation parameters: Lead Time={lead_time_days} days, Service Level={service_level*100}%, Min Order Qty={min_order_quantity}, Forecasting Model='{forecasting_model}', MA Window={moving_average_window}")

    all_sales_data = _check_and_load_sales_data(data_file_path, debug_mode_for_run)
    if all_sales_data is None or all_sales_data.empty:
        logging.error("Failed to load sales data. Simulation aborted.")
        return None

    # Check if the requested simulation period is valid
    sim_start_date = START_DATE
    sim_end_date = sim_start_date + timedelta(days=num_days - 1)

    start_ts = pd.Timestamp(sim_start_date)
    end_ts = pd.Timestamp(sim_end_date)
    
    # Check if the requested simulation range is valid with respect to the loaded data
    if start_ts < all_sales_data['ds'].min() or end_ts > all_sales_data['ds'].max():
        logging.error(f"No simulation data found for the period {sim_start_date.strftime('%Y-%m-%d')} to {sim_end_date.strftime('%Y-%m-%d')}.")
        logging.error(f"Available data range is from {all_sales_data['ds'].min().strftime('%Y-%m-%d')} to {all_sales_data['ds'].max().strftime('%Y-%m-%d')}.")
        return None
    
    
    # Slice the sales data for the simulation period
    sim_sales_data = all_sales_data[
        (all_sales_data['ds'] >= start_ts) & 
        (all_sales_data['ds'] <= end_ts)
    ].reset_index(drop=True)

    # Dictionary to hold forecasting-specific parameters
    forecasting_params = _get_forecasting_parameters(forecasting_model, moving_average_window, prophet_params)

    for i, row in sim_sales_data.iterrows():
        sim_date = row['ds']
        actual_sales_today = row['y']

        # --- Daily Operations ---
        
        # 1. Receive incoming orders
        deliveries_received_today = 0
        orders_to_remove = []
        for order_id, days_left in list(simulation_state['days_until_delivery'].items()):
            if days_left == 0:
                deliveries_received_today += simulation_state['order_history'][order_id]['quantity']
                orders_to_remove.append(order_id)
        
        for order_id in orders_to_remove:
            del simulation_state['days_until_delivery'][order_id]

        simulation_state['inventory'] += deliveries_received_today
        
        # 2. Process Sales
        sales_processed_today = min(simulation_state['inventory'], actual_sales_today)
        lost_sales_today = actual_sales_today - sales_processed_today
        simulation_state['inventory'] -= sales_processed_today

        simulation_state['total_sales'] += sales_processed_today
        simulation_state['total_lost_sales'] += lost_sales_today
        
        # 3. Calculate Daily Costs
        holding_cost = simulation_state['inventory'] * HOLDING_COST_PER_UNIT_PER_DAY
        stockout_cost = lost_sales_today * STOCKOUT_COST_PER_UNIT_LOST_SALE
        
        # 4. Place a new order? (Agent's decision)
        
        # Get historical data up to this point for forecasting
        historical_data_for_forecast = sim_sales_data[sim_sales_data['ds'] < sim_date].copy()
        
        # Calculate pending inventory from orders in transit
        pending_orders_inventory = sum(
            simulation_state['order_history'][order_id]['quantity']
            for order_id in simulation_state['days_until_delivery']
        )
        
        # The agent decides to place an order
        order_qty = _generate_and_log_recommendation(
            sim_date=sim_date,
            current_inventory=simulation_state['inventory'],
            pending_orders_inventory=pending_orders_inventory,
            sales_data_so_far=historical_data_for_forecast,
            forecasting_model=forecasting_model,
            forecasting_params=forecasting_params,
            min_order_quantity=min_order_quantity,
            lead_time_days=lead_time_days,
            service_level=service_level,
            actual_daily_sales=sales_processed_today,
            lost_sales_today=lost_sales_today,
            debug_mode=debug_mode_for_run
        )
        
        ordering_cost = 0
        if order_qty > 0:
            ordering_cost = ORDERING_COST_PER_ORDER
            
            # Place the order with a lead time
            order_id = len(simulation_state['order_history'])
            simulation_state['order_history'].append({
                'id': order_id,
                'quantity': order_qty,
                'order_date': sim_date
            })
            simulation_state['days_until_delivery'][order_id] = lead_time_days
            
        # 5. Log Financials
        total_daily_cost = holding_cost + ordering_cost + stockout_cost
        simulation_state['financial_log'].append({
            'date': sim_date,
            'holding_cost': holding_cost,
            'ordering_cost': ordering_cost,
            'stockout_cost': stockout_cost,
            'total_daily_cost': total_daily_cost,
            'deliveries_received': deliveries_received_today
        })

        # 6. Update pending order days
        for order_id in simulation_state['days_until_delivery']:
            simulation_state['days_until_delivery'][order_id] -= 1

    logging.info(f"Simulation for {product_key} at {store_key} finished.")
    
    # Store results in a dictionary and return
    results = {
        'product_key': product_key,
        'store_key': store_key,
        'financial_log': simulation_state['financial_log'],
        'inventory_log': simulation_state['inventory_log'],
        'total_sales': simulation_state['total_sales'],
        'total_lost_sales': simulation_state['total_lost_sales'],
        'final_inventory': simulation_state['inventory']
    }
    
    return results

def clear_all_state_data(debug_mode=False):
    """
    Deletes all saved state files and simulation data files.
    """
    files_to_delete = [FINANCIAL_LOG_FILE, INVENTORY_LOG_FILE, PERFORMANCE_LOG_FILE]
    
    # Delete state files
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            if debug_mode:
                logging.info(f"Deleted state file: {file_path}")
    
    # Delete simulation data files
    for file_name in os.listdir(SIM_DATA_DIR):
        file_path = os.path.join(SIM_DATA_DIR, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            if debug_mode:
                logging.info(f"Deleted simulation data file: {file_path}")

    logging.info("All saved state and simulation data files have been deleted.")