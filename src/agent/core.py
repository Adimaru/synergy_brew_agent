import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json

# Import modules from our new src structure
from src.agent.state_manager import load_state, save_state
from src.agent.forecasting import load_enriched_sales_data, train_and_forecast_model
from src.agent.inventory import generate_inventory_recommendation

# Import configurations
from config.settings import (
    PERFORMANCE_LOG_FILE, INVENTORY_LOG_FILE, START_DATE, END_DATE,
    DEFAULT_SERVICE_LEVEL,
    HOLDING_COST_PER_UNIT_PER_DAY, ORDERING_COST_PER_ORDER, STOCKOUT_COST_PER_UNIT_LOST_SALE
)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _initialize_daily_run(
    sim_date: datetime,
    initial_stock: int,
    agent_state: dict,
    debug_mode: bool
) -> tuple[int, list, list, list, list]:
    """Initializes stock and logs for a new simulation or continues a previous one."""
    
    current_stock = agent_state.get('current_stock', initial_stock)
    performance_log = agent_state.get('performance_log', [])
    inventory_log = agent_state.get('inventory_log', [])
    forecast_errors = agent_state.get('forecast_errors', [])
    financial_log = agent_state.get('financial_log', [])

    # If starting a truly fresh simulation (not just resuming from a break in run)
    # or if the loaded state is older than the current simulation start date.
    if not agent_state['last_run_date'] or \
       (pd.to_datetime(agent_state['last_run_date']) < pd.to_datetime(START_DATE) and debug_mode):
        current_stock = initial_stock
        performance_log = []
        inventory_log = []
        forecast_errors = []
        financial_log = []
        if debug_mode:
            logging.info(f"Initialized new simulation run for {sim_date.strftime('%Y-%m-%d')} with initial stock {initial_stock}. Logs reset.")
    else:
        # Resume from last known state
        if debug_mode:
            logging.info(f"Resuming simulation from {agent_state['last_run_date']} for {sim_date.strftime('%Y-%m-%d')}.")
    
    return current_stock, performance_log, inventory_log, forecast_errors, financial_log


def _process_previous_day_data(
    current_day_data: pd.Series,
    current_stock: int,
    # deliveries_expected_today: int, # This param is now effectively handled before this function call
    debug_mode: bool
) -> tuple[int, int, int]:
    """Processes sales, updates stock, and handles deliveries for the current day."""
    
    actual_sales_today = current_day_data['y'] # 'y' is the actual sales for the day
    
    # current_stock already includes deliveries processed earlier in simulate_agent_for_n_days

    # Process sales: ensure sales don't go below 0
    actual_sales_processed = min(actual_sales_today, current_stock) # Use current_stock directly after deliveries
    ending_stock = current_stock - actual_sales_processed
    
    lost_sales = 0
    if actual_sales_processed < actual_sales_today:
        lost_sales = actual_sales_today - actual_sales_processed
        if debug_mode:
            logging.warning(f"Stockout! Lost {lost_sales} sales on {current_day_data['ds'].strftime('%Y-%m-%d')}.")

    if debug_mode:
        logging.info(f"  Sales for {current_day_data['ds'].strftime('%Y-%m-%d')}: Actual {actual_sales_today}, Processed {actual_sales_processed}")
        logging.info(f"  Starting Stock (after deliveries, before sales): {current_stock + actual_sales_processed}, Ending Stock: {ending_stock}")
    
    return ending_stock, actual_sales_processed, lost_sales


def _generate_and_log_recommendation(
    sim_date: datetime,
    ending_stock: int,
    sales_history_df_for_forecast: pd.DataFrame,
    lead_time_days: int,
    safety_stock_factor: float,
    min_order_quantity: int,
    performance_log: list,
    inventory_log: list,
    financial_log: list,
    forecast_errors_history: list,
    current_day_data_actual_sales: int,
    service_level: float,
    deliveries_received_today: int,
    lost_sales_today: int,
    forecasting_model: str,        # <--- NEW: Forecasting model type
    moving_average_window: int,    # <--- NEW: Moving average window
    debug_mode: bool
) -> int:
    """Generates an order recommendation and logs daily inventory status."""

    forecast_model = None
    forecast_df = pd.DataFrame()
    forecasted_sales_today = 0
    demand_forecast_error_std_dev = 0 # Default to 0 for baseline/cold start

    # --- Special Handling for "Actual Sales Data (Baseline)" ---
    if forecasting_model == "Actual Sales Data (Baseline)":
        if debug_mode:
            logging.info("  Forecasting Model: Actual Sales Data (Baseline) - Simulating perfect knowledge.")
        
        forecasted_sales_today = current_day_data_actual_sales
        
        # Create a mock forecast_df for inventory recommendation, assuming perfect future knowledge
        # The 'yhat' for the next 'lead_time_days' is exactly current day's actual sales
        mock_future_dates = pd.date_range(start=sim_date + timedelta(days=1), periods=lead_time_days + 14, freq='D')
        forecast_df = pd.DataFrame({
            'ds': mock_future_dates,
            'yhat': current_day_data_actual_sales # Assuming future demand is constant at today's actual
        })
        # With perfect knowledge, forecast error std dev is 0
        demand_forecast_error_std_dev = 0 

    else:
        # --- Normal Forecasting (Prophet or Moving Average) ---
        # 1. Train model and forecast for next N days
        # Pass the new parameters to train_and_forecast_model
        forecast_model, forecast_df = train_and_forecast_model(
            sales_history_df_for_forecast,
            periods_to_forecast=lead_time_days + 14, # Forecast enough days to cover lead time + buffer
            forecasting_model_type=forecasting_model, # Pass model type
            moving_average_window=moving_average_window, # Pass MA window
            debug_mode=debug_mode
        )

        if not forecast_df.empty:
            # Get today's forecast (or the closest available if sim_date isn't exactly in forecast_df)
            today_forecast = forecast_df[forecast_df['ds'] == sim_date + timedelta(days=1)] # Forecast is for *next* day
            if not today_forecast.empty:
                forecasted_sales_today = today_forecast['yhat'].iloc[0]
                if debug_mode:
                    logging.info(f"  Forecasted sales for {sim_date.strftime('%Y-%m-%d')}: {forecasted_sales_today:.2f}")
            else:
                if debug_mode:
                    logging.warning(f"  No forecast available in generated forecast_df for current simulation date {sim_date.strftime('%Y-%m-%d')}.")

        # --- Calculate standard deviation of forecast errors for dynamic safety stock ---
        if len(forecast_errors_history) > 5: # Need a minimum number of data points for meaningful std dev
            demand_forecast_error_std_dev = np.std(forecast_errors_history)
            if debug_mode:
                logging.info(f"  Standard Deviation of Forecast Errors ({len(forecast_errors_history)} points): {demand_forecast_error_std_dev:.2f}")
        else:
            if debug_mode:
                logging.warning(f"  Not enough forecast errors ({len(forecast_errors_history)}) to calculate reliable standard deviation. Using fixed safety stock factor as fallback.")
            # If not enough history, demand_forecast_error_std_dev remains 0, which will trigger factor-based fallback in inventory.py

    # 2. Generate inventory recommendation
    # Pass the appropriate forecast_df and std dev
    recommendation = generate_inventory_recommendation(
        current_stock=ending_stock,
        forecast_df=forecast_df, # This will be the mock_forecast_df for baseline or actual forecast_df
        lead_time_days=lead_time_days,
        safety_stock_factor=safety_stock_factor,
        min_order_quantity=min_order_quantity,
        service_level=service_level,
        demand_forecast_error_std_dev=demand_forecast_error_std_dev, # This will be 0 for baseline
        debug_mode=debug_mode
    )
    order_qty = recommendation['order_quantity']

    # Log performance (actual vs. forecasted) for the current day
    calculate_and_log_performance(
        sim_date, current_day_data_actual_sales, forecasted_sales_today, performance_log, forecast_errors_history, debug_mode
    )

    # --- Calculate and Log Financial Metrics ---
    holding_cost = ending_stock * HOLDING_COST_PER_UNIT_PER_DAY
    ordering_cost = ORDERING_COST_PER_ORDER if order_qty > 0 else 0
    stockout_cost = lost_sales_today * STOCKOUT_COST_PER_UNIT_LOST_SALE
    total_daily_cost = holding_cost + ordering_cost + stockout_cost

    financial_log.append({
        'date': sim_date.strftime('%Y-%m-%d'),
        'holding_cost': holding_cost,
        'ordering_cost': ordering_cost,
        'stockout_cost': stockout_cost,
        'total_daily_cost': total_daily_cost
    })
    if debug_mode:
        logging.info(f"  Daily Costs for {sim_date.strftime('%Y-%m-%d')}: Holding=${holding_cost:.2f}, Ordering=${ordering_cost:.2f}, Stockout=${stockout_cost:.2f}, Total=${total_daily_cost:.2f}")

    # Log inventory status
    inventory_log.append({
        'date': sim_date.strftime('%Y-%m-%d'),
        'starting_stock': ending_stock + deliveries_received_today, # This should be stock before sales for the day
        'deliveries_today': deliveries_received_today,
        'actual_sales_today': current_day_data_actual_sales,
        'ending_stock': ending_stock,
        'order_placed_qty': order_qty,
        'reorder_point': recommendation['reorder_point'],
        'target_stock_level': recommendation['target_stock_level'],
        'forecasted_sales_today': forecasted_sales_today,
        'lost_sales_today': lost_sales_today,
        'calculated_safety_stock': recommendation.get('calculated_safety_stock', 0)
    })
    
    return order_qty


def _update_and_save_agent_state(
    sim_date: datetime,
    ending_stock: int,
    sales_history_df: pd.DataFrame,
    performance_log: list,
    inventory_log: list,
    forecast_errors: list,
    financial_log: list,
    product_key: str,
    store_key: str,
    debug_mode: bool
):
    """Updates the agent state and saves it."""
    agent_state = {
        'current_stock': ending_stock,
        'sales_history_df': sales_history_df,
        'last_run_date': sim_date.strftime('%Y-%m-%d'),
        'performance_log': performance_log,
        'inventory_log': inventory_log,
        'forecast_errors': forecast_errors,
        'financial_log': financial_log
    }
    save_state(agent_state, product_key, store_key, debug_mode)


def calculate_and_log_performance(
    date: datetime,
    actual_sales: float,
    forecasted_sales: float,
    performance_log: list,
    forecast_errors_list: list,
    debug_mode: bool
):
    """Calculates MAE and MAPE for the day and appends to log, also adds error to list."""
    mae = abs(actual_sales - forecasted_sales)
    mape = (mae / actual_sales) * 100 if actual_sales != 0 else 0

    performance_log.append({
        'forecast_date': date.strftime('%Y-%m-%d'),
        'actual_sales': actual_sales,
        'forecasted_sales': forecasted_sales,
        'mae': mae,
        'mape': mape
    })

    # Add the raw forecast error to the list
    forecast_errors_list.append(actual_sales - forecasted_sales)
    # Keep only the last N errors, e.g., 60 days, to ensure recent variability is captured
    MAX_ERROR_HISTORY = 60 # Arbitrary, tune based on data patterns
    if len(forecast_errors_list) > MAX_ERROR_HISTORY:
        forecast_errors_list[:] = forecast_errors_list[-MAX_ERROR_HISTORY:] # In-place slice update

    if debug_mode:
        logging.info(f"  Performance for {date.strftime('%Y-%m-%d')}: MAE={mae:.2f}, MAPE={mape:.2f}% (Error: {actual_sales - forecasted_sales:.2f})")


def simulate_agent_for_n_days(
    num_days: int,
    product_key: str,
    store_key: str,
    data_file_path: str,
    initial_stock: int = 100,
    debug_mode_for_run: bool = False,
    lead_time_days: int = 2,
    safety_stock_factor: float = 0.15,
    min_order_quantity: int = 20,
    service_level: float = DEFAULT_SERVICE_LEVEL,
    forecasting_model: str = "Prophet",        # <--- NEW: Forecasting model parameter
    moving_average_window: int = 7             # <--- NEW: Moving average window parameter
):
    """
    Simulates the inventory agent's behavior over a specified number of days.
    """
    logging.info(f"Starting simulation for {product_key} at {store_key} for {num_days} days.")
    logging.info(f"Simulation parameters: Lead Time={lead_time_days} days, Service Level={service_level*100:.1f}%, Min Order Qty={min_order_quantity}, Forecasting Model='{forecasting_model}', MA Window={moving_average_window}")
    
    # Load all enriched sales data
    full_sales_data = load_enriched_sales_data(data_file_path)
    if full_sales_data.empty:
        logging.error("No sales data available to run simulation.")
        return

    # Ensure full_sales_data 'ds' column is timezone-naive
    if full_sales_data['ds'].dt.tz is not None:
        full_sales_data['ds'] = full_sales_data['ds'].dt.tz_localize(None)

    # Filter data to the simulation period, ensuring it starts from START_DATE
    simulation_start_date = pd.to_datetime(START_DATE)
    simulation_end_date = simulation_start_date + timedelta(days=num_days - 1)
    
    # Ensure we don't go past the actual end of our generated data
    latest_data_date = full_sales_data['ds'].max()
    if simulation_end_date > latest_data_date:
        simulation_end_date = latest_data_date
        num_days = (simulation_end_date - simulation_start_date).days + 1
        if debug_mode_for_run:
            logging.warning(f"Simulation end date adjusted to latest available data: {simulation_end_date.strftime('%Y-%m-%d')}. Running for {num_days} days.")

    # Select only the data relevant for the simulation period
    sim_data_for_loop = full_sales_data[
        (full_sales_data['ds'] >= simulation_start_date) & 
        (full_sales_data['ds'] <= simulation_end_date)
    ].sort_values('ds').reset_index(drop=True)

    if sim_data_for_loop.empty:
        logging.error(f"No simulation data found for the period {simulation_start_date.strftime('%Y-%m-%d')} to {simulation_end_date.strftime('%Y-%m-%d')}.")
        return

    agent_state = load_state(product_key, store_key, debug_mode_for_run)
    current_stock, performance_log, inventory_log, forecast_errors, financial_log = _initialize_daily_run(
        sim_date=sim_data_for_loop['ds'].min(),
        initial_stock=initial_stock,
        agent_state=agent_state,
        debug_mode=debug_mode_for_run
    )

    # Dictionary to hold pending orders
    # Key: date of expected delivery, Value: quantity
    pending_deliveries = {} 
    
    # Main simulation loop
    for i, current_day_data in sim_data_for_loop.iterrows():
        sim_date = current_day_data['ds']
        actual_sales_today_raw = current_day_data['y']

        # 1. Process Deliveries for today (from previous orders)
        deliveries_today = pending_deliveries.pop(sim_date.strftime('%Y-%m-%d'), 0)
        current_stock += deliveries_today
        if debug_mode_for_run and deliveries_today > 0:
            logging.info(f"Day {sim_date.strftime('%Y-%m-%d')}: Received delivery of {deliveries_today} units. Stock now {current_stock}.")

        # 2. Process Sales for today
        ending_stock_after_sales, actual_sales_processed, lost_sales = _process_previous_day_data(
            current_day_data=current_day_data,
            current_stock=current_stock,
            # deliveries_expected_today=0, # Removed as it's handled by current_stock param
            debug_mode=debug_mode_for_run
        )
        current_stock = ending_stock_after_sales

        # For forecasting and ordering, use sales history up to and including *yesterday's* actual sales data.
        # This is the data the model has seen *before* making today's ordering decision.
        sales_history_for_forecast = full_sales_data[full_sales_data['ds'] < sim_date].copy()
        
        # If we need at least X days of history to forecast, handle cold start
        MIN_HISTORY_FOR_FORECAST = 30 # Maintain this threshold for Prophet or MA
        if len(sales_history_for_forecast) < MIN_HISTORY_FOR_FORECAST and forecasting_model != "Actual Sales Data (Baseline)":
            if debug_mode_for_run:
                logging.warning(f"  Insufficient sales history ({len(sales_history_for_forecast)} days) to forecast on {sim_date.strftime('%Y-%m-%d')} with {forecasting_model}. Skipping order generation.")
            order_qty = 0
            forecasted_sales_today_for_log = 0 
            # Log performance/inventory with default 0s if no forecast
            calculate_and_log_performance(
                sim_date, actual_sales_processed, forecasted_sales_today_for_log, performance_log, forecast_errors, debug_mode_for_run
            )
            # Log financial costs for days with no forecast/order logic too
            financial_log.append({
                'date': sim_date.strftime('%Y-%m-%d'),
                'holding_cost': current_stock * HOLDING_COST_PER_UNIT_PER_DAY,
                'ordering_cost': 0,
                'stockout_cost': lost_sales * STOCKOUT_COST_PER_UNIT_LOST_SALE,
                'total_daily_cost': (current_stock * HOLDING_COST_PER_UNIT_PER_DAY) + (lost_sales * STOCKOUT_COST_PER_UNIT_LOST_SALE)
            })
            inventory_log.append({
                'date': sim_date.strftime('%Y-%m-%d'),
                'starting_stock': current_stock + actual_sales_processed + deliveries_today, # Recalculate stock before sales for logging
                'deliveries_today': deliveries_today,
                'actual_sales_today': actual_sales_processed,
                'ending_stock': current_stock,
                'order_placed_qty': order_qty,
                'reorder_point': 0,
                'target_stock_level': 0,
                'forecasted_sales_today': forecasted_sales_today_for_log,
                'lost_sales_today': lost_sales,
                'calculated_safety_stock': 0
            })
            continue # Skip to next day if not enough history and not baseline model
        
        # Generate recommendation (for tomorrow's decision, based on today's state and yesterday's data)
        order_qty = _generate_and_log_recommendation(
            sim_date=sim_date,
            ending_stock=current_stock,
            sales_history_df_for_forecast=sales_history_for_forecast,
            lead_time_days=lead_time_days,
            safety_stock_factor=safety_stock_factor,
            min_order_quantity=min_order_quantity,
            performance_log=performance_log,
            inventory_log=inventory_log,
            financial_log=financial_log,
            forecast_errors_history=forecast_errors,
            current_day_data_actual_sales=actual_sales_processed,
            service_level=service_level,
            deliveries_received_today=deliveries_today,
            lost_sales_today=lost_sales,
            forecasting_model=forecasting_model,        # <--- NEW: Pass forecasting model
            moving_average_window=moving_average_window,# <--- NEW: Pass MA window
            debug_mode=debug_mode_for_run
        )

        # Schedule order delivery
        if order_qty > 0:
            delivery_date = sim_date + timedelta(days=lead_time_days)
            pending_deliveries[delivery_date.strftime('%Y-%m-%d')] = pending_deliveries.get(delivery_date.strftime('%Y-%m-%d'), 0) + order_qty
            if debug_mode_for_run:
                logging.info(f"  Ordered {order_qty} units for delivery on {delivery_date.strftime('%Y-%m-%d')}.")
        
        # At the end of each day's loop, save state. This ensures state is saved daily.
        _update_and_save_agent_state(
            sim_date=sim_date,
            ending_stock=current_stock,
            sales_history_df=full_sales_data[full_sales_data['ds'] <= sim_date].copy(),
            performance_log=performance_log,
            inventory_log=inventory_log,
            forecast_errors=forecast_errors,
            financial_log=financial_log,
            product_key=product_key,
            store_key=store_key,
            debug_mode=debug_mode_for_run
        )

    # Final save of logs to files (redundant if saved daily, but good for robust final state)
    try:
        with open(PERFORMANCE_LOG_FILE, 'w') as f:
            json.dump(performance_log, f, indent=4)
        if debug_mode_for_run:
            logging.info(f"Performance log saved to {PERFORMANCE_LOG_FILE}")

        with open(INVENTORY_LOG_FILE, 'w') as f:
            json.dump(inventory_log, f, indent=4)
        if debug_mode_for_run:
            logging.info(f"Inventory log saved to {INVENTORY_LOG_FILE}")
        
        FINANCIAL_LOG_FILE = INVENTORY_LOG_FILE.replace('inventory_log.json', 'financial_log.json') # Define a path for financial log
        with open(FINANCIAL_LOG_FILE, 'w') as f:
            json.dump(financial_log, f, indent=4)
        if debug_mode_for_run:
            logging.info(f"Financial log saved to {FINANCIAL_LOG_FILE}")

    except Exception as e:
        logging.error(f"Error saving logs: {e}")

    logging.info(f"Simulation completed for {product_key} at {store_key}.")