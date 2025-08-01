# src/agent/core.py

import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import timedelta, date
from typing import Dict, Any, List, Optional

# Ensure src directory is in the path for imports
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.settings import (
    SIM_DATA_DIR,
    HOLDING_COST_PER_UNIT_PER_DAY,
    ORDERING_COST_PER_ORDER,
    STOCKOUT_COST_PER_UNIT_LOST_SALE,
    PRICE_PER_UNIT, # Added PRICE_PER_UNIT
    FINANCIAL_LOG_FILE,
    INVENTORY_LOG_FILE,
    PERFORMANCE_LOG_FILE,
    DEFAULT_FORECAST_HORIZON_DAYS,
    START_DATE,
    END_DATE
)

# Optional: Add prophet and scipy if they're installed, otherwise provide a fallback
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
except ImportError:
    Prophet = None
    logging.warning("Prophet not found. Prophet forecasting will not be available.")

try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not found. Z-score calculation will use a simplified lookup table.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class InventorySimulationAgent:
    """
    A class to simulate an inventory management agent for a single product at a single store.
    This class encapsulates all the state and logic for a simulation run, avoiding global state.
    """
    def __init__(
        self,
        product_key: str,
        store_key: str,
        initial_stock: int,
        lead_time_days: int,
        min_order_quantity: int,
        service_level: float,
        forecasting_model: str,
        moving_average_window: int,
        prophet_params: Dict[str, Any],
        debug_mode: bool
    ):
        """
        Initializes the simulation agent with a set of parameters.
        """
        self.product_key = product_key
        self.store_key = store_key
        self.forecasting_model = forecasting_model
        self.forecasting_params = self._get_forecasting_parameters(
            forecasting_model, moving_average_window, prophet_params
        )
        self.lead_time_days = lead_time_days
        self.min_order_quantity = min_order_quantity
        self.service_level = service_level
        self.debug_mode = debug_mode

        # Initialize the simulation state
        self.state = {
            'inventory': initial_stock,
            'days_until_delivery': {},  # {order_id: days_left}
            'order_history': [],
            'financial_log': [],
            'inventory_log': [],
            'performance_log': [],
            'total_sales': 0,
            'total_lost_sales': 0,
        }

    @staticmethod
    def _get_forecasting_parameters(
        forecasting_model: str, 
        moving_average_window: int, 
        prophet_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Returns a dictionary of parameters for the chosen forecasting model.
        """
        if forecasting_model == "Moving Average":
            return {'window': moving_average_window}
        elif forecasting_model == "Prophet":
            # Ensure prophet_params is a dictionary, defaulting to an empty one if None
            return {'prophet_params': prophet_params if prophet_params is not None else {}}
        return {}

    @staticmethod
    def _check_and_load_sales_data(
        data_file_path: str, 
        debug_mode: bool
    ) -> Optional[pd.DataFrame]:
        """
        Loads sales data, fills missing dates, and returns a DataFrame.
        Returns None if the file does not exist or an error occurs.
        """
        if not os.path.exists(data_file_path):
            logging.error(f"Sales data file not found: {data_file_path}")
            return None

        try:
            df = pd.read_csv(data_file_path)
            # Ensure the necessary columns exist
            if 'ds' not in df.columns or 'y' not in df.columns:
                logging.error(f"Required columns 'ds' and 'y' not found in {data_file_path}")
                return None

            df['ds'] = pd.to_datetime(df['ds'])
            df = df.sort_values(by='ds').reset_index(drop=True)

            full_date_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
            full_df = pd.DataFrame(full_date_range, columns=['ds'])
            full_df = pd.merge(full_df, df, on='ds', how='left')
            full_df['y'] = full_df['y'].fillna(0)

            logging.info(f"Loaded {len(full_df)} rows of sales data from {data_file_path}.")
            if debug_mode:
                logging.debug(f"Sales data head:\n{full_df.head()}")
            return full_df

        except Exception as e:
            logging.error(f"Failed to load or process sales data from {data_file_path}: {e}")
            return None

    def _calculate_z_score(self, service_level: float) -> float:
        """
        Calculates the z-score for a given service level.
        Uses scipy if available, otherwise uses a simplified lookup table.
        """
        if SCIPY_AVAILABLE:
            # norm.ppf calculates the percent point function (inverse of CDF)
            return norm.ppf(service_level)
        else:
            # Fallback to a simplified lookup table
            z_score_map = {0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
            # Use the closest z-score if an exact match isn't found
            return z_score_map.get(service_level, 1.65)

    def _calculate_safety_stock(self, forecast_df: pd.DataFrame) -> int:
        """
        Calculates safety stock based on forecast error variance and desired service level.
        Assumes `forecast_df` contains 'y' (actual) and 'yhat' (forecast) columns.
        """
        if forecast_df.empty or len(forecast_df) < 2:
            return 0

        # Calculate forecast errors
        forecast_df['error'] = forecast_df['y'] - forecast_df['yhat']
        forecast_error_std = forecast_df['error'].std()

        # Get the z-score for the desired service level
        z_score = self._calculate_z_score(self.service_level)
        
        # This formula assumes a normal distribution of forecast errors.
        safety_stock = z_score * forecast_error_std
        return max(0, int(np.ceil(safety_stock)))

    def _forecast_demand(self, historical_sales: pd.DataFrame, forecast_horizon_days: int) -> pd.DataFrame:
        """
        Predicts future demand using the specified model.
        """
        if self.forecasting_model == "Moving Average":
            window = self.forecasting_params.get('window', 7)
            if len(historical_sales) < window:
                logging.warning(f"Historical sales data ({len(historical_sales)}) is less than MA window ({window}). Returning 0 forecast.")
                return pd.DataFrame({'yhat': [0] * forecast_horizon_days})

            last_n_sales = historical_sales['y'].tail(window)
            moving_average = last_n_sales.mean()
            forecast_values = [moving_average] * forecast_horizon_days
            return pd.DataFrame({'yhat': forecast_values})

        elif self.forecasting_model == "Prophet" and Prophet is not None:
            prophet_params = self.forecasting_params.get('prophet_params', {})
            try:
                m = Prophet(**prophet_params)
                m.fit(historical_sales)
                future = m.make_future_dataframe(periods=forecast_horizon_days)
                forecast = m.predict(future)
                return forecast[['ds', 'yhat']].tail(forecast_horizon_days)
            except Exception as e:
                logging.error(f"Prophet forecasting failed: {e}. Using a naive forecast instead.")
                last_day_sales = historical_sales['y'].iloc[-1] if not historical_sales.empty else 0
                return pd.DataFrame({'yhat': [last_day_sales] * forecast_horizon_days})

        elif self.forecasting_model == "Actual Sales Data (Baseline)":
            # A more useful baseline is a naive forecast (last known value).
            last_day_sales = historical_sales['y'].iloc[-1] if not historical_sales.empty else 0
            future_dates = pd.date_range(start=historical_sales['ds'].max() + timedelta(days=1), periods=forecast_horizon_days)
            return pd.DataFrame({'ds': future_dates, 'yhat': [last_day_sales] * forecast_horizon_days})
            
        else:
            logging.error(f"Unknown forecasting model: {self.forecasting_model}. Returning a forecast of 0.")
            return pd.DataFrame({'yhat': [0] * forecast_horizon_days})

    def _run_daily_agent_logic(self, sim_date: pd.Timestamp, historical_data_for_forecast: pd.DataFrame, actual_daily_sales: int, lost_sales_today: int) -> int:
        """
        Contains the core logic for the agent's daily decision-making.
        """
        # 1. Forecast Demand for the lead time period
        forecast_df = self._forecast_demand(
            historical_sales=historical_data_for_forecast,
            forecast_horizon_days=self.lead_time_days
        )
        
        # 2. Calculate Safety Stock
        safety_stock = 0
        if self.forecasting_model == "Prophet" and len(historical_data_for_forecast) >= 2:
            try:
                m_historical = Prophet(**self.forecasting_params['prophet_params'])
                m_historical.fit(historical_data_for_forecast)
                forecast_on_historical = m_historical.predict(historical_data_for_forecast)
                forecast_historical_data = pd.merge(
                    historical_data_for_forecast,
                    forecast_on_historical[['ds', 'yhat']],
                    on='ds',
                    how='left'
                )
                safety_stock = self._calculate_safety_stock(forecast_historical_data)
            except Exception as e:
                logging.error(f"Prophet failed to calculate safety stock: {e}. Falling back to simpler method.")
                # Fallback to a simpler safety stock calculation
                z_score = self._calculate_z_score(self.service_level)
                std_dev_daily_sales = historical_data_for_forecast['y'].std() if len(historical_data_for_forecast) > 1 else 0
                safety_stock = int(np.ceil(z_score * std_dev_daily_sales * np.sqrt(self.lead_time_days)))
        else:
            # Standard safety stock formula for non-Prophet models
            z_score = self._calculate_z_score(self.service_level)
            std_dev_daily_sales = historical_data_for_forecast['y'].std() if len(historical_data_for_forecast) > 1 else 0
            # Note: This formula assumes demand is normally distributed and independent from day to day.
            safety_stock = int(np.ceil(z_score * std_dev_daily_sales * np.sqrt(self.lead_time_days)))
        
        # 3. Calculate forecasted demand during lead time (DDLT)
        forecasted_demand_during_lead_time = int(np.ceil(forecast_df['yhat'].sum()))

        # 4. Calculate Reorder Point (ROP) and Target Inventory Level
        target_inventory = forecasted_demand_during_lead_time + safety_stock
        pending_orders_inventory = sum(
            self.state['order_history'][order_id]['quantity']
            for order_id in self.state['days_until_delivery']
        )
        current_inventory_position = self.state['inventory'] + pending_orders_inventory
        
        # 5. Determine Order Quantity
        order_qty = max(0, target_inventory - current_inventory_position)

        # 6. Apply Minimum Order Quantity constraint
        if 0 < order_qty < self.min_order_quantity:
            order_qty = self.min_order_quantity
        
        # Log the daily state
        self._log_daily_state(
            sim_date,
            actual_daily_sales,
            lost_sales_today,
            pending_orders_inventory,
            forecasted_demand_during_lead_time,
            safety_stock,
            order_qty
        )
        
        return order_qty

    def _log_daily_state(self, sim_date: pd.Timestamp, actual_sales_today: int, lost_sales_today: int, pending_orders_inventory: int, forecast: int, safety_stock: int, order_placed: int):
        """
        Logs the daily inventory and financial state.
        """
        # Log inventory state
        self.state['inventory_log'].append({
            'date': sim_date,
            'actual_sales_today': actual_sales_today,
            'lost_sales_today': lost_sales_today,
            'starting_stock': self.state['inventory'] + pending_orders_inventory,
            'ending_stock': self.state['inventory'],
            'pending_orders': pending_orders_inventory,
            'lead_time_forecast': forecast,
            'safety_stock': safety_stock,
            'order_placed': order_placed,
            'service_level_target': self.service_level
        })

        # Calculate daily costs
        holding_cost = self.state['inventory'] * HOLDING_COST_PER_UNIT_PER_DAY
        stockout_cost = lost_sales_today * STOCKOUT_COST_PER_UNIT_LOST_SALE
        ordering_cost = ORDERING_COST_PER_ORDER if order_placed > 0 else 0
        total_daily_cost = holding_cost + ordering_cost + stockout_cost
        
        # Calculate daily revenue
        daily_revenue = actual_sales_today * PRICE_PER_UNIT

        # Log financial state
        self.state['financial_log'].append({
            'date': sim_date,
            'holding_cost': holding_cost,
            'ordering_cost': ordering_cost,
            'stockout_cost': stockout_cost,
            'total_daily_cost': total_daily_cost,
            'daily_revenue': daily_revenue, # Added daily revenue
            'deliveries_received': 0 # Will be updated later in the main loop
        })

    def _calculate_kpis(self) -> None:
        """
        Calculates key performance indicators (KPIs) for the simulation and logs them.
        """
        total_cost = sum(d['total_daily_cost'] for d in self.state['financial_log'])
        total_revenue = sum(d['daily_revenue'] for d in self.state['financial_log'])
        total_lost_sales_units = self.state['total_lost_sales']
        total_actual_demand = self.state['total_sales'] + total_lost_sales_units
        service_level_achieved = (self.state['total_sales'] / total_actual_demand) if total_actual_demand > 0 else 1.0
        
        total_ordering_cost = sum(d['ordering_cost'] for d in self.state['financial_log'])
        total_holding_cost = sum(d['holding_cost'] for d in self.state['financial_log'])
        total_stockout_cost = sum(d['stockout_cost'] for d in self.state['financial_log'])

        kpis = {
            'product_key': self.product_key,
            'store_key': self.store_key,
            'total_simulation_cost': total_cost,
            'total_revenue': total_revenue,
            'total_profit': total_revenue - total_cost,
            'service_level_achieved': service_level_achieved,
            'total_lost_sales_units': total_lost_sales_units,
            'total_ordering_cost': total_ordering_cost,
            'total_holding_cost': total_holding_cost,
            'total_stockout_cost': total_stockout_cost,
        }
        self.state['performance_log'].append(kpis)
        logging.info("Simulation KPIs calculated.")
        if self.debug_mode:
            logging.info(f"Final KPIs:\n{json.dumps(kpis, indent=2)}")

    def run_simulation(self, num_days: int, data_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Main function to run the inventory simulation for a specified number of days.
        """
        logging.info(f"Starting simulation for {self.product_key} at {self.store_key} for {num_days} days.")
        
        all_sales_data = self._check_and_load_sales_data(data_file_path, self.debug_mode)
        if all_sales_data is None or all_sales_data.empty:
            logging.error("Failed to load sales data. Simulation aborted.")
            return None

        sim_start_date = START_DATE
        sim_end_date = sim_start_date + timedelta(days=num_days - 1)
        start_ts = pd.Timestamp(sim_start_date)
        end_ts = pd.Timestamp(sim_end_date)
        
        if start_ts < all_sales_data['ds'].min() or end_ts > all_sales_data['ds'].max():
            logging.error(f"Simulation period {sim_start_date} to {sim_end_date} is outside the available data range.")
            return None
        
        sim_sales_data = all_sales_data[
            (all_sales_data['ds'] >= start_ts) & 
            (all_sales_data['ds'] <= end_ts)
        ].reset_index(drop=True)

        # Loop through each day of the simulation
        for i, row in sim_sales_data.iterrows():
            sim_date = row['ds']
            actual_sales_today = row['y']

            # 1. Receive incoming orders and process deliveries
            deliveries_received_today = 0
            orders_to_remove = []
            for order_id, days_left in list(self.state['days_until_delivery'].items()):
                if days_left <= 1: # Delivered today or delivery is imminent
                    deliveries_received_today += self.state['order_history'][order_id]['quantity']
                    orders_to_remove.append(order_id)
            
            for order_id in orders_to_remove:
                del self.state['days_until_delivery'][order_id]

            self.state['inventory'] += deliveries_received_today
            
            # 2. Process Sales
            sales_processed_today = min(self.state['inventory'], actual_sales_today)
            lost_sales_today = actual_sales_today - sales_processed_today
            self.state['inventory'] -= sales_processed_today

            self.state['total_sales'] += sales_processed_today
            self.state['total_lost_sales'] += lost_sales_today
            
            # 3. Agent's Decision: Generate recommendation and log state
            historical_data_for_forecast = sim_sales_data[sim_sales_data['ds'] < sim_date].copy()
            order_qty = self._run_daily_agent_logic(
                sim_date=sim_date,
                historical_data_for_forecast=historical_data_for_forecast,
                actual_daily_sales=sales_processed_today,
                lost_sales_today=lost_sales_today
            )
            
            # Update the deliveries_received count in the financial log for today
            if self.state['financial_log'] and self.state['financial_log'][-1]['date'] == sim_date:
                self.state['financial_log'][-1]['deliveries_received'] = deliveries_received_today

            # 4. Place a new order if needed
            if order_qty > 0:
                order_id = len(self.state['order_history'])
                self.state['order_history'].append({
                    'id': order_id,
                    'quantity': order_qty,
                    'order_date': sim_date
                })
                self.state['days_until_delivery'][order_id] = self.lead_time_days
                
            # 5. Update pending order days for the next day
            for order_id in list(self.state['days_until_delivery'].keys()):
                self.state['days_until_delivery'][order_id] -= 1

        logging.info(f"Simulation for {self.product_key} at {self.store_key} finished.")
        
        self._calculate_kpis()

        # Store results in a dictionary and return
        results = {
            'product_key': self.product_key,
            'store_key': self.store_key,
            'financial_log': self.state['financial_log'],
            'inventory_log': self.state['inventory_log'],
            'performance_log': self.state['performance_log'],
            'total_sales': self.state['total_sales'],
            'total_lost_sales': self.state['total_lost_sales'],
            'final_inventory': self.state['inventory']
        }
        
        return results

# The `clear_all_state_data` function is kept as a standalone utility.
def clear_all_state_data(debug_mode: bool = False):
    """
    Deletes all saved state files and simulation data files.
    """
    files_to_delete = [FINANCIAL_LOG_FILE, INVENTORY_LOG_FILE, PERFORMANCE_LOG_FILE]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            if debug_mode:
                logging.info(f"Deleted state file: {file_path}")
    
    # Delete simulation data files
    if os.path.exists(SIM_DATA_DIR):
        for file_name in os.listdir(SIM_DATA_DIR):
            file_path = os.path.join(SIM_DATA_DIR, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                if debug_mode:
                    logging.info(f"Deleted simulation data file: {file_path}")
    else:
        logging.warning(f"Simulation data directory not found: {SIM_DATA_DIR}")

    logging.info("All saved state and simulation data files have been deleted.")
