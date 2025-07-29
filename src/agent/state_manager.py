import os
import json
import pandas as pd
import logging
# Prophet is not directly used in state_manager, so it's not strictly needed here
# from prophet import Prophet 

# Import settings for file paths
from config.settings import (
    AGENT_STATE_DIR, INVENTORY_STATE_FILE,
    PERFORMANCE_LOG_FILE, INVENTORY_LOG_FILE, FINANCIAL_LOG_FILE # Ensure these are imported
)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _get_agent_state_filepath(product_key: str, store_key: str) -> str:
    """Constructs the unique file path for an agent's state."""
    filename = f"{product_key}_{store_key}_{INVENTORY_STATE_FILE}"
    return os.path.join(AGENT_STATE_DIR, filename)

def load_state(product_key: str, store_key: str, debug_mode: bool = False) -> dict:
    """
    Loads the agent's state for a specific product and store.
    State includes sales history, last run date, performance, inventory, forecast errors, and financial log.
    """
    filepath = _get_agent_state_filepath(product_key, store_key)
    state = {
        'sales_history_df': pd.DataFrame(),
        'model': None, # Model is not directly saved via JSON and will be retrained
        'last_run_date': None,
        'performance_log': [],
        'inventory_log': [],
        'forecast_errors': [],
        'financial_log': [] 
    }
    
    # Ensure AGENT_STATE_DIR exists before trying to load
    os.makedirs(AGENT_STATE_DIR, exist_ok=True)

    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                saved_state = json.load(f)
            
            # Reconstruct DataFrame
            if 'sales_history_data' in saved_state:
                state['sales_history_df'] = pd.DataFrame(saved_state['sales_history_data'])
                state['sales_history_df']['ds'] = pd.to_datetime(state['sales_history_df']['ds'])

            state['last_run_date'] = saved_state.get('last_run_date')
            state['performance_log'] = saved_state.get('performance_log', [])
            state['inventory_log'] = saved_state.get('inventory_log', [])
            state['forecast_errors'] = saved_state.get('forecast_errors', [])
            state['financial_log'] = saved_state.get('financial_log', []) 

            if debug_mode:
                logging.info(f"Loaded agent state from {filepath}")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {filepath}: {e}. Starting with fresh state.")
        except Exception as e:
            logging.error(f"An unexpected error occurred loading state from {filepath}: {e}. Starting with fresh state.")
    else:
        if debug_mode:
            logging.info(f"No existing agent state found at {filepath}. Starting with fresh state.")
    
    return state

def save_state(state: dict, product_key: str, store_key: str, debug_mode: bool = False):
    """
    Saves the agent's current state.
    """
    filepath = _get_agent_state_filepath(product_key, store_key)
    
    # Prepare state for JSON serialization
    serializable_state = {
        'sales_history_data': state['sales_history_df'].to_dict(orient='records'),
        'last_run_date': state['last_run_date'],
        'performance_log': state['performance_log'],
        'inventory_log': state['inventory_log'],
        'forecast_errors': state['forecast_errors'],
        'financial_log': state['financial_log']
    }
    
    os.makedirs(AGENT_STATE_DIR, exist_ok=True) # Ensure directory exists
    try:
        with open(filepath, 'w') as f:
            json.dump(serializable_state, f, indent=4)
        if debug_mode:
            logging.info(f"Saved agent state to {filepath}")
    except Exception as e:
        logging.error(f"Error saving agent state to {filepath}: {e}")

def clear_all_state_data(debug_mode: bool = False):
    """
    Deletes all agent state files from the AGENT_STATE_DIR
    and all associated log files (performance, inventory, financial).
    """
    if os.path.exists(AGENT_STATE_DIR):
        for filename in os.listdir(AGENT_STATE_DIR):
            filepath = os.path.join(AGENT_STATE_DIR, filename)
            if os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                    if debug_mode:
                        logging.info(f"Deleted agent state file: {filename}")
                except Exception as e:
                    logging.error(f"Error deleting file {filename}: {e}")
    else:
        if debug_mode:
            logging.info(f"Agent state directory does not exist: {AGENT_STATE_DIR}")
    
    # Also delete the log files if they exist
    log_files = [PERFORMANCE_LOG_FILE, INVENTORY_LOG_FILE, FINANCIAL_LOG_FILE]
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
                if debug_mode:
                    logging.info(f"Deleted log file: {os.path.basename(log_file)}")
            except Exception as e:
                logging.error(f"Error deleting log file {os.path.basename(log_file)}: {e}")
    
    if debug_mode:
        logging.info("All agent state and log files cleared.")