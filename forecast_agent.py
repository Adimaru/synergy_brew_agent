import logging
import pandas as pd
import os
from datetime import datetime

# Import the main simulation function from our new structured module
from src.agent.core import simulate_agent_for_n_days
from src.data_generation.creator import generate_predefined_data_file

# Import configurations
from config.catalogs import PRODUCT_CATALOG, STORE_LOCATIONS
from config.settings import SIM_DATA_DIR, START_DATE, END_DATE

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # --- Example of how to run the agent directly ---
    # This section is for command-line execution or testing outside Streamlit

    # Define simulation parameters (these would come from Streamlit UI normally)
    SELECTED_PRODUCT_KEY = "espresso_beans"
    SELECTED_STORE_KEY = "downtown_store"
    NUM_DAYS_TO_SIMULATE = 365 # Example: 1 year
    INITIAL_STOCK = 300
    DEBUG_MODE = True # Set to True for more console output
    LEAD_TIME = 2
    SAFETY_STOCK_FACTOR = 0.15
    MIN_ORDER_QTY = 20

    st.info(f"The total available historical data spans from {START_DATE} to {END_DATE}.")

    data_file_name = f"{SELECTED_STORE_KEY}_{SELECTED_PRODUCT_KEY}_enriched_sales_history.csv"
    data_file_path = os.path.join(SIM_DATA_DIR, data_file_name)

    # Ensure data file exists before running simulation
    if not os.path.exists(data_file_path):
        logging.info(f"Sales data for {PRODUCT_CATALOG[SELECTED_PRODUCT_KEY]} at {STORE_LOCATIONS[SELECTED_STORE_KEY]} not found. Generating now...")
        try:
            generate_predefined_data_file(
                SELECTED_PRODUCT_KEY, SELECTED_STORE_KEY,
                spike_probability=0.01, # Example values for generation
                spike_multiplier=1.5
            )
            logging.info("Data generation complete.")
        except Exception as e:
            logging.error(f"Failed to generate data: {e}")
            exit() # Exit if data generation fails

    logging.info(f"Running simulation for {PRODUCT_CATALOG[SELECTED_PRODUCT_KEY]} at {STORE_LOCATIONS[SELECTED_STORE_KEY]}...")

    simulate_agent_for_n_days(
        num_days=NUM_DAYS_TO_SIMULATE,
        product_key=SELECTED_PRODUCT_KEY,
        store_key=SELECTED_STORE_KEY,
        data_file_path=data_file_path,
        initial_stock=INITIAL_STOCK,
        debug_mode_for_run=DEBUG_MODE,
        lead_time_days=LEAD_TIME,
        safety_stock_factor=SAFETY_STOCK_FACTOR,
        min_order_quantity=MIN_ORDER_QTY
    )

    logging.info("Simulation run via forecast_agent.py complete.")