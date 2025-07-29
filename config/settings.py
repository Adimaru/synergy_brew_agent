import os
from datetime import datetime

# --- Base Directory ---
# This assumes settings.py is in synergy_brew_agent/config/
# and BASE_DIR should point to synergy_brew_agent/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Directories ---
DATA_DIR = os.path.join(BASE_DIR, 'data')
SALES_HISTORY_DIR = os.path.join(DATA_DIR, 'sales_history')
SIM_DATA_DIR = os.path.join(DATA_DIR, 'sim_data')
AGENT_STATE_DIR = os.path.join(DATA_DIR, 'agent_state') # Directory to save agent's state

# Log Directory
LOG_DIR = os.path.join(DATA_DIR, 'logs') # Directory for performance and inventory logs

# --- Log File Paths ---
PERFORMANCE_LOG_FILE = os.path.join(LOG_DIR, 'performance_log.json')
INVENTORY_LOG_FILE = os.path.join(LOG_DIR, 'inventory_log.json')
FINANCIAL_LOG_FILE = os.path.join(LOG_DIR, 'financial_log.json') # New financial log file

# --- Agent State File Name ---
INVENTORY_STATE_FILE = 'agent_state.json' # Filename for the agent's saved state

# --- Simulation Date Range ---
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)

# --- Holiday Data (Example: New Year's Day, Christmas Day) ---
HOLIDAYS = [
    '2024-01-01', # New Year's Day
    '2024-12-25', # Christmas Day
    # Add more holidays as needed for your specific context
]

# --- Product and Store Catalogs (for Streamlit selection) ---
PRODUCT_CATALOG = {
    "Latte": "Latte",
    "Espresso": "Espresso",
    "Cappuccino": "Cappuccino",
    "CaramelMacchiato": "Caramel Macchiato", # Make sure this is consistent if you have this product
    "PikePlaceRoast": "Pike Place Roast",
    "ColdBrew": "Cold Brew",
    "GreenTeaLatte": "Green Tea Latte",
    "Frappuccino": "Frappuccino",
    # Add other items like Croissant and Muffin if they are coffee products
    # If Croissant/Muffin are not coffee products, they should not be in PRODUCT_CATALOG if the data generation is only for coffee
    # For now, let's assume they are "products" in general, but might need different data generation logic.
    # For now, I'll add them with generic config.
    "Croissant": "Croissant",
    "Muffin": "Muffin"
}

STORE_LOCATIONS = {
    "Store_A_Downtown": "Store A - Downtown",
    "Store_B_Campus": "Store B - Campus", # Changed from Suburb
    "Store_C_Suburban": "Store C - Suburban", # Changed from Mall
    "Store_D_Airport": "Store D - Airport",
    "Store_E_Mall": "Store E - Mall"
}


# --- Product Configurations (for detailed settings per product, including sales patterns) ---
PRODUCT_CONFIGS = {
    "Latte": {
        "name": "Latte",
        "base_sales": 100,
        "weekly_peak_factor": 1.2, # 20% higher on weekends
        "summer_factor": 0.9,      # 10% lower in summer
        "winter_factor": 1.1       # 10% higher in winter
    },
    "Espresso": {
        "name": "Espresso",
        "base_sales": 70,
        "weekly_peak_factor": 1.1,
        "summer_factor": 1.0,
        "winter_factor": 1.05
    },
    "Cappuccino": {
        "name": "Cappuccino",
        "base_sales": 90,
        "weekly_peak_factor": 1.25,
        "summer_factor": 0.95,
        "winter_factor": 1.15
    },
    "CaramelMacchiato": {
        "name": "Caramel Macchiato",
        "base_sales": 80,
        "weekly_peak_factor": 1.3,
        "summer_factor": 1.1, # More popular in summer
        "winter_factor": 0.9
    },
    "PikePlaceRoast": {
        "name": "Pike Place Roast",
        "base_sales": 120,
        "weekly_peak_factor": 1.1,
        "summer_factor": 1.0,
        "winter_factor": 1.0
    },
    "ColdBrew": {
        "name": "Cold Brew",
        "base_sales": 60,
        "weekly_peak_factor": 1.05,
        "summer_factor": 1.5, # Very popular in summer
        "winter_factor": 0.7
    },
    "GreenTeaLatte": {
        "name": "Green Tea Latte",
        "base_sales": 50,
        "weekly_peak_factor": 1.1,
        "summer_factor": 1.0,
        "winter_factor": 1.0
    },
    "Frappuccino": {
        "name": "Frappuccino",
        "base_sales": 75,
        "weekly_peak_factor": 1.3,
        "summer_factor": 1.3, # Popular in summer
        "winter_factor": 0.8
    },
    # Generic configs for Croissant and Muffin, adjust as needed
     "Croissant": {
        "name": "Croissant",
        "base_sales": 40,
        "weekly_peak_factor": 1.3,
        "summer_factor": 1.0,
        "winter_factor": 1.0
    },
    "Muffin": {
        "name": "Muffin",
        "base_sales": 30,
        "weekly_peak_factor": 1.2,
        "summer_factor": 1.0,
        "winter_factor": 1.0
    }
}

# --- Default Simulation Parameters ---
DEFAULT_INITIAL_STOCK = 100
DEFAULT_NUM_DAYS_TO_SIMULATE = 365 # Simulate one year by default
DEFAULT_LEAD_TIME = 2              # Days for an order to arrive
DEFAULT_SAFETY_STOCK_FACTOR = 0.15 # Multiplier for demand standard deviation
DEFAULT_MIN_ORDER_QTY = 20         # Minimum quantity for an order
DEFAULT_SERVICE_LEVEL = 0.95       # Desired service level for safety stock calculation

# --- Forecasting Parameters ---
DEFAULT_FORECAST_HORIZON_DAYS = 30 # How many days into the future to forecast

# --- Financial Parameters ---
HOLDING_COST_PER_UNIT_PER_DAY = 0.05       # Cost to hold one unit for one day (e.g., $0.05 per unit per day)
ORDERING_COST_PER_ORDER = 55.0             # Fixed cost incurred each time an order is placed (e.g., $55 per order)
STOCKOUT_COST_PER_UNIT_LOST_SALE = 5.0     # Penalty cost for each unit of lost sale (e.g., $5.00 per unit)