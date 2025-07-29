import pandas as pd
import os
import logging
from typing import Optional

# Import the core sales data generator
from src.data_generation.generator import generate_sales_data

# Import configurations
from config.catalogs import PRODUCT_CONFIGS, STORE_CONFIGS, PRODUCT_CATALOG, STORE_LOCATIONS
from config.settings import SIM_DATA_DIR

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_predefined_data_file(
    product_key: str,
    store_key: str,
    spike_probability: float = 0.0,
    spike_multiplier: float = 1.0,
    force_recreate: bool = False
) -> str:
    """
    Generates and saves sales data for a predefined product and store.
    """
    product_config = PRODUCT_CONFIGS.get(product_key)
    store_config = STORE_CONFIGS.get(store_key)

    if not product_config:
        logging.error(f"Product configuration for '{product_key}' not found.")
        raise ValueError(f"Product configuration for '{product_key}' not found.")
    if not store_config:
        logging.error(f"Store configuration for '{store_key}' not found.")
        raise ValueError(f"Store configuration for '{store_key}' not found.")

    file_name = f"{store_key}_{product_key}_enriched_sales_history.csv"
    file_path = os.path.join(SIM_DATA_DIR, file_name)

    os.makedirs(SIM_DATA_DIR, exist_ok=True)

    if os.path.exists(file_path) and not force_recreate:
        logging.info(f"Sales data for {PRODUCT_CATALOG.get(product_key, product_key)} at {STORE_LOCATIONS.get(store_key, store_key)} already exists. Skipping generation.")
        return file_path

    logging.info(f"Generating sales data for {PRODUCT_CATALOG.get(product_key, product_key)} at {STORE_LOCATIONS.get(store_key, store_key)}...")
    
    # Combine product and store specific factors
    # For now, base_sales from product config. If store has a sales factor, apply it here.
    # Assuming store_config might have a 'base_sales_multiplier'
    effective_base_sales = product_config['base_sales'] * store_config.get('base_sales_multiplier', 1.0)

    df = generate_sales_data(
        base_sales=effective_base_sales,
        weekly_peak_factor=product_config['weekly_peak_factor'],
        summer_factor=product_config['summer_factor'],
        winter_factor=product_config['winter_factor'],
        spike_probability=spike_probability,
        spike_multiplier=spike_multiplier,
        seed=product_config.get('seed', 42) + store_config.get('seed_offset', 0) # Combine seeds for unique data
    )
    df.to_csv(file_path, index=False)
    logging.info(f"Generated data saved to {file_path}")
    return file_path


def generate_custom_product_data(
    product_name: str,
    base_sales: int,
    weekly_peak_factor: float,
    summer_factor: float,
    winter_factor: float,
    store_key: str,
    spike_probability: float = 0.0,
    spike_multiplier: float = 1.0,
    seed: int = 100 # Use a different seed for custom products
) -> str:
    """
    Generates and saves sales data for a custom product.
    """
    # Create a unique key for the custom product
    product_key = f"custom_product_{product_name.replace(' ', '_').lower()}"

    file_name = f"{store_key}_{product_key}_enriched_sales_history.csv"
    file_path = os.path.join(SIM_DATA_DIR, file_name)

    os.makedirs(SIM_DATA_DIR, exist_ok=True)

    logging.info(f"Generating custom sales data for '{product_name}' at {STORE_LOCATIONS.get(store_key, store_key)}...")

    df = generate_sales_data(
        base_sales=base_sales,
        weekly_peak_factor=weekly_peak_factor,
        summer_factor=summer_factor,
        winter_factor=winter_factor,
        spike_probability=spike_probability,
        spike_multiplier=spike_multiplier,
        seed=seed + hash(product_name) % 1000 # Add product name to seed for more uniqueness
    )
    df.to_csv(file_path, index=False)
    logging.info(f"Generated custom data saved to {file_path}")
    
    # Optionally, add custom product to PRODUCT_CATALOG and PRODUCT_CONFIGS temporarily
    # for immediate selection in the app without restarting.
    # This requires modifying global variables, which is generally not ideal,
    # but for a Streamlit demo, it can enhance UX.
    # A more robust solution would involve a persistent config update or database.

    PRODUCT_CATALOG[product_key] = product_name
    PRODUCT_CONFIGS[product_key] = {
        "base_sales": base_sales,
        "weekly_peak_factor": weekly_peak_factor,
        "summer_factor": summer_factor,
        "winter_factor": winter_factor,
        "seed": seed # Storing the base seed
    }
    
    return file_path