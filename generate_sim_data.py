import logging
import os

# Import the data generation functions from our new structured module
from src.data_generation.creator import generate_predefined_data_file, generate_custom_product_data

# Import configurations
from config.catalogs import PRODUCT_CONFIGS, STORE_CONFIGS, PRODUCT_CATALOG, STORE_LOCATIONS
from config.settings import SIM_DATA_DIR # Ensure SIM_DATA_DIR is imported from settings

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_all_predefined_data(
    spike_probability: float = 0.01, 
    spike_multiplier: float = 1.5,
    force_recreate: bool = False
):
    """
    Generates sales data for all predefined product-store combinations.
    """
    logging.info("Starting generation of all predefined simulation data...")
    os.makedirs(SIM_DATA_DIR, exist_ok=True)

    for product_key in PRODUCT_CONFIGS.keys():
        for store_key in STORE_CONFIGS.keys():
            try:
                generate_predefined_data_file(
                    product_key,
                    store_key,
                    spike_probability=spike_probability,
                    spike_multiplier=spike_multiplier,
                    force_recreate=force_recreate
                )
                logging.info(f"Data generated for {PRODUCT_CATALOG[product_key]} at {STORE_LOCATIONS[store_key]}.")
            except Exception as e:
                logging.error(f"Failed to generate data for {product_key} at {store_key}: {e}")
    logging.info("All predefined simulation data generation complete.")


if __name__ == "__main__":
    # Example usage: Generate all predefined data with default spike parameters
    generate_all_predefined_data(
        spike_probability=0.01,
        spike_multiplier=1.5,
        force_recreate=True # Set to True to regenerate all files every time this script runs
    )
    # You can also use generate_custom_product_data if needed
    # Example:
    # generate_custom_product_data(
    #     product_name="Summer Delight",
    #     base_sales=150,
    #     weekly_peak_factor=1.3,
    #     summer_factor=1.5,
    #     winter_factor=0.8,
    #     store_key="urban_cafe",
    #     spike_probability=0.02,
    #     spike_multiplier=2.0
    # )