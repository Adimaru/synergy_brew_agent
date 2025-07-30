# src/data_manager/config_manager.py
import json
import os
from config.settings import CUSTOM_PRODUCTS_FILE, CUSTOM_STORES_FILE

def _load_json_file(file_path, default_value={}):
    """Helper to load JSON data from a file, returning default if file not found."""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump(default_value, f, indent=4)
        return default_value
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {file_path}. Returning default.")
        return default_value

def _save_json_file(data, file_path):
    """Helper to save JSON data to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True) # Ensure directory exists
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_custom_products():
    """Loads custom product definitions from file."""
    return _load_json_file(CUSTOM_PRODUCTS_FILE)

def save_custom_products(products_data):
    """Saves custom product definitions to file."""
    _save_json_file(products_data, CUSTOM_PRODUCTS_FILE)

def load_custom_stores():
    """Loads custom store definitions from file."""
    return _load_json_file(CUSTOM_STORES_FILE)

def save_custom_stores(stores_data):
    """Saves custom store definitions to file."""
    _save_json_file(stores_data, CUSTOM_STORES_FILE)