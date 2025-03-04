from flask import Blueprint, g, session, request, jsonify
import json
import os
from functools import wraps

config_blueprint = Blueprint('config', __name__)


class ConfigurationManager:
    def __init__(self):
        self.enterprise_configs = {}
        self.justice_configs = {}
        self.standard_config = {}
        self._load_all_configs()

    def _load_all_configs(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(current_dir, 'app', 'config')
        print(f"Loading configs from: {config_dir}")
        self._load_customer_configs(config_dir)

    def _load_customer_configs(self, config_dir):
        try:
            # Load enterprise configs
            enterprise_path = os.path.join(config_dir, 'enterprise_customers_config.json')
            if os.path.exists(enterprise_path):
                with open(enterprise_path) as f:
                    data = json.load(f)
                    self.enterprise_configs = {
                        customer['identifier']: customer 
                        for customer in data.get('enterprises', [])
                    }
                print(f"Loaded enterprise configs for: {list(self.enterprise_configs.keys())}")
            
            # Load justice configs
            justice_path = os.path.join(config_dir, 'justice_customers_config.json')
            if os.path.exists(justice_path):
                with open(justice_path) as f:
                    data = json.load(f)
                    self.justice_configs = {
                        customer['identifier']: customer 
                        for customer in data.get('justice', [])
                    }
                print(f"Loaded justice configs for: {list(self.justice_configs.keys())}")
            
            # Load standard config
            standard_path = os.path.join(config_dir, 'global_customers_config.json')
            if os.path.exists(standard_path):
                with open(standard_path) as f:
                    self.standard_config = json.load(f)
                print("Loaded standard config successfully")

        except Exception as e:
            print(f"Error loading configurations: {str(e)}")
            self.enterprise_configs = {}
            self.justice_configs = {}
            self.standard_config = {}

    def get_customer_type(self, host):
        potential_identifier = host.lower().split('.')[0]
        
        if potential_identifier in self.enterprise_configs:
            return 'enterprise', potential_identifier
            
        if potential_identifier in self.justice_configs:
            return 'justice', potential_identifier
        
        return 'standard', 'default'

    def load_config(self, customer_type, identifier):
        if customer_type == 'enterprise':
            customer_config = self.enterprise_configs.get(identifier)
            if customer_config:
                return {**self.standard_config, **customer_config}
            
        elif customer_type == 'justice':
            customer_config = self.justice_configs.get(identifier)
            if customer_config:
                return {**self.standard_config, **customer_config}
            
        return self.standard_config

    def refresh_configs(self):
        self._load_all_configs()

# Create global config manager instance
config_manager = ConfigurationManager()

@config_blueprint.route('/api/config/customer-type', methods=['POST'])
def get_customer_type_api():
    data = request.get_json()
    subdomain = data.get('subdomain', '').lower()
    
    customer_type, identifier = config_manager.get_customer_type(subdomain)
    config = config_manager.load_config(customer_type, identifier)
    
    return jsonify({
        'type': customer_type,
        'config': config,
        'identifier': identifier,
        'welcome_template': config.get('welcome_template', 'global_welcome.html')
    })

@config_blueprint.route('/api/config/refresh', methods=['POST'])
def refresh_configuration():
    config_manager.refresh_configs()
    return jsonify({
        'status': 'success',
        'message': 'Configurations refreshed'
    })

@config_blueprint.route('/api/config/status')
def config_status():
    return jsonify({
        'enterprise_customers': bool(config_manager.enterprise_configs),
        'justice_customers': bool(config_manager.justice_configs),
        'standard_config': bool(config_manager.standard_config)
    })