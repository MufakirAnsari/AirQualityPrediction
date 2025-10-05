import yaml
import os
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration loader and manager"""
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            return self.get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration if file loading fails"""
        return {
            'data': {
                'raw_file': 'Airquality9.csv',
                'target_classification': 'AQI_Category',
                'target_regression': 'PM2.5(µg/m³)'
            },
            'training': {
                'test_size': 0.2,
                'random_state': 42
            },
            'paths': {
                'artifacts_dir': 'artifacts',
                'models_dir': 'artifacts/models'
            }
        }
    
    def get(self, key_path, default=None):
        """
        Get configuration value using dot notation
        e.g., config.get('data.target_regression')
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Configuration key '{key_path}' not found, using default: {default}")
            return default
    
    def set(self, key_path, value):
        """
        Set configuration value using dot notation
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set the target key
        config_ref[keys[-1]] = value
        logger.info(f"Configuration updated: {key_path} = {value}")
    
    def save_config(self, path=None):
        """Save current configuration to file"""
        save_path = path or self.config_path
        try:
            with open(save_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def create_directories(self):
        """Create all directories specified in configuration"""
        paths_to_create = [
            self.get('paths.artifacts_dir'),
            self.get('paths.models_dir'),
            self.get('paths.evaluation_dir'),
            self.get('paths.eda_dir'),
            self.get('paths.plots_dir'),
            self.get('paths.logs_dir')
        ]
        
        for path in paths_to_create:
            if path:
                os.makedirs(path, exist_ok=True)
        
        logger.info("Required directories created")
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        log_level = getattr(logging, self.get('logging.level', 'INFO'))
        log_format = self.get('logging.format', '%(asctime)s - %(levelname)s - %(message)s')
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[]
        )
        
        # Console handler
        if self.get('logging.console_logging', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(console_handler)
        
        # File handler
        if self.get('logging.file_logging', False):
            log_dir = self.get('paths.logs_dir', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(f'{log_dir}/pipeline.log')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(file_handler)
        
        logger.info("Logging configured")

# Global configuration instance
config = Config()

def get_config():
    """Get global configuration instance"""
    return config

def reload_config(config_path="config.yaml"):
    """Reload configuration from file"""
    global config
    config = Config(config_path)
    return config