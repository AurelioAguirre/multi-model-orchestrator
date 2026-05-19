import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> 'AppConfig':
    """Load configuration from YAML file"""
    if config_path is None:
        # Default to resources/config.yml relative to project root
        config_path = Path(__file__).parent.parent.parent / "resources" / "config.yml"
    
    return AppConfig(config_path)


class AppConfig:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "resources" / "config.yml"
        
        self._config = self._load_config(Path(config_path))
        self._validate_config()
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Override with environment variables if they exist
        config = self._override_with_env(config)
        return config
    
    def _override_with_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Override server settings
        if 'server' in config:
            config['server']['host'] = os.getenv('HOST', config['server'].get('host', '0.0.0.0'))
            config['server']['port'] = int(os.getenv('PORT', config['server'].get('port', 8000)))
            config['server']['log_level'] = os.getenv('LOG_LEVEL', config['server'].get('log_level', 'info'))
        
        # Override model settings
        if 'model' in config:
            config['model']['base_dir'] = os.getenv('MODEL_BASE_DIR', config['model'].get('base_dir', './models'))
            config['model']['default_model'] = os.getenv('DEFAULT_MODEL', config['model'].get('default_model', ''))
        
        return config
    
    def _validate_config(self):
        required_sections = ['server', 'model', 'inference']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required config section: {section}")
    
    # Server configuration
    @property
    def app_name(self) -> str:
        return self._config.get('app_name', 'LLM Tensor Server')
    
    @property
    def version(self) -> str:
        return self._config.get('version', '1.0.0')
    
    @property
    def host(self) -> str:
        return self._config['server']['host']
    
    @property
    def port(self) -> int:
        return self._config['server']['port']
    
    @property
    def log_level(self) -> str:
        return self._config['server']['log_level']
    
    @property
    def access_log(self) -> bool:
        return self._config['server'].get('access_log', True)
    
    # Model configuration
    @property
    def model_base_dir(self) -> str:
        return self._config['model']['base_dir']
    
    @property
    def default_model(self) -> str:
        return self._config['model'].get('default_model', '')
    
    def get_model_path(self, model_name: str = None) -> str:
        """Get full path to a model directory or return HuggingFace model ID"""
        model = model_name or self.default_model
        if not model:
            raise ValueError("No model specified and no default_model configured")
        
        # If it's an absolute path, return as-is
        if os.path.isabs(model):
            return model
            
        # If it looks like a HuggingFace model ID (contains slash but no path separators), 
        # check if it exists as a local directory first
        if "/" in model and not model.startswith("./") and not model.startswith("../"):
            local_path = os.path.join(self.model_base_dir, model)
            if os.path.exists(local_path) and os.path.isdir(local_path):
                # Local directory exists, use it
                return local_path
            else:
                # No local directory, treat as HuggingFace model ID
                return model
        
        # Otherwise, treat as relative path to base_dir
        return os.path.join(self.model_base_dir, model)
    
    def get_engine_dir(self, model_name: str = None) -> str:
        """Get path to TensorRT engines for a model"""
        model_path = self.get_model_path(model_name)
        return os.path.join(model_path, "tensorrt-engines")
    
    def get_tokenizer_path(self, model_name: str = None) -> str:
        """Get tokenizer path (same as model path)"""
        return self.get_model_path(model_name)
    
    # Legacy properties for backward compatibility
    @property
    def model_path(self) -> str:
        return self.get_model_path()
    
    @property
    def engine_dir(self) -> str:
        return self.get_engine_dir()
    
    @property
    def tokenizer_path(self) -> str:
        return self.get_tokenizer_path()
    
    @property
    def max_batch_size(self) -> int:
        return self._config['model'].get('max_batch_size', 8)
    
    @property
    def max_input_length(self) -> int:
        return self._config['model'].get('max_input_length', 2048)
    
    @property
    def max_output_length(self) -> int:
        return self._config['model'].get('max_output_length', 1024)
    
    # Inference configuration
    @property
    def default_temperature(self) -> float:
        return self._config['inference'].get('default_temperature', 0.7)
    
    @property
    def default_top_p(self) -> float:
        return self._config['inference'].get('default_top_p', 0.9)
    
    @property
    def default_top_k(self) -> int:
        return self._config['inference'].get('default_top_k', 50)
    
    @property
    def default_max_tokens(self) -> int:
        return self._config['inference'].get('default_max_tokens', 512)
    
    # Engine configuration
    @property
    def engine_type(self) -> str:
        return self._config.get('engine', {}).get('type', 'auto')
    
    @property
    def chat_template(self) -> str:
        return self._config.get('engine', {}).get('chat_template', 'default')
    
    # Hardware configuration
    @property
    def gpu_memory_fraction(self) -> float:
        return self._config.get('hardware', {}).get('gpu_memory_fraction', 0.9)
    
    @property
    def world_size(self) -> int:
        return self._config.get('hardware', {}).get('world_size', 1)
    
    @property
    def rank(self) -> int:
        return self._config.get('hardware', {}).get('rank', 0)
    
    # API configuration
    @property
    def test_question(self) -> str:
        return self._config.get('api', {}).get('test_question', 'Tell me about yourself')
    
    # Logging configuration
    @property
    def logging_level(self) -> str:
        return self._config.get('logging', {}).get('level', 'INFO')
    
    @property
    def logging_format(self) -> str:
        return self._config.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    @property
    def logging_file(self) -> str:
        return self._config.get('logging', {}).get('file', None)