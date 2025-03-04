import json
import os
from typing import Any, Dict, List, Optional, Union

class JsonAccessor:
    def __init__(self, data: dict = None, file_path: str = None):
        """
        Initialize JsonAccessor with either direct data or a file path.
        
        Args:
            data (dict, optional): Direct JSON data
            file_path (str, optional): Path to JSON file
        """
        self.data = data
        if file_path:
            self.load_file(file_path)
            
    def load_file(self, file_path: str) -> None:
        """
        Load JSON data from file.
        
        Args:
            file_path (str): Path to JSON file
        
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        try:
            with open(file_path, 'r') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            self.data = {}
            
    def get_value(self, path: str, default: Any = None) -> Any:
        """
        Get nested value using dot notation path.
        
        Args:
            path (str): Path to value using dot notation (e.g., 'data.0.name')
            default (Any): Default value if path not found
            
        Returns:
            Any: Value at path or default if not found
        """
        try:
            current = self.data
            for key in path.split('.'):
                # Handle array indices
                if '[' in key and ']' in key:
                    array_key, index = key.split('[')
                    index = int(index.replace(']', ''))
                    current = current[array_key][index]
                else:
                    current = current[key]
            return current
        except Exception:
            return default
            
    def find_key_recursive(self, target_key: str, obj: Any = None, path: str = "") -> List[Dict]:
        """
        Recursively find all instances of a key in the JSON structure.
        
        Args:
            target_key (str): Key to search for
            obj (Any, optional): Current object being searched
            path (str, optional): Current path in the JSON structure
            
        Returns:
            List[Dict]: List of matches with path, value, and type information
        """
        if obj is None:
            obj = self.data
        
        results = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check if this is our target key
                if key == target_key:
                    results.append({
                        'path': current_path,
                        'value': value,
                        'type': type(value).__name__
                    })
                
                # Recurse into this value
                results.extend(self.find_key_recursive(target_key, value, current_path))
                
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                current_path = f"{path}[{i}]"
                results.extend(self.find_key_recursive(target_key, value, current_path))
        
        return results

    def get_schema(self, obj: Any = None, path: str = "") -> Union[Dict, List, str]:
        """
        Return schema of JSON structure.
        
        Args:
            obj (Any, optional): Current object to get schema for
            path (str, optional): Current path in the JSON structure
            
        Returns:
            Union[Dict, List, str]: Schema representation of the JSON structure
        """
        if obj is None:
            obj = self.data
            
        if isinstance(obj, dict):
            return {
                key: self.get_schema(value, f"{path}.{key}" if path else key)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            if obj:
                return [self.get_schema(obj[0], f"{path}[0]")]
            return []
        else:
            return type(obj).__name__

    def validate_path(self, path: str) -> Dict:
        """
        Check if a path exists and return information about it.
        
        Args:
            path (str): Path to validate
            
        Returns:
            Dict: Validation information including existence, type, and value
        """
        value = self.get_value(path)
        exists = value is not None
        data_type = type(value).__name__ if exists else None
        return {
            'exists': exists,
            'type': data_type,
            'value': value,
            'path': path
        }

    def search_values(self, search_term: str, case_sensitive: bool = False) -> List[Dict]:
        """
        Search for values in the JSON structure.
        
        Args:
            search_term (str): Term to search for
            case_sensitive (bool, optional): Whether to do case-sensitive search
            
        Returns:
            List[Dict]: List of matches with path and value information
        """
        results = []
        
        def search_recursive(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    search_recursive(value, new_path)
            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    search_recursive(value, f"{path}[{i}]")
            else:
                str_value = str(obj)
                if not case_sensitive:
                    str_value = str_value.lower()
                    search_term_lower = search_term.lower()
                    if search_term_lower in str_value:
                        results.append({
                            'path': path,
                            'value': obj,
                            'type': type(obj).__name__
                        })
                else:
                    if search_term in str_value:
                        results.append({
                            'path': path,
                            'value': obj,
                            'type': type(obj).__name__
                        })
        
        search_recursive(self.data)
        return results

    def keys_at_level(self, level: int = 0) -> List[str]:
        """
        Get all keys at a specific nesting level.
        
        Args:
            level (int): Nesting level to get keys from
            
        Returns:
            List[str]: List of keys at specified level
        """
        def get_keys_recursive(obj: Any, current_level: int = 0) -> List[str]:
            if current_level == level:
                if isinstance(obj, dict):
                    return list(obj.keys())
                elif isinstance(obj, list):
                    return [str(i) for i in range(len(obj))]
                return []
                
            keys = []
            if isinstance(obj, dict):
                for value in obj.values():
                    keys.extend(get_keys_recursive(value, current_level + 1))
            elif isinstance(obj, list):
                for item in obj:
                    keys.extend(get_keys_recursive(item, current_level + 1))
            return keys
            
        return get_keys_recursive(self.data)
