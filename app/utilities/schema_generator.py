# app/utils/schema_generator.py
import json
import re

def generate_initial_schema(fields, schema_name):
    """
    Generate an initial schema structure from identified fields
    
    Args:
        fields: List of identified fields with name, type, description
        schema_name: Name of the schema
        
    Returns:
        dict: Generated schema object
    """
    schema = {
        "name": schema_name,
        "description": f"Schema for {schema_name} data extraction",
        "fields": []
    }
    
    # Process fields and organize them
    for field in fields:
        field_name = field.get('name', '').strip()
        if not field_name:
            continue
            
        # Convert field name to snake_case if not already
        field_name = to_snake_case(field_name)
        
        # Determine field type
        field_type = field.get('type', 'string').lower()
        
        # Create field object
        schema_field = {
            "name": field_name,
            "type": field_type,
            "description": field.get('description', ''),
            "required": field.get('required', False)
        }
        
        # Add format specification for certain types
        if field_type == 'string' and field.get('format'):
            schema_field['format'] = field['format']
        
        # Handle nested fields if type is object
        if field_type == 'object' and 'fields' in field:
            schema_field['fields'] = field['fields']
        
        schema['fields'].append(schema_field)
    
    return schema

def convert_to_sql(schema, dialect='postgresql', include_indexes=True, include_timestamps=True):
    """
    Convert schema to SQL DDL statements
    
    Args:
        schema: Schema object
        dialect: SQL dialect (postgresql, mysql, etc.)
        include_indexes: Whether to include index creation statements
        include_timestamps: Whether to include timestamp columns
        
    Returns:
        str: SQL DDL statements
    """
    table_name = schema.get('name', 'extracted_data')
    fields = schema.get('fields', [])
    
    # Start building SQL
    sql = f"CREATE TABLE {table_name} (\n"
    sql += "  id SERIAL PRIMARY KEY,\n"
    
    # Add fields
    for field in fields:
        sql += f"  {field_to_sql(field, dialect, nested_path='')},\n"
    
    # Add timestamps if requested
    if include_timestamps:
        sql += "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n"
        sql += "  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
    else:
        # Remove trailing comma from last field
        sql = sql.rstrip(",\n") + "\n"
    
    sql += ");\n\n"
    
    # Add indexes if requested
    if include_indexes:
        sql += generate_indexes(schema, dialect)
    
    return sql

def field_to_sql(field, dialect, nested_path=''):
    """
    Convert a schema field to SQL column definition
    
    Args:
        field: Field object
        dialect: SQL dialect
        nested_path: Path to the field if nested
        
    Returns:
        str: SQL column definition
    """
    field_name = field['name']
    field_type = field['type'].lower()
    required = field.get('required', False)
    
    # Handle nested path
    if nested_path:
        field_name = f"{nested_path}_{field_name}"
    
    # Map schema types to SQL types
    sql_type = map_type_to_sql(field_type, field.get('format'), dialect)
    
    # Build column definition
    column_def = f"{field_name} {sql_type}"
    
    # Add NOT NULL constraint if required
    if required:
        column_def += " NOT NULL"
    
    # Handle nested objects
    if field_type == 'object' and 'fields' in field:
        # For SQL, we flatten nested objects
        nested_columns = []
        for nested_field in field['fields']:
            nested_columns.append(field_to_sql(nested_field, dialect, nested_path=field_name))
        
        return ",\n  ".join(nested_columns)
    
    return column_def

def map_type_to_sql(field_type, format_spec=None, dialect='postgresql'):
    """
    Map schema type to SQL type
    
    Args:
        field_type: Field type
        format_spec: Format specification
        dialect: SQL dialect
        
    Returns:
        str: SQL type
    """
    if dialect == 'postgresql':
        # PostgreSQL type mapping
        if field_type == 'string':
            if format_spec == 'date':
                return 'DATE'
            elif format_spec == 'date-time':
                return 'TIMESTAMP'
            elif format_spec == 'email':
                return 'VARCHAR(255)'
            elif format_spec == 'uri':
                return 'VARCHAR(2048)'
            elif format_spec == 'uuid':
                return 'UUID'
            else:
                return 'TEXT'
        elif field_type == 'number':
            if format_spec == 'float' or format_spec == 'double':
                return 'DOUBLE PRECISION'
            elif format_spec == 'decimal':
                return 'DECIMAL(15,2)'
            else:
                return 'NUMERIC'
        elif field_type == 'integer':
            return 'INTEGER'
        elif field_type == 'boolean':
            return 'BOOLEAN'
        elif field_type == 'array':
            return 'JSONB'
        elif field_type == 'object':
            return 'JSONB'
        else:
            return 'TEXT'
    
    elif dialect == 'mysql':
        # MySQL type mapping
        if field_type == 'string':
            if format_spec == 'date':
                return 'DATE'
            elif format_spec == 'date-time':
                return 'DATETIME'
            elif format_spec == 'email':
                return 'VARCHAR(255)'
            elif format_spec == 'uri':
                return 'VARCHAR(2048)'
            elif format_spec == 'uuid':
                return 'CHAR(36)'
            else:
                return 'TEXT'
        elif field_type == 'number':
            if format_spec == 'float' or format_spec == 'double':
                return 'DOUBLE'
            elif format_spec == 'decimal':
                return 'DECIMAL(15,2)'
            else:
                return 'DECIMAL(15,2)'
        elif field_type == 'integer':
            return 'INT'
        elif field_type == 'boolean':
            return 'TINYINT(1)'
        elif field_type == 'array':
            return 'JSON'
        elif field_type == 'object':
            return 'JSON'
        else:
            return 'TEXT'
    
    # Default to generic SQL
    if field_type == 'string':
        return 'VARCHAR(255)'
    elif field_type == 'number':
        return 'NUMERIC'
    elif field_type == 'integer':
        return 'INTEGER'
    elif field_type == 'boolean':
        return 'BOOLEAN'
    else:
        return 'TEXT'

def generate_indexes(schema, dialect='postgresql'):
    """
    Generate SQL index creation statements
    
    Args:
        schema: Schema object
        dialect: SQL dialect
        
    Returns:
        str: SQL index creation statements
    """
    table_name = schema.get('name', 'extracted_data')
    fields = schema.get('fields', [])
    
    sql = ""
    
    # Basic strategy: index fields that are likely to be queried
    # This includes IDs, dates, and fields marked as searchable
    for field in fields:
        field_name = field['name']
        field_type = field['type'].lower()
        
        # Add index for potential foreign keys (fields ending with _id)
        if field_name.endswith('_id') or field_name == 'id':
            sql += f"CREATE INDEX idx_{table_name}_{field_name} ON {table_name}({field_name});\n"
        
        # Add index for date fields
        elif field_type == 'string' and field.get('format') in ['date', 'date-time']:
            sql += f"CREATE INDEX idx_{table_name}_{field_name} ON {table_name}({field_name});\n"
        
        # Add index for searchable fields
        elif field.get('searchable', False):
            if dialect == 'postgresql' and field_type == 'string':
                # Use GIN index for text search in PostgreSQL
                sql += f"CREATE INDEX idx_{table_name}_{field_name}_gin ON {table_name} USING GIN (to_tsvector('english', {field_name}));\n"
            else:
                sql += f"CREATE INDEX idx_{table_name}_{field_name} ON {table_name}({field_name});\n"
    
    return sql

def convert_to_json_schema(schema, draft='draft-07', include_examples=True):
    """
    Convert schema to JSON Schema format
    
    Args:
        schema: Schema object
        draft: JSON Schema draft version
        include_examples: Whether to include example values
        
    Returns:
        str: JSON Schema as a string
    """
    json_schema = {
        "$schema": f"http://json-schema.org/draft-07/schema#",
        "title": schema.get('name', 'Document Data Schema'),
        "type": "object",
        "description": schema.get('description', ''),
        "required": [],
        "properties": {}
    }
    
    # Process fields
    for field in schema.get('fields', []):
        field_to_json_schema(field, json_schema)
    
    # Convert to formatted JSON string
    return json.dumps(json_schema, indent=2)

def field_to_json_schema(field, json_schema):
    """
    Convert field to JSON Schema property
    
    Args:
        field: Field object
        json_schema: JSON Schema object being built
        
    Returns:
        None (modifies json_schema in place)
    """
    field_name = field['name']
    field_type = field['type'].lower()
    
    # Add to required fields if needed
    if field.get('required', False):
        json_schema['required'].append(field_name)
    
    # Create property definition
    property_def = {
        "type": map_type_to_json_schema(field_type),
        "description": field.get('description', '')
    }
    
    # Add format if specified
    if field.get('format'):
        property_def["format"] = field['format']
    
    # Handle nested objects
    if field_type == 'object' and 'fields' in field:
        property_def["properties"] = {}
        property_def["required"] = []
        
        for nested_field in field['fields']:
            # Recursively handle nested fields
            nested_name = nested_field['name']
            
            # Add to required fields if needed
            if nested_field.get('required', False):
                property_def['required'].append(nested_name)
            
            # Create property for nested field
            property_def["properties"][nested_name] = {
                "type": map_type_to_json_schema(nested_field['type']),
                "description": nested_field.get('description', '')
            }
            
            # Add format if specified
            if nested_field.get('format'):
                property_def["properties"][nested_name]["format"] = nested_field['format']
    
    # Handle arrays
    elif field_type == 'array' and 'items' in field:
        items_type = field['items'].get('type', 'string')
        
        property_def["items"] = {
            "type": map_type_to_json_schema(items_type)
        }
        
        # Handle array of objects
        if items_type == 'object' and 'fields' in field['items']:
            property_def["items"]["properties"] = {}
            property_def["items"]["required"] = []
            
            for item_field in field['items']['fields']:
                item_name = item_field['name']
                
                # Add to required fields if needed
                if item_field.get('required', False):
                    property_def["items"]["required"].append(item_name)
                
                # Create property for item field
                property_def["items"]["properties"][item_name] = {
                    "type": map_type_to_json_schema(item_field['type']),
                    "description": item_field.get('description', '')
                }
    
    # Add property to schema
    json_schema['properties'][field_name] = property_def

def map_type_to_json_schema(field_type):
    """
    Map schema type to JSON Schema type
    
    Args:
        field_type: Field type
        
    Returns:
        str: JSON Schema type
    """
    if field_type == 'string':
        return 'string'
    elif field_type in ['number', 'float', 'double', 'decimal']:
        return 'number'
    elif field_type in ['integer', 'int']:
        return 'integer'
    elif field_type == 'boolean':
        return 'boolean'
    elif field_type == 'array':
        return 'array'
    elif field_type == 'object':
        return 'object'
    elif field_type == 'null':
        return 'null'
    else:
        return 'string'  # Default to string for unknown types

def to_snake_case(name):
    """
    Convert string to snake_case
    
    Args:
        name: String to convert
        
    Returns:
        str: snake_case string
    """
    # Replace non-alphanumeric characters with spaces
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    
    # Replace spaces and other characters with underscores
    s3 = re.sub(r'[^a-zA-Z0-9]', '_', s2)
    
    # Convert to lowercase and remove duplicate underscores
    s4 = re.sub(r'_+', '_', s3).lower().strip('_')
    
    return s4

def identify_primary_keys(fields):
    """
    Identify potential primary key fields
    
    Args:
        fields: List of fields
        
    Returns:
        list: Potential primary key fields
    """
    potential_pks = []
    
    for field in fields:
        field_name = field.get('name', '').lower()
        field_type = field.get('type', '').lower()
        
        # Check for ID-like field names
        if field_name == 'id' or field_name.endswith('_id'):
            potential_pks.append(field)
        
        # Check for UUID fields
        if field_type == 'string' and field.get('format') == 'uuid':
            potential_pks.append(field)
    
    return potential_pks

def identify_relationships(schema):
    """
    Identify potential relationships between entities
    
    Args:
        schema: Schema object
        
    Returns:
        list: Potential relationships
    """
    relationships = []
    fields = schema.get('fields', [])
    
    for field in fields:
        field_name = field.get('name', '').lower()
        
        # Check for foreign key fields (ending with _id)
        if field_name.endswith('_id') and field_name != 'id':
            # Extract the referenced entity name
            entity_name = field_name[:-3]  # Remove _id suffix
            
            relationships.append({
                "from": schema.get('name', 'document'),
                "to": entity_name,
                "type": "many-to-one",
                "foreign_key": field_name
            })
    
    return relationships
