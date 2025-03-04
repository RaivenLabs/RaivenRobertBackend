# app/schema_blueprint.py
from flask import Blueprint, request, jsonify, current_app
import json
import yaml
import os
import boto3
import re
import copy
import psycopg2
from psycopg2.extras import DictCursor
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from ..utilities.document_processor import extract_text_from_file, parse_csv, parse_excel
from ..utilities.schema_generator import generate_initial_schema, convert_to_sql, convert_to_json_schema

# Use the existing Anthropic client from main_blueprint
from .main_routes import anthropic_client

load_dotenv()

# Store PostgreSQL connection parameters at module level
PG_CONFIG = {
    'user': os.getenv('PG_USER'),
    'password': os.getenv('PG_PASSWORD'),
    'host': os.getenv('PG_HOST'),
    'port': os.getenv('PG_PORT'),
    'database': os.getenv('PG_DATABASE')
}


# Log connection parameters (without exposing password)
print("==== PostgreSQL Connection Parameters ====")
print(f"PG_USER: {PG_CONFIG['user']}")
print(f"PG_PASSWORD: {'set' if PG_CONFIG['password'] else 'not set'}")
print(f"PG_HOST: {PG_CONFIG['host']}")
print(f"PG_PORT: {PG_CONFIG['port']}")
print(f"PG_DATABASE: {PG_CONFIG['database']}")

# Helper function to get database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            user=PG_CONFIG['user'],
            password=PG_CONFIG['password'],
            host=PG_CONFIG['host'],
            port=PG_CONFIG['port'],
            database=PG_CONFIG['database']
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


# Create a Blueprint for schema generation routes
schema_blueprint = Blueprint('schema', __name__, url_prefix='/api')

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'csv', 'xlsx', 'json'}
print(f"PG_USER: {os.getenv('PG_USER', 'not set')}")
print(f"PG_PASSWORD: {'set' if os.getenv('PG_PASSWORD') else 'not set'}")
print(f"PG_HOST: {os.getenv('PG_HOST', 'not set')}")
print(f"PG_PORT: {os.getenv('PG_PORT', 'not set')}")
print(f"PG_DATABASE: {os.getenv('PG_DATABASE', 'not set')}")


def allowed_file(filename):
    """Check if a filename has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_schema(project_id, schema_data):
    """
    Save a schema to persistent storage (S3 or local file system)
    
    Args:
        project_id: ID of the project
        schema_data: The schema data to save
    
    Returns:
        bool: Success status
    """
    try:
        # Debug prints
        print(f"[save_schema] Saving schema for project: {project_id}")
        flask_env = current_app.config.get('FLASK_ENV') or os.environ.get('FLASK_ENV')
        use_local_files = os.environ.get('USE_LOCAL_FILES', 'False')
        
        print(f"[save_schema] FLASK_ENV: {flask_env}")
        print(f"[save_schema] USE_LOCAL_FILES: {use_local_files}")
        
        is_development = flask_env == 'development'
        should_use_local = use_local_files.lower() == 'true'
        
        # For now, always save locally
        # Navigate up from backend/app to transaction_platform_app, then to static/data
        schema_dir = os.path.abspath(os.path.join(
            current_app.root_path,  # This is backend/app
            '..',                   # Up to backend
            '..',                   # Up to transaction_platform_app
            'static', 
            'data',
            'schemas', 
            project_id
        ))
        os.makedirs(schema_dir, exist_ok=True)
        print(f"[save_schema] Created directory: {schema_dir}")
        
        schema_path = os.path.join(schema_dir, 'schema.json')
        with open(schema_path, 'w') as f:
            json.dump(schema_data, f, indent=2)
        print(f"[save_schema] Schema saved locally to: {schema_path}")
        return True

    except Exception as e:
        current_app.logger.error(f"Error saving schema: {str(e)}")
        print(f"[save_schema] ERROR: {str(e)}")
        return False

def get_db_connection():
    """Create and return a database connection."""
    print(f"PG_USER: {os.getenv('PG_USER', 'not set')}")
    print(f"PG_PASSWORD: {'set' if os.getenv('PG_PASSWORD') else 'not set'}")
    print(f"PG_HOST: {os.getenv('PG_HOST', 'not set')}")
    print(f"PG_PORT: {os.getenv('PG_PORT', 'not set')}")
    print(f"PG_DATABASE: {os.getenv('PG_DATABASE', 'not set')}")
    
    
    
    conn = psycopg2.connect(
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
      
        host=os.getenv("PG_HOST", "localhost"), 
        port=os.getenv("PG_PORT", "5432"),
        dbname=os.getenv("PG_DATABASE", "hawkeye_db")
    )
  
    return conn

def get_next_dimension_order(cursor):
    """Get the next available order number for dimensions."""
    cursor.execute("SELECT MAX(order_num) FROM acquisitiondimensions")
    max_order = cursor.fetchone()[0] or 0
    return max_order + 1

def slugify(text):
    """Convert a display name to a dimension_id or attribute_id"""
    # Convert to lowercase
    text = text.lower()
    # Replace spaces and other special characters with underscores
    text = re.sub(r'[^a-z0-9_]', '_', text)
    # Remove consecutive underscores
    text = re.sub(r'_+', '_', text)
    # Remove leading/trailing underscores
    text = text.strip('_')
    return text

def extract_json_from_response(response):
    """Extract and parse JSON from Claude's response text with enhanced debugging and repair"""
    print("\n--- Extracting JSON from Claude's response ---")
    
    # If response is None or empty
    if not response:
        print("[extract_json] Response is empty")
        return []
    
    # If response is already a dict or list
    if isinstance(response, (dict, list)):
        print(f"[extract_json] Response is already a {type(response).__name__}, returning as is")
        return response
    
    # Ensure we're working with a string
    if not isinstance(response, str):
        print(f"[extract_json] Converting response from {type(response).__name__} to string")
        response = str(response)
    
    # Print a preview of the response
    print(f"[extract_json] Response preview (first 200 chars): {response[:200]}...")
    
    # Try to extract JSON from the response
    try:
        # Look for JSON code block
        import re
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, response)
        
        if match:
            json_str = match.group(1).strip()
            print(f"[extract_json] Found JSON in code block, first 100 chars: {json_str[:100]}...")
            try:
                parsed_json = json.loads(json_str)
                print(f"[extract_json] Successfully parsed JSON from code block: {type(parsed_json).__name__}")
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"[extract_json] Error parsing JSON in code block: {str(e)}")
                # Continue to other methods
        
        # Remove markdown code blocks if present (alternative method)
        if response.startswith("```") and "```" in response[3:]:
            # Find the end of the opening code block marker
            start_pos = response.find("\n", 3) + 1
            # Find the start of the closing code block marker
            end_pos = response.rfind("```")
            
            # Extract what's inside the code blocks
            clean_response = response[start_pos:end_pos].strip()
            # If it starts with "json\n" remove that too
            if clean_response.startswith("json\n"):
                clean_response = clean_response[5:]
            # If any other format identifier is there
            elif "\n" in clean_response and clean_response.find("\n") < 10:
                clean_response = clean_response[clean_response.find("\n")+1:]
                
            try:
                parsed_json = json.loads(clean_response)
                print(f"[extract_json] Successfully parsed JSON from cleaned code block: {type(parsed_json).__name__}")
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"[extract_json] Error parsing cleaned JSON: {str(e)}")
                # Continue to other methods
        
        # Find the first [ and the last ] for an array
        array_start_idx = response.find('[')
        array_end_idx = response.rfind(']')
        
        if array_start_idx != -1 and array_end_idx > array_start_idx:
            json_str = response[array_start_idx:array_end_idx+1].strip()
            print(f"[extract_json] Found JSON array in raw text, first 100 chars: {json_str[:100]}...")
            
            # Try to parse the JSON array
            try:
                parsed_json = json.loads(json_str)
                print(f"[extract_json] Successfully parsed JSON array: {len(parsed_json)} items")
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"[extract_json] Error parsing JSON array: {str(e)}")
                print(f"[extract_json] Error location: line {e.lineno}, column {e.colno}, char {e.pos}")
                
                # Let's try to manually extract the main dimensions
                print("[extract_json] Attempting to extract complete dimension objects")
                try:
                    # Find all dimension objects
                    dimension_pattern = r'(\{\s*"name":\s*"[^"]+",\s*"attributeGroups":\s*\[\s*\{.*?\}\s*\]\s*\})'
                    dimensions = re.findall(dimension_pattern, json_str, re.DOTALL)
                    
                    if dimensions:
                        # Try to parse each dimension individually
                        valid_dimensions = []
                        for i, dim in enumerate(dimensions):
                            try:
                                # Add commas between dimensions
                                if i < len(dimensions) - 1:
                                    dim += ","
                                valid_dimensions.append(json.loads(dim))
                                print(f"[extract_json] Successfully parsed dimension {i+1}")
                            except:
                                print(f"[extract_json] Failed to parse dimension {i+1}")
                        
                        if valid_dimensions:
                            print(f"[extract_json] Extracted {len(valid_dimensions)} valid dimensions")
                            return valid_dimensions
                except Exception as inner_e:
                    print(f"[extract_json] Error during manual extraction: {str(inner_e)}")
        
        # If we can't find or parse JSON, try finding a JSON object
        obj_start_idx = response.find('{')
        obj_end_idx = response.rfind('}')
        
        if obj_start_idx != -1 and obj_end_idx > obj_start_idx:
            json_str = response[obj_start_idx:obj_end_idx+1].strip()
            print(f"[extract_json] Found JSON object in raw text, first 100 chars: {json_str[:100]}...")
            
            try:
                parsed_json = json.loads(json_str)
                print(f"[extract_json] Successfully parsed JSON object")
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"[extract_json] Error parsing JSON object: {str(e)}")
                
        # If all previous methods fail, use fallback
        try:
            # Create a fallback response
            fallback_json = """[
              {
                "name": "ESG_Compliance",
                "attributeGroups": [
                  {
                    "name": "Environmental Compliance",
                    "count": 6,
                    "attributes": [
                      {
                        "id": "environmental_permits",
                        "displayName": "Environmental Permits and Licenses",
                        "dataType": "text",
                        "required": true,
                        "maxLength": 2000,
                        "order": 1
                      },
                      {
                        "id": "emissions_data",
                        "displayName": "Emissions Data and Reporting",
                        "dataType": "text",
                        "required": true,
                        "maxLength": 2000,
                        "order": 2
                      }
                    ]
                  }
                ]
              },
              {
                "name": "Cybersecurity_Data_Privacy",
                "attributeGroups": [
                  {
                    "name": "Security Infrastructure",
                    "count": 5,
                    "attributes": [
                      {
                        "id": "security_architecture",
                        "displayName": "Security Architecture Overview",
                        "dataType": "text",
                        "required": true,
                        "maxLength": 2000,
                        "order": 1
                      },
                      {
                        "id": "network_security",
                        "displayName": "Network Security Measures",
                        "dataType": "text",
                        "required": true,
                        "maxLength": 2000,
                        "order": 2
                      }
                    ]
                  }
                ]
              }
            ]"""
            parsed_fallback = json.loads(fallback_json)
            print(f"[extract_json] Using fallback JSON with {len(parsed_fallback)} dimensions")
            return parsed_fallback
        except Exception as fallback_e:
            print(f"[extract_json] Error using fallback: {str(fallback_e)}")
    
        # If we can't find or parse JSON, return an empty list
        print("[extract_json] Could not extract valid JSON, returning empty list")
        return []
        
    except Exception as e:
        print(f"[extract_json] Unexpected error: {str(e)}")
        return []

@schema_blueprint.route('/briefing/<project_id>', methods=['GET'])
def get_briefing(project_id):
    """Get the briefing context for a project including both context.yaml and scope.txt"""
    try:
        print(f"\n====== BRIEFING ENDPOINT CALLED FOR PROJECT {project_id} ======")
        
        # Define paths for both context files
        context_yaml_path = os.path.abspath(os.path.join(
            current_app.root_path,  # This is backend/app
            '..',                   # Up to backend
            '..',                   # Up to transaction_platform_app
            'static',
            'data',
            'background',
            'context.yaml'
        ))
        
        scope_txt_path = os.path.abspath(os.path.join(
            current_app.root_path,  # This is backend/app
            '..',                   # Up to backend
            '..',                   # Up to transaction_platform_app
            'static',
            'data',
            'background',
            'scope.txt'
        ))
        
        print(f"[get_briefing] Context path: {context_yaml_path}")
        print(f"[get_briefing] Scope path: {scope_txt_path}")
        
        # Initialize response object
        briefing_data = {}
        
        # Read context.yaml file
        try:
            with open(context_yaml_path, 'r') as f:
                yaml_context = yaml.safe_load(f)
                briefing_data["context"] = yaml_context
                print(f"[get_briefing] Successfully loaded context.yaml")
        except Exception as e:
            print(f"[get_briefing] Error reading context.yaml: {str(e)}")
            briefing_data["context"] = None
        
        # Read scope.txt file
        try:
            with open(scope_txt_path, 'r') as f:
                scope_text = f.read()
                briefing_data["scope"] = scope_text
                print(f"[get_briefing] Successfully loaded scope.txt ({len(scope_text)} chars)")
        except Exception as e:
            print(f"[get_briefing] Error reading scope.txt: {str(e)}")
            briefing_data["scope"] = None
        
        # Add project ID to the response
        briefing_data["project_id"] = project_id
        
        print(f"[get_briefing] Returning briefing data with keys: {list(briefing_data.keys())}")
        print("====== BRIEFING ENDPOINT COMPLETE ======\n")
        
        return jsonify(briefing_data)
    except Exception as e:
        import traceback
        print(f"\n====== ERROR IN BRIEFING ENDPOINT ======")
        print(f"Exception: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("=========================================\n")
        
        current_app.logger.error(f"Error loading briefing context: {str(e)}")
        return jsonify({"error": f"Error loading briefing context: {str(e)}"}), 500

@schema_blueprint.route('/suggest-schema-enhancements', methods=['POST'])
def suggest_schema_enhancements():
    """
    Analyze documents and briefing context to suggest schema enhancements for star schema
    using SQL database instead of JSON files.
    """
    try:
        print("\n====== SUGGEST SCHEMA ENHANCEMENTS ENDPOINT CALLED ======")
        
        # Parse request data
        context = json.loads(request.form['context'])
        print(context)
        document_type = context.get('documentType', 'unknown')
        project_name = context.get('projectName', 'Unnamed Project')
        customer_id = context.get('customerId')
        briefing_context = context.get('briefingContext', {})
        
        # Process uploaded files if any
        uploaded_content = ""
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '' and allowed_file(file.filename):
                print(f"[suggest] Processing uploaded file: {file.filename}")
                file_ext = file.filename.rsplit('.', 1)[1].lower()
                
                # Save the file temporarily
                temp_dir = os.path.join(current_app.root_path, 'temp')
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, secure_filename(file.filename))
                file.save(temp_path)
                
                # Extract text based on file type
                if file_ext == 'pdf':
                    uploaded_content = extract_text_from_file(temp_path, file_ext)
                elif file_ext == 'csv':
                    uploaded_content = parse_csv(temp_path)
                elif file_ext in ['xlsx', 'xls']:
                    uploaded_content = parse_excel(temp_path)
                else:
                    # Read as text file
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        uploaded_content = f.read()
                
                print(f"[suggest] Extracted {len(uploaded_content)} characters from uploaded file")
                # Clean up the temp file
                os.remove(temp_path)
        
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # STEP 1: Fetch current dimensions and attributes from database
        print("[suggest] STEP 1: Fetching dimensions from database")
        cursor.execute("""
            SELECT id, dimension_id, display_name, description, program_type, order_num 
            FROM acquisitiondimensions 
            WHERE status = 'approved'
            AND (customer_id IS NULL OR customer_id = %s)
            ORDER BY order_num
        """, (customer_id,))
        dimensions = cursor.fetchall()
        
        # Create dimensions data in the same format as before for the prompt
        dimensions_data = {
            "dimensions": [
                {
                    "id": dim["dimension_id"],
                    "displayName": dim["display_name"],
                    "order": dim["order_num"]
                }
                for dim in dimensions
            ]
        }
        
        # STEP 2: Fetch attributes for each dimension
        print("[suggest] STEP 2: Fetching attributes from database")
        attributes_data = {}
        for dim in dimensions:
            cursor.execute("""
                SELECT attribute_id, display_name, data_type, required, max_length, order_num 
                FROM acquisitionattributes 
                WHERE dimension_id = %s
                AND status = 'approved'
                AND (customer_id IS NULL OR customer_id = %s)
                ORDER BY order_num
            """, (dim["id"], customer_id))
            
            attributes = cursor.fetchall()
            
            # Convert to the same format as before
            attributes_data[dim["dimension_id"]] = [
                {
                    "id": attr["attribute_id"],
                    "displayName": attr["display_name"],
                    "dataType": attr["data_type"],
                    "required": attr["required"],
                    "maxLength": attr["max_length"],
                    "order": attr["order_num"]
                }
                for attr in attributes
            ]
        
        # STEP 3: Create the prompt for Claude
        print("[suggest] STEP 3: Creating prompt for Claude")
        
        scope_text = briefing_context.get('scope', '')
        context_data = briefing_context.get('context', {})
        
        # Format dimensions and attributes for the prompt
        dimensions_text = json.dumps(dimensions_data, indent=2)
        attributes_text = json.dumps(attributes_data, indent=2)
        
        prompt = f"""
You are an experienced database architect and M&A due diligence specialist. I need your help to analyze some customer documents and suggest enhancements to our schema for an M&A due diligence platform.

PROJECT CONTEXT:
Project Name: {project_name}
Customer: {"Custom client" if customer_id else "Standard M&A"}

BACKGROUND INFORMATION:
{scope_text}

CURRENT SCHEMA:
We have a PostgreSQL database with dimensions (categories) and attributes (fields) for M&A due diligence. THese tables are intended to help us implement a star schema in which eahc M&A tarsanction is a star, the dimesnio tbale holds over a dozen dimensions for the star (like ip technology, supply chain and so forth and the attributes table holds the attributes fo reach dimesnion.)

Here are our current dimensions:
{dimensions_text}

Here are the attributes for each dimension:
{attributes_text}

CUSTOMER DOCUMENT:
{uploaded_content[:50000]}  # Truncate if needed

TASK:
Based on the customer document and project context, please suggest new dimensions and attributes that would enhance our schema for this specific M&A transaction.

For each new dimension, provide:
1. A clear, professional name
2. 1-3 attribute groups that logically organize the attributes

For each attribute group, provide:
1. A clear name for the group
2. 3-5 specific attributes that are relevant

For each attribute, provide:
1. id: snake_case identifier
2. displayName: Human-readable name
3. dataType: One of [text, boolean, date, number]
4. required: true/false
5. maxLength: For text fields, usually 2000
6. order: Sequence number starting from 1

FORMAT YOUR RESPONSE AS A JSON ARRAY OF OBJECTS:
[
  {{
    "name": "Dimension Name",
    "attributeGroups": [
      {{
        "name": "Group Name",
        "count": 5,
        "attributes": [
          {{
            "id": "attribute_id",
            "displayName": "Attribute Display Name",
            "dataType": "text",
            "required": true,
            "maxLength": 2000,
            "order": 1
          }},
          ...more attributes...
        ]
      }},
      ...more groups...
    ]
  }},
  ...more dimensions...
]

IMPORTANT GUIDELINES:
- Focus on areas NOT covered by the existing dimensions
- Suggest industry-specific or transaction-specific dimensions when relevant
- Only include attributes that are valuable for M&A due diligence
- All suggested dimensions should conform to professional due diligence standards
- Return ONLY valid JSON, no explanations or other text

Your suggestions will be reviewed by the customer and may be added to their schema.
"""
        
        # STEP 4: Call Claude API
        print("[suggest] STEP 4: Calling Claude API")
        response = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # STEP 5: Extract and parse the response
        print("[suggest] STEP 5: Processing Claude's response")
        response_text = response.content[0].text
        suggested_enhancements = extract_json_from_response(response_text)
        
        # STEP 6: Store the suggested dimensions and attributes in the database
        print("[suggest] STEP 6: Storing suggested dimensions and attributes")
        suggested_dimensions = []
        
        for enhancement in suggested_enhancements:
            # Create a dimension_id from the name
            dimension_name = enhancement["name"]
            dimension_id = slugify(dimension_name)
            
            # Check if dimension_id already exists
            cursor.execute("""
                SELECT id FROM acquisitiondimensions WHERE dimension_id = %s AND (customer_id IS NULL OR customer_id = %s)
            """, (dimension_id, customer_id))
            
            existing_dim = cursor.fetchone()
            if existing_dim:
                print(f"[suggest] Dimension {dimension_id} already exists, generating unique ID")
                # Generate a unique ID by adding a suffix
                base_id = dimension_id
                suffix = 1
                while True:
                    new_id = f"{base_id}_{suffix}"
                    cursor.execute("""
                        SELECT id FROM acquisitiondimensions WHERE dimension_id = %s AND (customer_id IS NULL OR customer_id = %s)
                    """, (new_id, customer_id))
                    if not cursor.fetchone():
                        dimension_id = new_id
                        break
                    suffix += 1
            
            # Insert the new dimension
            cursor.execute("""
                INSERT INTO acquisitiondimensions 
                (dimension_id, display_name, description, is_core, program_type, order_num, origin, status, customer_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                dimension_id,
                dimension_name,
                f"Suggested by Claude based on customer materials",
                False,  # is_core
                "M&A",  # program_type
                get_next_dimension_order(cursor),  # Get next available order number
                "claude",  # origin
                "proposed",  # status - important change here!
                customer_id  # customer_id - now we store the customer ID
            ))
            
            new_dimension_id = cursor.fetchone()[0]
            
            # Add to list for response
            suggested_dimensions.append({
                "id": new_dimension_id,
                "name": dimension_name,
                "dimension_id": dimension_id
            })
            
            # Insert attributes for this dimension
            attribute_order = 1
            for group in enhancement.get("attributeGroups", []):
                group_name = group.get("name", "General")
                
                for attr in group.get("attributes", []):
                    # Ensure attribute_id is valid
                    attribute_id = attr.get("id", slugify(attr.get("displayName", f"attr_{attribute_order}")))
                    
                    cursor.execute("""
                        INSERT INTO acquisitionattributes 
                        (dimension_id, attribute_id, display_name, data_type, required, max_length, 
                        order_num, attribute_group, is_core, origin, status, customer_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        new_dimension_id,
                        attribute_id,
                        attr.get("displayName", f"Attribute {attribute_order}"),
                        attr.get("dataType", "text"),
                        attr.get("required", True),
                        attr.get("maxLength", 2000),
                        attr.get("order", attribute_order),
                        group_name,  # Store the attribute group name
                        False,  # is_core
                        "claude",  # origin
                        "proposed",  # status - also important!
                        customer_id  # customer_id - store customer ID here too
                    ))
                    attribute_order += 1
        
        conn.commit()
        
        # STEP 7: Prepare response for frontend
        print("[suggest] STEP 7: Preparing response for frontend")
        response_data = {
            "projectName": project_name,
            "currentDimensions": [
                {
                    "name": dim["display_name"],
                    "id": dim["dimension_id"],
                    "attributeCount": len(attributes_data.get(dim["dimension_id"], []))
                }
                for dim in dimensions
            ],
            "suggestedEnhancements": suggested_enhancements,
            "suggestedDimensions": suggested_dimensions
        }
        
        print("====== SUGGEST SCHEMA ENHANCEMENTS ENDPOINT COMPLETE ======\n")
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print("\n====== ERROR IN SUGGEST SCHEMA ENHANCEMENTS ENDPOINT ======")
        print(f"Exception: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("=========================================\n")
        
        return jsonify({"error": f"Error suggesting schema enhancements: {str(e)}"}), 500
    finally:
        # Close database connections
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@schema_blueprint.route('/apply-schema-enhancements', methods=['POST'])
def apply_schema_enhancements():
    """
    Apply selected schema enhancements by updating the database
    This now updates the status from 'proposed' to 'approved' for selected enhancements
    """
    try:
        print("\n====== APPLY SCHEMA ENHANCEMENTS ENDPOINT CALLED ======")
        
        data = request.json
        if not data or 'selectedEnhancements' not in data or 'projectId' not in data:
            print("[apply] ERROR: Missing selected enhancements or project ID")
            return jsonify({"error": "Missing required data"}), 400
        
        selected_enhancements = data['selectedEnhancements']
        project_id = data['projectId']
        customer_id = data.get('customerId')
        
        print(f"[apply] Project ID: {project_id}")
        print(f"[apply] Customer ID: {customer_id}")
        print(f"[apply] Selected enhancements count: {len(selected_enhancements)}")
        
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Update status of selected dimensions from 'proposed' to 'approved'
        dimensions_approved = 0
        attributes_approved = 0
        
        for enhancement in selected_enhancements:
            dimension_id = enhancement.get('dimension_id') or slugify(enhancement.get('name', ''))
            
            if not dimension_id:
                print(f"[apply] Skipping enhancement without valid dimension_id: {enhancement}")
                continue
                
            print(f"[apply] Processing dimension: {dimension_id}")
            
            # Update the dimension status
            cursor.execute("""
                UPDATE acquisitiondimensions
                SET status = 'approved'
                WHERE dimension_id = %s 
                AND status = 'proposed'
                AND (customer_id IS NULL OR customer_id = %s)
                RETURNING id
            """, (dimension_id, customer_id))
            
            result = cursor.fetchone()
            if result:
                dimension_db_id = result[0]
                dimensions_approved += 1
                
                # Update all attributes for this dimension
                cursor.execute("""
                    UPDATE acquisitionattributes
                    SET status = 'approved'
                    WHERE dimension_id = %s
                    AND status = 'proposed'
                    AND (customer_id IS NULL OR customer_id = %s)
                """, (dimension_db_id, customer_id))
                
                attr_result = cursor.rowcount
                if attr_result:
                    attributes_approved += attr_result
                    print(f"[apply] Approved {attr_result} attributes for dimension {dimension_id}")
                else:
                    print(f"[apply] No attributes found to approve for dimension {dimension_id}")
            else:
                print(f"[apply] No dimension found with ID {dimension_id} and status 'proposed'")
        
        conn.commit()
        
        # Prepare response
        response_data = {
            "success": True,
            "projectId": project_id,
            "dimensionsApproved": dimensions_approved,
            "attributesApproved": attributes_approved,
            "message": f"Successfully approved {dimensions_approved} dimensions and {attributes_approved} attributes."
        }
        
        print(f"[apply] Summary: Approved {dimensions_approved} dimensions and {attributes_approved} attributes")
        print("====== APPLY SCHEMA ENHANCEMENTS ENDPOINT COMPLETE ======\n")
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print("\n====== ERROR IN APPLY SCHEMA ENHANCEMENTS ENDPOINT ======")
        print(f"Exception: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("================================================\n")
        
        return jsonify({"error": f"Error applying schema enhancements: {str(e)}"}), 500
    finally:
        # Close database connections
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@schema_blueprint.route('/get-dimensions', methods=['GET'])
def get_dimensions():
    """
    Get all dimensions from the database, including their status
    Returns both core and customer-specific dimensions
    """
    try:
        print("\n====== GET DIMENSIONS ENDPOINT CALLED ======")
        
        customer_id = request.args.get('customerId')
        include_proposed = request.args.get('includeProposed', 'false').lower() == 'true'
        
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Build the query with proper filtering
        status_filter = ""
        if not include_proposed:
            status_filter = "AND status = 'approved'"
            
        # Get all dimensions
        cursor.execute(f"""
            SELECT 
                id, 
                dimension_id, 
                display_name, 
                description, 
                is_core,
                program_type, 
                order_num, 
                origin,
                status,
                customer_id
            FROM acquisitiondimensions
            WHERE (customer_id IS NULL OR customer_id = %s)
            {status_filter}
            ORDER BY order_num
        """, (customer_id,))
        
        dimensions = cursor.fetchall()
        
        # Count attributes for each dimension
        dimension_data = []
        for dim in dimensions:
            cursor.execute("""
                SELECT COUNT(*) 
                FROM acquisitionattributes 
                WHERE dimension_id = %s
                AND (customer_id IS NULL OR customer_id = %s)
            """, (dim['id'], customer_id))
            
            attribute_count = cursor.fetchone()[0]
            
            dimension_data.append({
                "id": dim['id'],
                "dimension_id": dim['dimension_id'],
                "display_name": dim['display_name'],
                "description": dim['description'],
                "is_core": dim['is_core'],
                "program_type": dim['program_type'],
                "order_num": dim['order_num'],
                "attribute_count": attribute_count,
                "origin": dim['origin'],
                "status": dim['status'],
                "customer_id": dim['customer_id']
            })
        
        print(f"[get_dimensions] Returning {len(dimension_data)} dimensions")
        print("====== GET DIMENSIONS ENDPOINT COMPLETE ======\n")
        
        return jsonify(dimension_data)
        
    except Exception as e:
        import traceback
        print("\n====== ERROR IN GET DIMENSIONS ENDPOINT ======")
        print(f"Exception: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("=========================================\n")
        
        return jsonify({"error": f"Error getting dimensions: {str(e)}"}), 500
    finally:
        # Close database connections
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@schema_blueprint.route('/get-attributes/<dimension_id>', methods=['GET'])
def get_attributes(dimension_id):
    """
    Get all attributes for a specific dimension
    Returns both core and customer-specific attributes
    """
    try:
        print(f"\n====== GET ATTRIBUTES ENDPOINT CALLED FOR DIMENSION {dimension_id} ======")
        
        customer_id = request.args.get('customerId')
        include_proposed = request.args.get('includeProposed', 'false').lower() == 'true'
        
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # First get the dimension's database ID
        cursor.execute("""
            SELECT id FROM acquisitiondimensions 
            WHERE dimension_id = %s
            AND (customer_id IS NULL OR customer_id = %s)
        """, (dimension_id, customer_id))
        
        dim_result = cursor.fetchone()
        if not dim_result:
            return jsonify({"error": f"Dimension '{dimension_id}' not found"}), 404
            
        dim_db_id = dim_result['id']
        
        # Build the query with proper filtering
        status_filter = ""
        if not include_proposed:
            status_filter = "AND status = 'approved'"
            
        # Get all attributes for this dimension
        cursor.execute(f"""
            SELECT 
                id,
                attribute_id,
                display_name,
                data_type,
                required,
                max_length,
                order_num,
                attribute_group,
                is_core,
                origin,
                status,
                customer_id
            FROM acquisitionattributes
            WHERE dimension_id = %s
            AND (customer_id IS NULL OR customer_id = %s)
            {status_filter}
            ORDER BY order_num
        """, (dim_db_id, customer_id))
        
        attributes = cursor.fetchall()
        
        # Format attributes for the response
        attribute_data = []
        for attr in attributes:
            attribute_data.append({
                "id": attr['id'],
                "attribute_id": attr['attribute_id'],
                "display_name": attr['display_name'],
                "data_type": attr['data_type'],
                "required": attr['required'],
                "max_length": attr['max_length'],
                "order_num": attr['order_num'],
                "attribute_group": attr['attribute_group'],
                "is_core": attr['is_core'],
                "origin": attr['origin'],
                "status": attr['status'],
                "customer_id": attr['customer_id']
            })
        
        print(f"[get_attributes] Returning {len(attribute_data)} attributes for dimension {dimension_id}")
        print("====== GET ATTRIBUTES ENDPOINT COMPLETE ======\n")
        
        return jsonify(attribute_data)
        
    except Exception as e:
        import traceback
        print(f"\n====== ERROR IN GET ATTRIBUTES ENDPOINT ======")
        print(f"Exception: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("=========================================\n")
        
        return jsonify({"error": f"Error getting attributes: {str(e)}"}), 500
    finally:
        # Close database connections
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
