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

def get_prompt_template():
    """Load the prompt template from file."""
    prompt_path = os.path.abspath(os.path.join(
        current_app.root_path,  # This is backend/app
        '..',                   # Up to backend
        '..',                   # Up to transaction_platform_app
        'static', 
        'data',
        'prompts', 
        'vectorpromptpackage.txt'
    ))
    
    try:
        with open(prompt_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading prompt template: {str(e)}")
        # Return a fallback simple prompt if file loading fails
        return """
You are an experienced database architect and M&A due diligence specialist. I need your help to analyze customer documents and suggest enhancements to our schema for an M&A due diligence platform.
PROJECT CONTEXT:
Project Name: {project_name}
Customer: {customer_type}
BACKGROUND INFORMATION:
{scope_text}
CURRENT SCHEMA:
We have a PostgreSQL database with dimensions (categories) and attributes (fields) for M&A due diligence.
Here are our current dimensions:
{dimensions_text}
Here are the attributes for each dimension:
{attributes_text}
CUSTOMER DOCUMENT:
{uploaded_content}
TASK:
Based on the customer document and project context, please suggest EXACTLY 3 NEW DIMENSIONS with 3-5 attributes each that would enhance our schema for this specific M&A transaction.
IMPORTANT CONSTRAINTS:
- Suggest ONLY 3 dimensions total
- Each dimension should have 3-5 attributes (no more, no less)
- Focus on areas NOT covered by existing dimensions
- All suggestions should be valuable for M&A due diligence
FORMAT YOUR RESPONSE USING THIS EXACT STRUCTURE (with no extra text before or after):
DIMENSION 1: [Dimension Name]
GROUP: [Group Name]
- ATTRIBUTE 1: [Attribute Name]|text|true|2000
- ATTRIBUTE 2: [Attribute Name]|text|true|2000
- ATTRIBUTE 3: [Attribute Name]|text|true|2000
- ATTRIBUTE 4: [Attribute Name]|text|true|2000
- ATTRIBUTE 5: [Attribute Name]|text|true|2000
DIMENSION 2: [Dimension Name]
GROUP: [Group Name]
- ATTRIBUTE 1: [Attribute Name]|text|true|2000
- ATTRIBUTE 2: [Attribute Name]|text|true|2000
- ATTRIBUTE 3: [Attribute Name]|text|true|2000
- ATTRIBUTE 4: [Attribute Name]|text|true|2000
- ATTRIBUTE 5: [Attribute Name]|text|true|2000
DIMENSION 3: [Dimension Name]
GROUP: [Group Name]
- ATTRIBUTE 1: [Attribute Name]|text|true|2000
- ATTRIBUTE 2: [Attribute Name]|text|true|2000
- ATTRIBUTE 3: [Attribute Name]|text|true|2000
- ATTRIBUTE 4: [Attribute Name]|text|true|2000
- ATTRIBUTE 5: [Attribute Name]|text|true|2000
The attribute format is: [Name]|[data_type]|[required]|[max_length]
Where:
- [data_type] is one of: text, boolean, date, number
- [required] is either true or false
- [max_length] is a number (2000 for text fields)
DO NOT include any explanation or additional text. Start with "DIMENSION 1:" and end with the last attribute.
"""

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

def parse_claude_structured_response(response_text):
    """
    Parse Claude's structured response format directly without using JSON.
    
    Expected format:
    DIMENSION 1: [Dimension Name]
    GROUP: [Group Name]
    - ATTRIBUTE 1: [Attribute Name]|[data_type]|[required]|[max_length]
    ...
    """
    print("[parse_response] Parsing Claude's structured response")
    results = []
    
    try:
        # Split by "DIMENSION" to get each dimension section
        dimension_pattern = r'DIMENSION \d+:\s*(.*?)(?=DIMENSION \d+:|$)'
        dimension_matches = re.findall(dimension_pattern, response_text, re.DOTALL)
        
        if not dimension_matches:
            print("[parse_response] No dimensions found using regex pattern, trying alternative parsing")
            # Fallback: try splitting by lines
            lines = response_text.strip().split('\n')
            dimension_sections = []
            current_section = ""
            
            for line in lines:
                if line.strip().startswith("DIMENSION"):
                    if current_section:
                        dimension_sections.append(current_section)
                    current_section = line + "\n"
                else:
                    current_section += line + "\n"
            
            if current_section:
                dimension_sections.append(current_section)
                
            # Process each section
            for section in dimension_sections:
                section_lines = section.strip().split('\n')
                if not section_lines:
                    continue
                
                # First line contains dimension name
                dim_line = section_lines[0]
                dimension_name = dim_line.split(':', 1)[1].strip() if ':' in dim_line else "Unknown Dimension"
                
                # Find group name
                group_name = "General"
                for line in section_lines:
                    if line.strip().startswith("GROUP:"):
                        group_name = line.split(':', 1)[1].strip()
                        break
                
                # Find attributes
                attributes = []
                for i, line in enumerate(section_lines):
                    if line.strip().startswith("- ATTRIBUTE"):
                        if ':' in line:
                            attr_parts = line.split(':', 1)[1].strip().split('|')
                            
                            attr_name = attr_parts[0].strip()
                            data_type = attr_parts[1].strip() if len(attr_parts) > 1 else "text"
                            required = True if len(attr_parts) <= 2 or attr_parts[2].strip().lower() == 'true' else False
                            max_length = 2000
                            
                            if len(attr_parts) > 3:
                                try:
                                    max_length = int(attr_parts[3].strip())
                                except ValueError:
                                    pass
                            
                            attributes.append({
                                "id": slugify(attr_name),
                                "displayName": attr_name,
                                "dataType": data_type,
                                "required": required,
                                "maxLength": max_length,
                                "order": i + 1
                            })
                
                # Create dimension object
                if dimension_name and attributes:
                    dimension = {
                        "name": dimension_name,
                        "attributeGroups": [
                            {
                                "name": group_name,
                                "count": len(attributes),
                                "attributes": attributes
                            }
                        ]
                    }
                    results.append(dimension)
        else:
            # Process the regex matches
            for match in dimension_matches:
                section = match.strip()
                lines = section.split('\n')
                
                # First line is the dimension name
                dimension_name = lines[0].strip()
                
                # Find the group name
                group_name = "General"
                group_match = re.search(r'GROUP:\s*(.*)', section)
                if group_match:
                    group_name = group_match.group(1).strip()
                
                # Find all attribute lines
                attribute_lines = [line.strip() for line in lines if line.strip().startswith('- ATTRIBUTE')]
                
                attributes = []
                for i, attr_line in enumerate(attribute_lines):
                    # Extract just the part after "ATTRIBUTE N:"
                    attr_parts = attr_line.split(':', 1)[1].strip().split('|')
                    
                    attr_name = attr_parts[0].strip()
                    data_type = attr_parts[1].strip() if len(attr_parts) > 1 else "text"
                    required = True if len(attr_parts) <= 2 or attr_parts[2].strip().lower() == 'true' else False
                    max_length = 2000
                    
                    if len(attr_parts) > 3:
                        try:
                            max_length = int(attr_parts[3].strip())
                        except ValueError:
                            pass
                    
                    attributes.append({
                        "id": slugify(attr_name),
                        "displayName": attr_name,
                        "dataType": data_type,
                        "required": required,
                        "maxLength": max_length,
                        "order": i + 1
                    })
                
                # Add the dimension to results
                if dimension_name and attributes:
                    dimension = {
                        "name": dimension_name,
                        "attributeGroups": [
                            {
                                "name": group_name,
                                "count": len(attributes),
                                "attributes": attributes
                            }
                        ]
                    }
                    results.append(dimension)
        
        # Return empty list if no dimensions found (no fallback)
        if not results:
            print("[parse_response] No valid dimensions found, returning empty list")
            return []
        print(f"[parse_response] Successfully parsed {len(results)} dimensions")
        return results
    
    except Exception as e:
        print(f"[parse_response] Error parsing response: {str(e)}")
        import traceback
        print(f"[parse_response] Traceback: {traceback.format_exc()}")
        
        # Return empty list instead of fallback
        print("[parse_response] Returning empty list due to parsing error")
        return []

def convert_to_json_format(structured_dimensions):
    """
    Convert the structured dimensions back to JSON format for frontend compatibility.
    """
    # This is actually a no-op since our parser already created the right structure
    return structured_dimensions

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
    Analyze documents and briefing context to suggest schema enhancements for star schema.
    Simplified approach with direct structured data and terminal checkpoints.
    """
    try:
        print("\n====== SUGGEST SCHEMA ENHANCEMENTS ENDPOINT CALLED ======")
        
        # Parse request data
        context = json.loads(request.form['context'])
        print(f"[suggest] Received context: {context}")
        document_type = context.get('documentType', 'unknown')
        project_name = context.get('projectName', 'Unnamed Project')
       
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
                WHERE availability = 'All'
                ORDER BY order_num
            """)
        dimensions = cursor.fetchall()
        
        print(f"[suggest] Found {len(dimensions)} existing dimensions")
        for dim in dimensions:
            print(f"  - {dim['display_name']} ({dim['dimension_id']})")
        
        # Create dimensions data for the prompt
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
        total_attributes = 0
        
        for dim in dimensions:
            cursor.execute("""
                 SELECT attribute_id, display_name, data_type, required, max_length, order_num 
                 FROM acquisitionattributes 
                 WHERE dimension_id = %s
                 AND availability = 'All'                   
                 ORDER BY order_num
                """, (dim["id"],))
            
            attributes = cursor.fetchall()
            total_attributes += len(attributes)
            
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
        
        print(f"[suggest] Found {total_attributes} existing attributes across all dimensions")
        
        # STEP 3: Create the prompt for Claude with SIMPLIFIED FORMAT
        print("[suggest] STEP 3: Creating prompt for Claude")
        
        scope_text = briefing_context.get('scope', '')
        context_data = briefing_context.get('context', {})
        
        # Format dimensions and attributes for the prompt
        dimensions_text = json.dumps(dimensions_data, indent=2)
        attributes_text = json.dumps(attributes_data, indent=2)
        
        # Load prompt template with robust error handling
        try:
            prompt_template = get_prompt_template()
            print("[suggest] Successfully loaded prompt template")
            
            # Evaluate customer type - do this BEFORE format
            customer_type = "Standard M&A"
            print(f"[suggest] Customer type: {customer_type}")
            
            # Create the prompt with format
            prompt = prompt_template.format(
                project_name=project_name,
                customer_type=customer_type,
                scope_text=scope_text,
                dimensions_text=dimensions_text,
                attributes_text=attributes_text,
                uploaded_content=uploaded_content[:50000]  # Truncate if needed
            )
            print(f"[suggest] Successfully formatted prompt, length: {len(prompt)} characters")
            
        except Exception as e:
            print(f"[suggest] Error in prompt preparation: {str(e)}")
            import traceback
            print(f"[suggest] Traceback: {traceback.format_exc()}")
            
            # Use a much simpler fallback prompt
            prompt = f"""
You are a database expert. Please suggest 3 dimensions for M&A due diligence, each with 3-5 attributes.

Respond EXACTLY in this format:
DIMENSION 1: [Name]
GROUP: General
- ATTRIBUTE 1: [Attribute]|text|true|2000
- ATTRIBUTE 2: [Attribute]|text|true|2000
- ATTRIBUTE 3: [Attribute]|text|true|2000

DIMENSION 2: [Name]
GROUP: General
- ATTRIBUTE 1: [Attribute]|text|true|2000
- ATTRIBUTE 2: [Attribute]|text|true|2000
- ATTRIBUTE 3: [Attribute]|text|true|2000

DIMENSION 3: [Name]
GROUP: General
- ATTRIBUTE 1: [Attribute]|text|true|2000
- ATTRIBUTE 2: [Attribute]|text|true|2000
- ATTRIBUTE 3: [Attribute]|text|true|2000
"""
            print("[suggest] Using simple fallback prompt")

        # Print the prompt length for debugging
        print(f"[suggest] Final prompt length: {len(prompt)} characters")
        
        # STEP 4: Call Claude API with higher token limits
        print("[suggest] STEP 4: Calling Claude API")
        print("[suggest] ----- TERMINAL CHECKPOINT 1: ABOUT TO CALL CLAUDE -----")
        print("[suggest] Proceed? (automatically continuing in 5 seconds...)")
        import time
        time.sleep(5)  # Optional delay for manual terminal intervention
        
        response = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=50000,  # Increased token limit to 50,000
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # STEP 5: Process Claude's response
        response_text = response.content[0].text
        print("\n[suggest] ----- CLAUDE'S RAW RESPONSE -----")
        print(response_text)
        print("[suggest] ----- END OF CLAUDE'S RESPONSE -----\n")
        
        print("[suggest] ----- TERMINAL CHECKPOINT 2: RECEIVED CLAUDE RESPONSE -----")
        print("[suggest] Proceed with parsing? (automatically continuing in 5 seconds...)")
        time.sleep(5)  # Optional delay for manual terminal intervention
        
        # Parse the structured response
        suggested_enhancements = parse_claude_structured_response(response_text)
        
        # If no enhancements were found, return an empty response
        if not suggested_enhancements:
            print("[suggest] No valid dimensions were parsed from Claude's response")
            return jsonify({
                "projectName": project_name,
                "currentDimensions": [
                    {
                        "name": dim["display_name"],
                        "id": dim["dimension_id"],
                        "attributeCount": len(attributes_data.get(dim["dimension_id"], []))
                    }
                    for dim in dimensions
                ],
                "suggestedEnhancements": [],
                "suggestedDimensions": [],
                "error": "No valid dimensions could be parsed from the response."
            }), 200
        
        # Print the parsed results for verification
        print("\n[suggest] ----- PARSED RESULTS -----")
        for dim in suggested_enhancements:
            print(f"Dimension: {dim['name']}")
            for group in dim['attributeGroups']:
                print(f"  Group: {group['name']} ({group['count']} attributes)")
                for attr in group['attributes']:
                    print(f"    - {attr['displayName']} ({attr['dataType']}, required={attr['required']}, maxLength={attr['maxLength']})")
        print("[suggest] ----- END OF PARSED RESULTS -----\n")
        
        print("[suggest] ----- TERMINAL CHECKPOINT 3: PARSED RESPONSE -----")
        print("[suggest] Proceed with database insertion? (automatically continuing in 5 seconds...)")
        time.sleep(5)  # Optional delay for manual terminal intervention
        
        # STEP 6: Store the suggested dimensions and attributes in the database
        print("[suggest] STEP 6: Storing suggested dimensions and attributes")
        suggested_dimensions = []

        for enhancement in suggested_enhancements:
            try:
                # Create a dimension_id from the name
                dimension_name = enhancement["name"]
                dimension_id = slugify(dimension_name)
                
                # Check if dimension_id already exists
                cursor.execute("""
                    SELECT id FROM acquisitiondimensions WHERE dimension_id = %s
                """, (dimension_id,))
                
                existing_dim = cursor.fetchone()
                if existing_dim:
                    print(f"[suggest] Dimension {dimension_id} already exists, generating unique ID")
                    # Generate a unique ID by adding a suffix
                    base_id = dimension_id
                    suffix = 1
                    while True:
                        new_id = f"{base_id}_{suffix}"
                        cursor.execute("""
                            SELECT id FROM acquisitiondimensions WHERE dimension_id = %s
                        """, (new_id,))
                        if not cursor.fetchone():
                            dimension_id = new_id
                            break
                        suffix += 1
                
                # Insert the new dimension - now using our new schema structure
                # Note: program_type, order_num, availability, and hawkeyeview will be set automatically
                cursor.execute("""
                    INSERT INTO acquisitiondimensions 
                    (dimension_id, display_name, description, origin)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, dimension_id, display_name, program_type, order_num, availability, hawkeyeview
                """, (
                    dimension_id,
                    dimension_name,
                    f"Suggested by Claude based on customer materials",
                    "claude"  # origin
                ))
                
                dim_result = cursor.fetchone()
                if not dim_result:
                    print(f"[suggest] ERROR: Failed to insert dimension {dimension_name}, no ID returned")
                    continue  # Skip to next dimension
                    
                # Extract all the returned fields
                new_dimension_id = dim_result[0]
                returned_dimension_id = dim_result[1]
                returned_display_name = dim_result[2]
                returned_program_type = dim_result[3]
                returned_order_num = dim_result[4]
                returned_availability = dim_result[5]
                returned_hawkeyeview = dim_result[6]
                
                print(f"[suggest] Created new dimension: {dimension_name} (ID: {new_dimension_id}, dimension_id: {dimension_id})")
                print(f"[suggest] Program Type: {returned_program_type}, Order: {returned_order_num}")
                print(f"[suggest] Availability: {returned_availability}, View: {returned_hawkeyeview}")
                
                # Add to list for response
                suggested_dimensions.append({
                    "id": new_dimension_id,
                    "name": dimension_name,
                    "dimension_id": dimension_id
                })
                
                # Insert attributes for this dimension
                for group in enhancement.get("attributeGroups", []):
                    group_name = group.get("name", "General")
                    print(f"[suggest] Processing attribute group: {group_name}")
                    
                    for attr in group.get("attributes", []):
                        try:
                            # Ensure attribute_id is valid
                            attr_display_name = attr.get("displayName", "")
                            attribute_id = attr.get("id", slugify(attr_display_name))
                            
                            # Insert attribute - now using our new schema structure
                            # Note: availability and hawkeyeview will be set automatically
                            cursor.execute("""
                                INSERT INTO acquisitionattributes 
                                (dimension_id, attribute_id, display_name, data_type, required, max_length, origin)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                RETURNING id, attribute_id, display_name, order_num, availability, hawkeyeview
                            """, (
                                new_dimension_id,
                                attribute_id,
                                attr_display_name,
                                attr.get("dataType", "text"),
                                attr.get("required", True),
                                attr.get("maxLength", 2000),
                                "claude"  # origin
                            ))
                            
                            attr_result = cursor.fetchone()
                            if attr_result:
                                new_attr_id = attr_result[0]
                                returned_attr_id = attr_result[1]
                                returned_attr_name = attr_result[2]
                                returned_attr_order = attr_result[3]
                                returned_attr_availability = attr_result[4]
                                returned_attr_view = attr_result[5]
                                
                                print(f"[suggest]   - Created attribute: {attr_display_name} (ID: {new_attr_id}, attribute_id: {attribute_id})")
                                print(f"[suggest]   - Order: {returned_attr_order}, Availability: {returned_attr_availability}, View: {returned_attr_view}")
                            else:
                                print(f"[suggest]   - Failed to create attribute: {attr_display_name}")
                        except Exception as attr_err:
                            print(f"[suggest]   - Error creating attribute {attr_display_name}: {str(attr_err)}")
                            # Continue to the next attribute
                            continue
                            
            except Exception as dim_err:
                print(f"[suggest] Error processing dimension {enhancement.get('name', 'Unknown')}: {str(dim_err)}")
                # Continue to the next dimension
                continue

        # Commit the database transaction
        conn.commit()
        print("[suggest] Database changes committed successfully")

        print("[suggest] ----- TERMINAL CHECKPOINT 4: DATABASE UPDATED -----")
        print("[suggest] Database updates complete. Preparing frontend response...")

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
            "suggestedEnhancements": convert_to_json_format(suggested_enhancements),  # Convert to JSON format for frontend
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
