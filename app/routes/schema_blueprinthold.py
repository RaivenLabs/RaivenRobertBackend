# app/schema_blueprint.py
from flask import Blueprint, request, jsonify, current_app
import json
import yaml
import os
import boto3
import re
import copy
import psycopg2
import PyPDF2
import re
import tempfile
import traceback



from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad



from flask import session 
import datetime
from datetime import date
from psycopg2.extras import DictCursor
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from ..utilities.document_processor import extract_text_from_file, parse_csv, parse_excel
from ..utilities.schema_generator import generate_initial_schema, convert_to_sql, convert_to_json_schema




# Define pandas_available and handle imports with try/except
try:
    import pandas as pd
    import numpy as np
    pandas_available = True
except ImportError:
    pandas_available = False
    print("Warning: pandas is not installed. Excel and CSV processing will not work.")


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








def save_temp_file(file):
    """Save uploaded file to a temporary location and return the path"""
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    file.save(temp_file.name)
    return temp_file.name














def extract_rates_from_amendment_pdf(pdf_path, max_items=500):
    """Extract rate card data from a PDF amendment using AI assistance"""
    try:
        import PyPDF2
        
        print(f"[extract_rates_from_amendment_pdf] Processing PDF: {pdf_path}")
        
        # 1. Read the PDF content
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        pdf_text = ""
        
        # Extract text from all pages
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            pdf_text += f"\n\n--- PAGE {page_num + 1} ---\n\n{page_text}"
        
        # 2. Prepare prompt for Claude to extract the rate table
        prompt = f"""
You are a specialized data extraction assistant focused on extracting rate card tables from contract amendments.

I need you to extract the rate card table data from the following PDF content. This is from a contract amendment between NIKE and a consulting firm.

Look for sections titled "Exhibit B-1", "CONTRACTOR'S RATES", or similar. The table will have columns like Beeline Job Code, Job Title, and various regional bill rates.

Extract all rows from the rate card table and return them in a structured format.

Return your response as a JSON object with the following structure:
{{
  "rateItems": [
    {{
      // Include fields for EVERY column found in the rate table
      // The below are just examples - use the actual column names from the document
      // Convert column names to camelCase or snake_case for consistency
      "job_code": "B9071",  // or whatever field contains the job/role code
      "job_title": "Agile Lead 1",  // or whatever field contains the job/role title
      "us_region1_rate": 155.00,  // Numeric values should be actual numbers, not strings with $ signs
      // Include ALL other rate columns found in the table
    }},
    // Include ALL rows from the rate table
  ],
  "totalCount": 123,  // Total number of rate items found
  "columnInfo": [
    // Include an entry for EVERY column found in the rate table
    {{
      "original_name": "The exact column name as it appears in the document",
      "sql_column": "a_valid_sql_column_name",  // Convert to snake_case with only alphanumeric and underscore
      "sql_type": "TEXT",  // Use "TEXT", "NUMERIC(15,2)", or "DATE" as appropriate
      "mapping": "job_code"  // A suggested standard mapping if this appears to be a common field
    }},
    // One entry for each column
  ]
}}

IMPORTANT: 
1. Your response structure should ADAPT to match ALL columns found in the actual document
2. Do NOT limit yourself to just the example fields shown above
3. Include ALL rows from the rate table in your response
4. Make sure numeric values are actual numbers, not strings with currency symbols
5. For "sql_column", ensure names are valid for SQL (letters, numbers, underscores only)
6. If a value contains a dollar sign ($), remove it and convert to a numeric value

Here is the extracted PDF content:

{pdf_text[:50000]}  # Truncate if needed
"""
        
        print("[extract_rates_from_amendment_pdf] Calling Claude to extract rate table")
        
        # 3. Call Claude to extract the rate table
        response = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=100000,  # Large max_tokens to handle many rate items
            temperature=0.0,    # Use 0 for deterministic extraction
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # 4. Parse Claude's response
        extraction_result = {}
        try:
            # Extract the JSON response
            response_text = response.content[0].text
            
            # Look for JSON block
            import re
            import json
            
            json_pattern = r'```json\s*([\s\S]*?)\s*```'
            json_match = re.search(json_pattern, response_text)
            
            if json_match:
                extraction_result = json.loads(json_match.group(1))
            else:
                # Try to extract JSON directly if no code block
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    extraction_result = json.loads(response_text[json_start:json_end])
                else:
                    print("[extract_rates_from_amendment_pdf] Failed to extract JSON from Claude response")
        except Exception as e:
            print(f"[extract_rates_from_amendment_pdf] Error parsing Claude response: {str(e)}")
            traceback.print_exc()
            return [], 0, []
        
        # 5. Extract rate items and column info
        rate_items = extraction_result.get('rateItems', [])
        total_count = extraction_result.get('totalCount', len(rate_items))
        column_info = extraction_result.get('columnInfo', [])
        
        # Limit the number of preview items if needed
        preview_items = rate_items[:max_items]
        
        print(f"[extract_rates_from_amendment_pdf] Extracted {len(preview_items)} preview items out of {total_count} total")
        
        # 6. Print sample rows for verification
        print("\n==== CONFIRMING EXTRACTION BY CLAUDETTE! ====")
     
        
        print("Yay:)")
        
        print("============================================\n")
        
        return preview_items, total_count, column_info
        
    except Exception as e:
        print(f"Error extracting rates from PDF: {str(e)}")
        traceback.print_exc()
        return [], 0, []

















































def extract_rates_from_excel(excel_path, sheet_name=None, max_items=100):
    """Extract rates from an Excel file using heuristics to find the header row"""
    if not pandas_available:
        print("[extract_rates_from_excel] Pandas not available")
        return [], 0, []
        
    try:
        # If we don't have explicit guidance, use hardcoded values for your specific file
        if sheet_name is None:
            sheet_name = "Rate Card"  # Hardcoded sheet name
            
        print(f"[extract_rates_from_excel] Using sheet: {sheet_name}")
        
        # Get available sheets for debugging
        xls = pd.ExcelFile(excel_path)
        print(f"[extract_rates_from_excel] Available sheets: {xls.sheet_names}")
        
        # Read the Excel file with specified sheet
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        print(f"[extract_rates_from_excel] Read dataframe with shape: {df.shape}")
        
        # Find the header row using heuristics
        header_row = 0
        for i in range(min(20, len(df))):  # Check first 20 rows
            row_values = [str(val).lower() for val in df.iloc[i].values if pd.notna(val)]
            
            # Look for indicators of a header row: role + (level or region or usd)
            if 'role' in row_values and any(term in ' '.join(row_values) for term in ['level', 'region', 'usd']):
                header_row = i
                print(f"[extract_rates_from_excel] Found header row at index {header_row}")
                break
                
        # Use the header row as column names
        print(f"[extract_rates_from_excel] Using row {header_row} as header row")
        headers = df.iloc[header_row]
        df = pd.DataFrame(df.values[header_row+1:], columns=headers)
        print(f"[extract_rates_from_excel] After header adjustment, shape: {df.shape}")
        
        # Clean column names and continue processing as before...
        df.columns = [str(col).strip() if pd.notna(col) else f"column_{i}" 
                     for i, col in enumerate(df.columns)]
        
        # Drop empty rows
        df = df.dropna(how='all')
        print(f"[extract_rates_from_excel] After dropping empty rows, shape: {df.shape}")
        
        # Generate column info for database table creation
        column_info = []
        for i, column in enumerate(df.columns):
            col_name = str(column).strip() if pd.notna(column) else f"column_{i}"
            if not col_name:
                col_name = f"column_{i}"
                
            # Sanitize for SQL
            sql_column = re.sub(r'[^\w]', '_', col_name.lower()).strip('_')
            if not sql_column:
                sql_column = f"column_{i}"
                
            # Ensure uniqueness
            base_name = sql_column
            counter = 1
            while any(info['sql_column'] == sql_column for info in column_info):
                sql_column = f"{base_name}_{counter}"
                counter += 1
            
            # Determine data type
            sql_type = 'TEXT'  # Default
            sample_values = df[column].dropna().head(5).values.tolist() if len(df[column].dropna()) > 0 else []
            if sample_values:
                if all(isinstance(val, (int, np.integer)) for val in sample_values if pd.notna(val)):
                    sql_type = 'INTEGER'
                elif all(isinstance(val, (float, np.floating, int, np.integer)) for val in sample_values if pd.notna(val)):
                    sql_type = 'NUMERIC(15,2)'
                elif all(isinstance(val, (datetime.date, datetime.datetime)) for val in sample_values if pd.notna(val)):
                    sql_type = 'DATE'
            
            column_info.append({
                'original_name': col_name,
                'sql_column': sql_column,
                'sql_type': sql_type
            })
        
        print(f"[extract_rates_from_excel] Generated {len(column_info)} column definitions")
        
        # Process rows into rate items
        rate_items = []
        preview_count = min(max_items, len(df))
        
        for i in range(preview_count):
            if i >= len(df):
                break
                
            row = df.iloc[i]
            
            # Skip empty rows
            if all(pd.isna(val) or str(val).strip() == '' for val in row):
                continue
                
            # Create rate item
            rate_item = {}
            for j, col in enumerate(df.columns):
                if j >= len(column_info):
                    continue  # Skip if column_info is missing
                    
                col_name = column_info[j]['original_name']
                value = row.iloc[j] if j < len(row) else None
                
                # Handle different data types
                if pd.isna(value):
                    rate_item[col_name] = None
                elif isinstance(value, (int, np.integer)):
                    rate_item[col_name] = int(value)
                elif isinstance(value, (float, np.floating)):
                    rate_item[col_name] = float(value)
                elif isinstance(value, (datetime.date, datetime.datetime)):
                    rate_item[col_name] = value.isoformat()
                else:
                    rate_item[col_name] = str(value).strip()
            
            rate_items.append(rate_item)
            
        print(f"[extract_rates_from_excel] Processed {len(rate_items)} preview items")
        return rate_items, len(df), column_info
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        traceback.print_exc()
        return [], 0, []














def extract_rates_from_csv(csv_path, max_items=100):
    """Extract rates from a CSV file"""
    # Placeholder for CSV extraction logic
    import pandas as pd
    
    try:
        df = pd.read_csv(csv_path)
        # Assuming the CSV file has columns that correspond to our rate structure
        rate_items = []
        
        for i, row in df.head(max_items).iterrows():
            # Adapt this to match your CSV structure
            rate_item = {
                'role_code': str(row.get('Beeline Job Code', '')),
                'role_title': str(row.get('Beeline Job Title (with Level)', '')),
                'us_region1_rate': float(row.get('US Region 1 Bill Rate (USD)', 0)),
                'us_region2_rate': float(row.get('US Region 2 Bill Rate (USD)', 0)),
                'us_region3_rate': float(row.get('US Region 3 Bill Rate (USD)', 0)),
                # Add other rates as needed
            }
            rate_items.append(rate_item)
            
        return rate_items, len(df)
    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")
        return [], 0


























def extract_initial_rate_batch():
    """Simple placeholder function to extract rate items"""
    # For testing, just return some dummy data
    return [
        {"jobCode": "B9071", "jobTitle": "Agile Lead 1", "rate": 155.00},
        {"jobCode": "B9072", "jobTitle": "Agile Lead 2", "rate": 175.00},
        {"jobCode": "B9073", "jobTitle": "Agile Lead 3", "rate": 190.00}
    ]

def estimate_total_entries():
    """Simple placeholder function to estimate total entries"""
    # For testing, just return a fixed number
    return 100

def extract_rate_batch(file_path, offset, batch_size):
    """Simple placeholder function to extract a batch of rate items"""
    # For testing, just return some dummy data based on offset
    return [
        {"roleTitle": f"Role {i+offset}", "rateType": "hourly", "rate": 150.00+i+offset, "location": "US"} 
        for i in range(min(batch_size, 10))  # Return at most 10 items
    ]







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




@schema_blueprint.route('/extract-document-data', methods=['POST'])
def extract_document_data():
    """Extract key information from uploaded supplier documents"""
    try:
        print("\n====== EXTRACT DOCUMENT DATA ENDPOINT CALLED ======")
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        document_type = request.form.get('documentType', 'unknown')
        
        print(f"[extract] Processing {document_type} document: {file.filename}")
        
        # Save the file temporarily
        temp_dir = os.path.join(current_app.root_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(temp_path)
        
        # Extract text from the document
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        document_text = ""
        
        if file_ext == 'pdf':
            document_text = extract_text_from_file(temp_path, file_ext)
        elif file_ext == 'csv':
            document_text = parse_csv(temp_path)
        elif file_ext in ['xlsx', 'xls']:
            document_text = parse_excel(temp_path)
        else:
            # Read as text file
            with open(temp_path, 'r', encoding='utf-8', errors='replace') as f:
                document_text = f.read()
        
        print(f"[extract] Extracted {len(document_text)} characters from document")
        
        # Create a document-specific prompt based on document type
        prompt = get_document_extraction_prompt(document_type, document_text)
        
        # Call Claude to extract information
        response = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4096,
            temperature=0.0,  # Use 0 for deterministic extraction
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse Claude's response
        extracted_data = parse_extraction_response(response.content[0].text)
        print(f"[extract] Extracted data: {json.dumps(extracted_data, indent=2)}")
        
        # Clean up temp file
        os.remove(temp_path)
        
        print("====== EXTRACT DOCUMENT DATA ENDPOINT COMPLETE ======\n")
        return jsonify(extracted_data)
        
    except Exception as e:
        import traceback
        print("\n====== ERROR IN EXTRACT DOCUMENT DATA ENDPOINT ======")
        print(f"Exception: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("=========================================\n")
        
        return jsonify({"error": f"Error extracting document data: {str(e)}"}), 500

def get_document_extraction_prompt(document_type, document_text):
    """Generate a prompt for extracting data based on document type"""
    base_prompt = """
You are an expert at extracting information from legal and business documents.
Please extract the following key information from this {document_type} document.
Return ONLY a valid JSON object with the extracted information - no other explanation or text.

{extraction_fields}

Here is the document content:
{document_text}
"""
    
    # Define extraction fields based on document type
    if document_type == 'masterAgreement':
        extraction_fields = """
Extract these fields (return null if not found):
- name: The supplier/vendor company name
- msaReference: The MSA reference or contract number
- agreementType: The type of agreement (e.g., "Fixed Term", "Evergreen", "Time & Materials")
- effectiveDate: The effective date in YYYY-MM-DD format
- termEndDate: The end date in YYYY-MM-DD format (null if not specified or evergreen)
- autoRenewal: Boolean indicating if the agreement auto-renews
- address: The supplier's address
- website: The supplier's website
- contacts: An array of contact persons with name, role, email, and phone if available
- serviceCategories: Categories of services offered
"""
    elif document_type == 'rateCard':
        extraction_fields = """
Extract these fields (return null if not found):
- effectiveDate: The rate card effective date in YYYY-MM-DD format
- expirationDate: The rate card expiration date in YYYY-MM-DD format
- currency: The currency used for rates
- rateItems: An array of rate items, each containing:
  * roleTitle: The job title or role
  * rateType: The type of rate (hourly, daily, etc.)
  * rate: The numeric rate amount
  * location: The work location if specified
  * level: The seniority level if specified
"""
    elif document_type in ['amendments', 'localCountryAgreements', 'orderTemplates']:
        extraction_fields = """
Extract these fields (return null if not found):
- documentTitle: The title of the document
- effectiveDate: The effective date in YYYY-MM-DD format
- referenceNumber: Any reference number for this document
- relatedDocuments: Any references to other documents (MSA number, etc.)
- keyChanges: A summary of key changes or terms introduced by this document
"""

    elif document_type == 'serviceOrders':
        extraction_fields = """
Extract these fields (return null if not found):
- orderNumber: The service order number or reference
- msaReference: The reference or agreement number of the master agreement this order falls under
- msaName: The complete name/title of the master agreement (e.g., "Master Professional Services Agreement")
- msaEffectiveDate: The date of the master agreement in YYYY-MM-DD format
- purchaser: The full legal name of the client/purchaser
- contractor: The full legal name of the service provider/contractor
- projectName: The name or title of the project or service
- projectDescription: A brief description of the project scope
- effectiveDate: The order start date in YYYY-MM-DD format
- completionDate: The order end date in YYYY-MM-DD format
- totalValue: The total monetary value of the order (numeric value only)
- currency: The currency used for the order value
- serviceCategories: An array of categories of services provided in this order
- resourceTypes: An array of types of resources/roles utilized in this order
- deliverables: An array of key deliverables specified in the order
- specialTerms: Any special terms or conditions specific to this order
- purchaserAddress: The address of the purchaser
- contractorAddress: The address of the contractor
- purchaserContact: Name and contact information of the purchaser's representative
- contractorContact: Name and contact information of the contractor's representative

Pay special attention to the master agreement details. Look for the full formal name of the master agreement, which is typically mentioned in phrases like "subject to the terms of the [FULL AGREEMENT NAME] between Purchaser and Contractor dated [DATE]". The msaName should include the complete title of the agreement (e.g., "Master Professional Services Agreement" or "Master Consulting Agreement").
"""

    else:
        extraction_fields = """
Extract any key information you can find, including:
- dates: Any important dates in YYYY-MM-DD format
- parties: Names of organizations involved
- references: Any contract or document references
- amounts: Any monetary amounts mentioned
- terms: Key terms or conditions
"""


    
    # Limit document text to avoid token limits
    max_text_length = 50000
    if len(document_text) > max_text_length:
        document_text = document_text[:max_text_length] + "... [document truncated]"
    
    return base_prompt.format(
        document_type=document_type,
        extraction_fields=extraction_fields,
        document_text=document_text
    )

def parse_extraction_response(response_text):
    """Parse Claude's JSON response safely"""
    try:
        # Find JSON object in response text
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        json_match = re.search(json_pattern, response_text)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no code block, try to find JSON directly
            # Look for the first { and the last }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx+1]
            else:
                print("[parse_extraction] No JSON object found in response")
                return {}
        
        return json.loads(json_str)
    except Exception as e:
        print(f"[parse_extraction] Error parsing JSON response: {str(e)}")
        print(f"[parse_extraction] Raw response: {response_text}")
        # Return empty dict rather than fail
        return {}
    
  
  
  
  
  
@schema_blueprint.route('/extract-rate-card', methods=['POST'])
def extract_rate_card():
    """Extract and preview rate card data from uploaded file"""
    try:
        print("\n====== EXTRACT RATE CARD ENDPOINT CALLED ======")
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        # Get form parameters
        card_type = request.form.get('cardType', 'standalone')
        provider_id = request.form.get('providerId')
        customer_id = request.form.get('customerId')
        master_agreement_id = request.form.get('masterAgreementId')
        
        print(f"[extract_rate_card] Processing {card_type} rate card: {file.filename}")
        
        # Save the file temporarily
        temp_path = save_temp_file(file)
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        # Process initial batch of rate items based on file type and card type
        try:
            initial_rates = []
            estimated_total = 0
            column_info = []
            
            if file_ext in ['xlsx', 'xls']:
                # Use our hardcoded approach for Excel files
                initial_rates, estimated_total, column_info = extract_rates_from_excel(temp_path)
            elif file_ext == 'pdf':
              
                    initial_rates, estimated_total, column_info = extract_rates_from_amendment_pdf(temp_path)
               
            elif file_ext == 'csv':
                initial_rates, estimated_total, column_info = extract_rates_from_csv(temp_path)
            else:
                return jsonify({"error": f"Unsupported file type: {file_ext}"}), 400
                
            print(f"[extract_rate_card] Extracted {len(initial_rates)} rate items (estimated total: {estimated_total})")
        except Exception as e:
            print(f"[extract_rate_card] Error extracting rate items: {str(e)}")
            traceback.print_exc()
            initial_rates = []
            estimated_total = 0
            column_info = []
            
        # Store file path in session for later processing
        session['pending_rate_card'] = {
            'file_path': temp_path,
            'file_ext': file_ext,
            'card_type': card_type,
            'provider_id': provider_id,
            'customer_id': customer_id,
            'master_agreement_id': master_agreement_id,
            'column_info': column_info,
            'processed': False
        }
        
        print("====== EXTRACT RATE CARD ENDPOINT COMPLETE ======\n")
        return jsonify({
            'previewItems': initial_rates,
            'estimatedTotal': estimated_total,
            'columnInfo': column_info
        })
        
    except Exception as e:
        traceback.print_exc()
        print("\n====== ERROR IN EXTRACT RATE CARD ENDPOINT ======")
        print(f"Exception: {str(e)}")
        print("=========================================\n")
        
        return jsonify({"error": f"Error extracting rate card: {str(e)}"}), 500

















@schema_blueprint.route('/process-complete-rate-card', methods=['POST'])
def process_complete_rate_card():
    # Get the pending rate card info
    if 'pending_rate_card' not in session:
        return jsonify({'error': 'No pending rate card to process'}), 400
            
    pending_info = session['pending_rate_card']
    
    # Process the entire file and store directly to database
    provider_id = pending_info['provider_id']
    file_path = pending_info['file_path']
        
    # Create rate card header
    conn = get_db_connection()  # Get a new connection
    if not conn:
        return jsonify({'error': 'Unable to connect to database'}), 500
    
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO rate_cards (provider_id, effective_date, name)
            VALUES (%s, %s, %s) RETURNING id
        """, (provider_id, date.today(), "Extracted Rate Card"))
        rate_card_id = cursor.fetchone()[0]
        
        # Process the file in batches
        batch_size = 50
        offset = 0
        total_processed = 0
        
        while True:
            # Extract the next batch
            batch = extract_rate_batch(file_path, offset, batch_size)
            if not batch:
                break  # No more items to process
                    
            # Insert batch into database
            for item in batch:
                cursor.execute("""
                    INSERT INTO rate_card_items (rate_card_id, role_title, rate_type, rate, location)
                    VALUES (%s, %s, %s, %s, %s)
                """, (rate_card_id, item['roleTitle'], item['rateType'], item['rate'], item['location']))
                    
            conn.commit()
            total_processed += len(batch)
            offset += batch_size
        
        # Clean up
        os.remove(file_path)
        session.pop('pending_rate_card', None)
        
        return jsonify({
            'success': True,
            'rateCardId': rate_card_id,
            'totalProcessed': total_processed
        })
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()
















































def extract_rates_from_amendment_pdfhold(pdf_path, max_items=100):
    """Extract rate card data from a PDF amendment"""
    try:
        # Import PDF processing libraries
        import PyPDF2
        import re
        import tabula
        import pandas as pd
        import numpy as np
        
        print(f"[extract_rates_from_amendment_pdf] Processing PDF: {pdf_path}")
        
        # 1. Find the rate card section in the PDF
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        rate_card_start_page = None
        rate_card_markers = ["CONTRACTOR'S RATES", "Exhibit \"B-1\"", "Exhibit B-1", "RATES"]
        
        for i in range(len(pdf_reader.pages)):
            text = pdf_reader.pages[i].extract_text()
            if any(marker in text for marker in rate_card_markers):
                rate_card_start_page = i + 1  # Pages are 1-indexed in tabula
                print(f"[extract_rates_from_amendment_pdf] Found rate card on page {rate_card_start_page}")
                break
                
        if rate_card_start_page is None:
            print("[extract_rates_from_amendment_pdf] Could not find rate card section")
            return [], 0, []
            
        # 2. Extract tables from the rate card pages
        all_tables = []
        current_page = rate_card_start_page
        
        while current_page <= len(pdf_reader.pages):
            try:
                # Try to extract tables from the current page
                tables = tabula.read_pdf(pdf_path, pages=current_page, multiple_tables=True)
                
                if tables:
                    # Check if any table looks like a rate table (has "$" or numbers)
                    for table in tables:
                        if not table.empty and table.shape[1] >= 3:  # At least 3 columns
                            # Look for "$" or number patterns to confirm it's a rate table
                            table_str = table.to_string()
                            if "$" in table_str or re.search(r'\d+\.\d{2}', table_str):
                                all_tables.append(table)
                                print(f"[extract_rates_from_amendment_pdf] Found rate table on page {current_page} with shape {table.shape}")
                
                # Check the next page
                current_page += 1
                
                # Stop if we've processed 10 pages past the start (to avoid unnecessary processing)
                if current_page > rate_card_start_page + 10:
                    break
                    
            except Exception as e:
                print(f"[extract_rates_from_amendment_pdf] Error processing page {current_page}: {str(e)}")
                current_page += 1
                
        if not all_tables:
            print("[extract_rates_from_amendment_pdf] No rate tables found")
            return [], 0, []
            
        # 3. Process and combine the extracted tables
        combined_df = None
        
        for table in all_tables:
            # Clean up table - drop rows where all cells are empty or NaN
            table = table.dropna(how='all')
            
            # Check for header rows
            for i, row in table.iterrows():
                row_values = [str(val).lower() for val in row if pd.notna(val)]
                if any("job code" in val or "beeline" in val for val in row_values):
                    # Use this as the header row
                    headers = row
                    table = pd.DataFrame(table.values[i+1:], columns=headers)
                    break
            
            # Append to combined table
            if combined_df is None:
                combined_df = table
            else:
                # Try to append the table to the combined one
                try:
                    # Make sure columns match
                    table.columns = combined_df.columns[:len(table.columns)]
                    combined_df = pd.concat([combined_df, table], ignore_index=True)
                except Exception as e:
                    print(f"[extract_rates_from_amendment_pdf] Error combining tables: {str(e)}")
        
        if combined_df is None or combined_df.empty:
            print("[extract_rates_from_amendment_pdf] Failed to process rate tables")
            return [], 0, []
            
        # 4. Generate column info for database table creation
        column_info = []
        for i, column in enumerate(combined_df.columns):
            col_name = str(column).strip() if pd.notna(column) else f"column_{i}"
            if not col_name or col_name.lower() == 'nan':
                col_name = f"column_{i}"
                
            # Sanitize for SQL
            sql_column = re.sub(r'[^\w]', '_', col_name.lower()).strip('_')
            if not sql_column:
                sql_column = f"column_{i}"
                
            # Clean up common patterns in rate card columns
            sql_column = sql_column.replace('__', '_')
            
            # Ensure uniqueness
            base_name = sql_column
            counter = 1
            while any(info['sql_column'] == sql_column for info in column_info):
                sql_column = f"{base_name}_{counter}"
                counter += 1
                
            # Determine data type
            sample_values = combined_df[column].dropna().head(5).values.tolist() if len(combined_df[column].dropna()) > 0 else []
            
            sql_type = 'TEXT'  # Default
            if sample_values:
                # Check if it's likely a numeric field (e.g., rate amount)
                if all(isinstance(val, (int, float, np.integer, np.floating)) or 
                       (isinstance(val, str) and re.match(r'^\$?\s*\d+\.?\d*$', val)) 
                       for val in sample_values if pd.notna(val)):
                    sql_type = 'NUMERIC(15,2)'
                elif all(isinstance(val, (datetime.date, datetime.datetime)) for val in sample_values if pd.notna(val)):
                    sql_type = 'DATE'
                    
            column_info.append({
                'original_name': col_name,
                'sql_column': sql_column,
                'sql_type': sql_type
            })
        
        print(f"[extract_rates_from_amendment_pdf] Generated {len(column_info)} column definitions")
        
        # 5. Extract the rate items
        rate_items = []
        for i, row in combined_df.head(max_items).iterrows():
            # Skip rows that don't look like rate data
            if all(pd.isna(val) or str(val).strip() == '' for val in row):
                continue
                
            # Create rate item
            rate_item = {}
            for j, col in enumerate(combined_df.columns):
                if j >= len(column_info):
                    continue
                    
                col_name = column_info[j]['original_name']
                value = row.iloc[j] if j < len(row) else None
                
                # Process the value
                if pd.isna(value):
                    rate_item[col_name] = None
                elif isinstance(value, (int, np.integer)):
                    rate_item[col_name] = int(value)
                elif isinstance(value, (float, np.floating)):
                    rate_item[col_name] = float(value)
                elif isinstance(value, (datetime.date, datetime.datetime)):
                    rate_item[col_name] = value.isoformat()
                elif isinstance(value, str) and '$' in value:
                    # Handle dollar amounts in string form
                    clean_value = value.replace('$', '').replace(',', '').strip()
                    try:
                        rate_item[col_name] = float(clean_value)
                    except ValueError:
                        rate_item[col_name] = value
                else:
                    rate_item[col_name] = str(value).strip()
            
            # Add the item if it has data
            if any(val is not None and val != '' for val in rate_item.values()):
                rate_items.append(rate_item)
        
        total_count = len(combined_df)
        print(f"[extract_rates_from_amendment_pdf] Processed {len(rate_items)} preview items out of {total_count} total")
        
        # Try to identify key fields for mapping
        for item in column_info:
            lower_name = item['original_name'].lower()
            # Tag important fields for database mapping
            if any(term in lower_name for term in ['code', 'beeline', 'job code']):
                item['mapping'] = 'job_code'
            elif any(term in lower_name for term in ['title', 'job title', 'role']):
                item['mapping'] = 'job_title'
            elif any(term in lower_name for term in ['level']):
                item['mapping'] = 'job_level'
            elif any(term in lower_name for term in ['region 1', 'us region 1']):
                item['mapping'] = 'us_region1_rate'
            elif any(term in lower_name for term in ['region 2', 'us region 2']):
                item['mapping'] = 'us_region2_rate'
            elif any(term in lower_name for term in ['region 3', 'us region 3']):
                item['mapping'] = 'us_region3_rate'
             
             
             
                
        print("\n==== SAMPLE COMPLETE ROWS FROM EXTRACTION ====")
        sample_roles = ["Agile Lead 3", "Architect - Application 3", "Engagement Manager", "Data Analyst 3", "Consultant - IT"]
        sample_count = 0

        for item in rate_items:
            for role_name in sample_roles:
                if role_name.lower() in str(item).lower() and sample_count < 5:
                    print(f"\nROLE: {role_name}")
                    for key, value in item.items():
                        print(f"  {key}: {value}")
                    sample_count += 1
                    break
        print("============================================\n")         
                
        return rate_items, total_count, column_info
        
    except Exception as e:
        print(f"Error extracting rates from PDF: {str(e)}")
        traceback.print_exc()
        return [], 0, []

# Helper function for Python environments without tabula
def extract_rates_from_pdf_using_native(pdf_path, max_items=100):
    """Fallback PDF extraction using just PyPDF2 and regex when tabula is not available"""
    try:
        import PyPDF2
        import re
        
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        
        # Find rate card pages
        rate_card_start_page = None
        for i in range(len(pdf_reader.pages)):
            text = pdf_reader.pages[i].extract_text()
            if "CONTRACTOR'S RATES" in text or "Exhibit B-1" in text:
                rate_card_start_page = i
                break
        
        if rate_card_start_page is None:
            return [], 0, []
            
        # Extract text from rate card pages
        all_text = ""
        for i in range(rate_card_start_page, len(pdf_reader.pages)):
            all_text += pdf_reader.pages[i].extract_text() + "\n"
            
        # Use regex to find rate entries
        # Pattern looks for: code, title, and dollar amounts
        rate_pattern = re.compile(r'(B\d{4})\s+([\w\s\-]+?)\s+\$\s*(\d+\.\d{2})\s+\$\s*(\d+\.\d{2})\s+\$\s*(\d+\.\d{2})')
        
        matches = rate_pattern.finditer(all_text)
        
        rate_items = []
        match_count = 0
        
        for match in matches:
            if match_count >= max_items:
                break
                
            code = match.group(1)
            title = match.group(2).strip()
            us_region1 = float(match.group(3))
            us_region2 = float(match.group(4))
            us_region3 = float(match.group(5))
            
            rate_item = {
                'job_code': code,
                'job_title': title,
                'us_region1_rate': us_region1,
                'us_region2_rate': us_region2,
                'us_region3_rate': us_region3
            }
            
            rate_items.append(rate_item)
            match_count += 1
        
        # Generate column info
        column_info = [
            {'original_name': 'job_code', 'sql_column': 'job_code', 'sql_type': 'TEXT', 'mapping': 'job_code'},
            {'original_name': 'job_title', 'sql_column': 'job_title', 'sql_type': 'TEXT', 'mapping': 'job_title'},
            {'original_name': 'us_region1_rate', 'sql_column': 'us_region1_rate', 'sql_type': 'NUMERIC(15,2)', 'mapping': 'us_region1_rate'},
            {'original_name': 'us_region2_rate', 'sql_column': 'us_region2_rate', 'sql_type': 'NUMERIC(15,2)', 'mapping': 'us_region2_rate'},
            {'original_name': 'us_region3_rate', 'sql_column': 'us_region3_rate', 'sql_type': 'NUMERIC(15,2)', 'mapping': 'us_region3_rate'}
        ]
        
        # Count total matches for estimation
        total_matches = len(re.findall(r'B\d{4}', all_text))
        
                # Add this after line 118 (after processing the data but before returning it)
        # This will print complete row details for 5 sample roles
        print("\n==== SAMPLE COMPLETE ROWS FROM EXTRACTION ====")
        sample_roles = ["Agile Lead 3", "Architect - Application 3", "Engagement Manager", "Data Analyst 3", "Consultant - IT"]
        sample_count = 0

        for item in rate_items:
            for role_name in sample_roles:
                if role_name.lower() in str(item).lower() and sample_count < 5:
                    print(f"\nROLE: {role_name}")
                    for key, value in item.items():
                        print(f"  {key}: {value}")
                    sample_count += 1
                    break
        print("============================================\n")
                
        
        
        return rate_items, total_matches, column_info
        
    except Exception as e:
        print(f"Error in native PDF extraction: {str(e)}")
        traceback.print_exc()
        return [], 0, []
    
    
    
def extract_rates_from_amendment_pdfholdWednesday(pdf_path, max_items=100):
    """Extract rate card data from a PDF amendment"""
    try:
        # Import PDF processing libraries
        import PyPDF2
        import re
        import tabula
        import pandas as pd
        import numpy as np
        
        print(f"[extract_rates_from_amendment_pdf] Processing PDF: {pdf_path}")
        
        # 1. Find the rate card section in the PDF
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        rate_card_start_page = None
        rate_card_markers = ["CONTRACTOR'S RATES", "Exhibit \"B-1\"", "Exhibit B-1", "RATES"]
        
        for i in range(len(pdf_reader.pages)):
            text = pdf_reader.pages[i].extract_text()
            if any(marker in text for marker in rate_card_markers):
                rate_card_start_page = i + 1  # Pages are 1-indexed in tabula
                print(f"[extract_rates_from_amendment_pdf] Found rate card on page {rate_card_start_page}")
                break
                
        if rate_card_start_page is None:
            print("[extract_rates_from_amendment_pdf] Could not find rate card section")
            return [], 0, []
            
        # 2. Extract tables from the rate card pages
        all_tables = []
        current_page = rate_card_start_page
        
        while current_page <= len(pdf_reader.pages):
            try:
                # Try to extract tables from the current page
                tables = tabula.read_pdf(pdf_path, pages=current_page, multiple_tables=True)
                
                if tables:
                    # Check if any table looks like a rate table (has "$" or numbers)
                    for table in tables:
                        if not table.empty and table.shape[1] >= 3:  # At least 3 columns
                            # Look for "$" or number patterns to confirm it's a rate table
                            table_str = table.to_string()
                            if "$" in table_str or re.search(r'\d+\.\d{2}', table_str):
                                all_tables.append(table)
                                print(f"[extract_rates_from_amendment_pdf] Found rate table on page {current_page} with shape {table.shape}")
                
                # Check the next page
                current_page += 1
                
                # Stop if we've processed 10 pages past the start (to avoid unnecessary processing)
                if current_page > rate_card_start_page + 10:
                    break
                    
            except Exception as e:
                print(f"[extract_rates_from_amendment_pdf] Error processing page {current_page}: {str(e)}")
                current_page += 1
                
        if not all_tables:
            print("[extract_rates_from_amendment_pdf] No rate tables found")
            return [], 0, []
            
        # 3. Process and combine the extracted tables
        combined_df = None
        
        for table in all_tables:
            # Clean up table - drop rows where all cells are empty or NaN
            table = table.dropna(how='all')
            
            # Check for header rows
            for i, row in table.iterrows():
                row_values = [str(val).lower() for val in row if pd.notna(val)]
                if any("job code" in val or "beeline" in val for val in row_values):
                    # Use this as the header row
                    headers = row
                    table = pd.DataFrame(table.values[i+1:], columns=headers)
                    break
            
            # Append to combined table
            if combined_df is None:
                combined_df = table
            else:
                # Try to append the table to the combined one
                try:
                    # Make sure columns match
                    table.columns = combined_df.columns[:len(table.columns)]
                    combined_df = pd.concat([combined_df, table], ignore_index=True)
                except Exception as e:
                    print(f"[extract_rates_from_amendment_pdf] Error combining tables: {str(e)}")
        
        if combined_df is None or combined_df.empty:
            print("[extract_rates_from_amendment_pdf] Failed to process rate tables")
            return [], 0, []
            
        # 4. Generate column info for database table creation
        column_info = []
        for i, column in enumerate(combined_df.columns):
            col_name = str(column).strip() if pd.notna(column) else f"column_{i}"
            if not col_name or col_name.lower() == 'nan':
                col_name = f"column_{i}"
                
            # Sanitize for SQL
            sql_column = re.sub(r'[^\w]', '_', col_name.lower()).strip('_')
            if not sql_column:
                sql_column = f"column_{i}"
                
            # Clean up common patterns in rate card columns
            sql_column = sql_column.replace('__', '_')
            
            # Ensure uniqueness
            base_name = sql_column
            counter = 1
            while any(info['sql_column'] == sql_column for info in column_info):
                sql_column = f"{base_name}_{counter}"
                counter += 1
                
            # Determine data type
            sample_values = combined_df[column].dropna().head(5).values.tolist() if len(combined_df[column].dropna()) > 0 else []
            
            sql_type = 'TEXT'  # Default
            if sample_values:
                # Check if it's likely a numeric field (e.g., rate amount)
                if all(isinstance(val, (int, float, np.integer, np.floating)) or 
                       (isinstance(val, str) and re.match(r'^\$?\s*\d+\.?\d*$', val)) 
                       for val in sample_values if pd.notna(val)):
                    sql_type = 'NUMERIC(15,2)'
                elif all(isinstance(val, (datetime.date, datetime.datetime)) for val in sample_values if pd.notna(val)):
                    sql_type = 'DATE'
                    
            column_info.append({
                'original_name': col_name,
                'sql_column': sql_column,
                'sql_type': sql_type
            })
        
        print(f"[extract_rates_from_amendment_pdf] Generated {len(column_info)} column definitions")
        
        # 5. Extract the rate items
        rate_items = []
        for i, row in combined_df.head(max_items).iterrows():
            # Skip rows that don't look like rate data
            if all(pd.isna(val) or str(val).strip() == '' for val in row):
                continue
                
            # Create rate item
            rate_item = {}
            for j, col in enumerate(combined_df.columns):
                if j >= len(column_info):
                    continue
                    
                col_name = column_info[j]['original_name']
                value = row.iloc[j] if j < len(row) else None
                
                # Process the value
                if pd.isna(value):
                    rate_item[col_name] = None
                elif isinstance(value, (int, np.integer)):
                    rate_item[col_name] = int(value)
                elif isinstance(value, (float, np.floating)):
                    rate_item[col_name] = float(value)
                elif isinstance(value, (datetime.date, datetime.datetime)):
                    rate_item[col_name] = value.isoformat()
                elif isinstance(value, str) and '$' in value:
                    # Handle dollar amounts in string form
                    clean_value = value.replace('$', '').replace(',', '').strip()
                    try:
                        rate_item[col_name] = float(clean_value)
                    except ValueError:
                        rate_item[col_name] = value
                else:
                    rate_item[col_name] = str(value).strip()
            
            # Add the item if it has data
            if any(val is not None and val != '' for val in rate_item.values()):
                rate_items.append(rate_item)
        
        total_count = len(combined_df)
        print(f"[extract_rates_from_amendment_pdf] Processed {len(rate_items)} preview items out of {total_count} total")
        
        # Try to identify key fields for mapping
        for item in column_info:
            lower_name = item['original_name'].lower()
            # Tag important fields for database mapping
            if any(term in lower_name for term in ['code', 'beeline', 'job code']):
                item['mapping'] = 'job_code'
            elif any(term in lower_name for term in ['title', 'job title', 'role']):
                item['mapping'] = 'job_title'
            elif any(term in lower_name for term in ['level']):
                item['mapping'] = 'job_level'
            elif any(term in lower_name for term in ['region 1', 'us region 1']):
                item['mapping'] = 'us_region1_rate'
            elif any(term in lower_name for term in ['region 2', 'us region 2']):
                item['mapping'] = 'us_region2_rate'
            elif any(term in lower_name for term in ['region 3', 'us region 3']):
                item['mapping'] = 'us_region3_rate'
             
             
             
                
        print("\n==== SAMPLE COMPLETE ROWS FROM EXTRACTION ====")
        sample_roles = ["Agile Lead 3", "Architect - Application Niche 3", "Engagement Manager", "Data Analyst 3", "Consultant - IT"]
        sample_count = 0

        for item in rate_items:
            for role_name in sample_roles:
                if role_name.lower() in str(item).lower() and sample_count < 5:
                    print(f"\nROLE: {role_name}")
                    for key, value in item.items():
                        print(f"  {key}: {value}")
                    sample_count += 1
                    break
        print("============================================\n")         
                
        return rate_items, total_count, column_info
        
    except Exception as e:
        print(f"Error extracting rates from PDF: {str(e)}")
        traceback.print_exc()
        return [], 0, []



