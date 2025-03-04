# app/schema_blueprint.py
from flask import Blueprint, request, jsonify
import json
import yaml
import os
import tempfile
from werkzeug.utils import secure_filename
from ..utilities.document_processor import extract_text_from_file, parse_csv, parse_excel
from ..utilities.schema_generator import generate_initial_schema, convert_to_sql, convert_to_json_schema

# Use the existing Anthropic client from main_blueprint
from .main_routes import anthropic_client

# Create a Blueprint for schema generation routes
schema_blueprint = Blueprint('schema', __name__, url_prefix='/api')

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'csv', 'xlsx', 'json'}

def allowed_file(filename):
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
        is_development = current_app.config.get('FLASK_ENV') == 'development'
        
        if is_development and os.environ.get('USE_LOCAL_FILES', 'False').lower() == 'true':
            # Save to local file system in development
            schema_dir = os.path.abspath(os.path.join(
                current_app.root_path, '..', '..', 'static', 'data',
                'schemas', project_id
            ))
            os.makedirs(schema_dir, exist_ok=True)
            
            schema_path = os.path.join(schema_dir, 'schema.json')
            with open(schema_path, 'w') as f:
                json.dump(schema_data, f, indent=2)
        else:
            # Save to S3 in production
            s3_client.put_object(
                Bucket='juniperproductiondata',
                Key=f'data/schemas/{project_id}/schema.json',
                Body=json.dumps(schema_data, indent=2)
            )
        return True
    except Exception as e:
        current_app.logger.error(f"Error saving schema: {str(e)}")
        return False




@schema_blueprint.route('/analyze-documents', methods=['POST'])
def analyze_documents():
    """Analyze uploaded documents to identify data structure patterns"""
    if 'context' not in request.form:
        return jsonify({"error": "Missing context information"}), 400
    
    context = json.loads(request.form['context'])
    document_type = context.get('documentType', 'unknown')
    extraction_goals = context.get('extractionGoals', [])

    # Check if any files were uploaded
    files = []
    for key in request.files:
        if key.startswith('file_'):
            file = request.files[key]
            if file and allowed_file(file.filename):
                files.append(file)
    
    if not files:
        return jsonify({"error": "No valid files uploaded"}), 400

    # Process each file
    results = []
    for file in files:
        try:
            filename = secure_filename(file.filename)
            file_extension = filename.rsplit('.', 1)[1].lower()
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp:
                file.save(temp.name)
                
                # Process file based on type
                if file_extension in ['pdf', 'docx', 'txt']:
                    file_content = extract_text_from_file(temp.name, file_extension)
                    file_type = "document"
                    
                    # Analyze text with Claude
                    document_analysis = analyze_text_with_claude(
                        file_content, 
                        document_type, 
                        extraction_goals
                    )
                    analysis = {
                        "file_type": file_type,
                        "filename": filename,
                        "analysis": document_analysis
                    }
                
                elif file_extension == 'csv':
                    df = parse_csv(temp.name)
                    file_type = "structured_data"
                    
                    # Get column info and sample data
                    columns = df.columns.tolist()
                    sample_data = df.head(5).to_dict(orient='records')
                    data_types = {col: str(df[col].dtype) for col in columns}
                    
                    analysis = {
                        "file_type": file_type,
                        "filename": filename,
                        "columns": columns,
                        "data_types": data_types,
                        "sample_data": sample_data,
                        "row_count": len(df)
                    }
                
                elif file_extension == 'xlsx':
                    sheets_data = parse_excel(temp.name)
                    file_type = "structured_data"
                    
                    analysis = {
                        "file_type": file_type,
                        "filename": filename,
                        "sheets": sheets_data
                    }
                
                elif file_extension == 'json':
                    with open(temp.name, 'r') as f:
                        json_data = json.load(f)
                    
                    file_type = "structured_data"
                    
                    # Analyze JSON structure
                    analysis = {
                        "file_type": file_type,
                        "filename": filename,
                        "structure": analyze_json_structure(json_data)
                    }
                
                results.append(analysis)
                
                # Clean up temporary file
                os.unlink(temp.name)
                
        except Exception as e:
            return jsonify({"error": f"Error processing file {file.filename}: {str(e)}"}), 500
    
    # Try to load briefing context for more information
    try:
        with open('app/data/context.yaml', 'r') as f:
            briefing_context = yaml.safe_load(f)
    except:
        briefing_context = None

    # Return consolidated analysis results
    return jsonify({
        "document_type": document_type,
        "extraction_goals": extraction_goals,
        "file_analyses": results,
        "briefing_context": briefing_context
    })

@schema_blueprint.route('/generate-schema', methods=['POST'])
def generate_schema():
    """Generate a schema proposal based on document analysis"""
    data = request.json
    
    if not data or 'analysisResults' not in data:
        return jsonify({"error": "Missing analysis results"}), 400
    
    analysis_results = data['analysisResults']
    context = data.get('context', {})
    model_settings = data.get('modelSettings', {
        'temperature': 0.2,
        'maxTokens': 4000,
        'model': 'claude-3-7-sonnet-20250219'
    })
    
    try:
        # Build prompt for Claude
        prompt = build_schema_generation_prompt(analysis_results, context)
        
        # Call Claude API
        response = client.messages.create(
            model=model_settings.get('model', 'claude-3-7-sonnet-20250219'),
            max_tokens=model_settings.get('maxTokens', 4000),
            temperature=model_settings.get('temperature', 0.2),
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and parse the schema from Claude's response
        schema_json = extract_schema_from_response(response.content)
        
        # Ensure the schema has the expected structure
        if 'name' not in schema_json:
            schema_json['name'] = f"{context.get('documentType', 'document')}_schema"
        
        if 'fields' not in schema_json:
            return jsonify({"error": "Generated schema is missing fields"}), 500
        
        return jsonify(schema_json)
        
    except Exception as e:
        return jsonify({"error": f"Error generating schema: {str(e)}"}), 500

@schema_blueprint.route('/generate-export-code', methods=['POST'])
def generate_export_code():
    """Generate SQL DDL or JSON Schema from a schema definition"""
    data = request.json
    
    if not data or 'schema' not in data or 'format' not in data:
        return jsonify({"error": "Missing schema or format"}), 400
    
    schema = data['schema']
    export_format = data['format']
    options = data.get('options', {})
    
    try:
        if export_format == 'sql':
            # Convert schema to SQL DDL statements
            sql_options = options.get('sqlOptions', {})
            code = convert_to_sql(
                schema, 
                dialect=sql_options.get('dialect', 'postgresql'),
                include_indexes=sql_options.get('includeIndexes', True),
                include_timestamps=sql_options.get('includeTimestamps', True)
            )
        elif export_format == 'json':
            # Convert schema to JSON Schema
            json_options = options.get('jsonOptions', {})
            code = convert_to_json_schema(
                schema,
                draft=json_options.get('draft', 'draft-07'),
                include_examples=json_options.get('includeExamples', True)
            )
        else:
            return jsonify({"error": f"Unsupported export format: {export_format}"}), 400
        
        return jsonify({"code": code})
        
    except Exception as e:
        return jsonify({"error": f"Error generating export code: {str(e)}"}), 500

@schema_blueprint.route('/refine-schema', methods=['POST'])
def refine_schema():
    """Refine an existing schema based on user feedback"""
    data = request.json
    
    if not data or 'schema' not in data or 'feedback' not in data:
        return jsonify({"error": "Missing schema or feedback"}), 400
    
    schema = data['schema']
    feedback = data['feedback']
    
    try:
        # Build prompt for Claude
        prompt = f"""
        You are an expert data architect tasked with refining a database schema based on user feedback.
        
        CURRENT SCHEMA:
        ```json
        {json.dumps(schema, indent=2)}
        ```
        
        USER FEEDBACK:
        {feedback}
        
        Please refine the schema according to the user's feedback. Make thoughtful adjustments while 
        maintaining the overall structure and integrity of the schema.
        
        Return only the updated schema as a JSON object with no additional text.
        """
        
        # Call Claude API
        response = client.messages.create(
            model='claude-3-7-sonnet-20250219',
            max_tokens=4000,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and parse the schema from Claude's response
        refined_schema = extract_schema_from_response(response.content)
        
        return jsonify(refined_schema)
        
    except Exception as e:
        return jsonify({"error": f"Error refining schema: {str(e)}"}), 500

@schema_blueprint.route('/validate-schema', methods=['POST'])
def validate_schema():
    """Validate a schema for consistency and completeness"""
    data = request.json
    
    if not data or 'schema' not in data:
        return jsonify({"error": "Missing schema"}), 400
    
    schema = data['schema']
    
    try:
        # Build prompt for Claude
        prompt = f"""
        You are an expert data architect tasked with validating a database schema.
        
        SCHEMA:
        ```json
        {json.dumps(schema, indent=2)}
        ```
        
        Please validate this schema for:
        1. Consistency (e.g., naming conventions, data types)
        2. Completeness (e.g., missing required fields)
        3. Best practices (e.g., proper relationships, indexing needs)
        
        Identify any issues or recommendations for improvement.
        
        Return a JSON object with the following structure:
        {{
          "valid": true/false,
          "issues": [
            {{
              "severity": "error"/"warning"/"info",
              "message": "Description of the issue",
              "location": "Path to the problematic field or area"
            }}
          ],
          "recommendations": [
            {{
              "type": "improvement"/"addition"/"removal",
              "message": "Description of the recommendation",
              "location": "Path to the relevant field or area"
            }}
          ]
        }}
        """
        
        # Call Claude API
        response = client.messages.create(
            model='claude-3-7-sonnet-20250219',
            max_tokens=4000,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and parse the validation results from Claude's response
        validation_results = extract_json_from_response(response.content)
        
        return jsonify(validation_results)
        
    except Exception as e:
        return jsonify({"error": f"Error validating schema: {str(e)}"}), 500

@schema_blueprint.route('/briefing/<project_id>', methods=['GET'])
def get_briefing(project_id):
    """Get the briefing context for a project"""
    try:
        # In a real implementation, this would query a database
        # For now, we'll just read from a static YAML file
        with open('app/data/context.yaml', 'r') as f:
            context = yaml.safe_load(f)
            
        return jsonify(context)
    except Exception as e:
        return jsonify({"error": f"Error loading briefing context: {str(e)}"}), 500

@schema_blueprint.route('/generate-sample-data', methods=['POST'])
def generate_sample_data():
    """Generate sample data based on a schema"""
    data = request.json
    
    if not data or 'schema' not in data:
        return jsonify({"error": "Missing schema"}), 400
    
    schema = data['schema']
    count = data.get('count', 5)
    
    try:
        # Build prompt for Claude
        prompt = f"""
        You are an expert data engineer tasked with generating realistic sample data based on a schema.
        
        SCHEMA:
        ```json
        {json.dumps(schema, indent=2)}
        ```
        
        Please generate {count} records of sample data that conform to this schema.
        Make the data realistic and varied, with appropriate values for each field based on its type and description.
        
        Return the data as a JSON array of objects, with no additional text.
        """
        
        # Call Claude API
        response = client.messages.create(
            model='claude-3-7-sonnet-20250219',
            max_tokens=4000,
            temperature=0.7,  # Higher temperature for more variety
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and parse the sample data from Claude's response
        sample_data = extract_json_from_response(response.content)
        
        return jsonify(sample_data)
        
    except Exception as e:
        return jsonify({"error": f"Error generating sample data: {str(e)}"}), 500

def analyze_text_with_claude(text_content, document_type, extraction_goals):
    """Analyze document text with Claude to identify structure and key information"""
    # Truncate text if too long
    if len(text_content) > 50000:
        text_content = text_content[:50000] + "...[truncated]"
    
    goals_text = "\n".join([f"- {goal}" for goal in extraction_goals])
    
    prompt = f"""
    You are an expert data analyst tasked with analyzing a document to identify its structure and key information.
    
    DOCUMENT TYPE: {document_type}
    
    EXTRACTION GOALS:
    {goals_text if extraction_goals else "No specific goals provided. Identify all key information fields."}
    
    DOCUMENT CONTENT:
    ```
    {text_content}
    ```
    
    Please analyze this document and provide the following:
    1. A list of all key information fields identified in the document
    2. For each field, provide:
       - The field name
       - The data type (string, number, date, boolean, etc.)
       - A brief description of what the field represents
       - Example values from the document (if available)
    3. Any observed patterns or structures in the data
    
    Return your analysis as a JSON object with the following structure:
    {{
      "identified_fields": [
        {{
          "name": "field_name",
          "type": "data_type",
          "description": "description",
          "examples": ["example1", "example2"]
        }}
      ],
      "observed_patterns": [
        "pattern description"
      ],
      "recommended_structure": "brief description of how data should be structured"
    }}
    """
    
    try:
        response = client.messages.create(
            model='claude-3-7-sonnet-20250219',
            max_tokens=4000,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and parse the JSON from Claude's response
        return extract_json_from_response(response.content)
    except Exception as e:
        print(f"Error analyzing text with Claude: {str(e)}")
        return {
            "error": f"Failed to analyze document: {str(e)}",
            "identified_fields": [],
            "observed_patterns": [],
            "recommended_structure": "Unable to determine"
        }

def analyze_json_structure(json_data):
    """Analyze the structure of a JSON object"""
    result = {
        "type": type(json_data).__name__
    }
    
    if isinstance(json_data, dict):
        result["fields"] = {}
        for key, value in json_data.items():
            result["fields"][key] = {
                "type": type(value).__name__,
                "sample": str(value)[:100] if value is not None else None
            }
            if isinstance(value, (dict, list)) and value:
                result["fields"][key]["nested_structure"] = analyze_json_structure(value)
    
    elif isinstance(json_data, list) and json_data:
        result["length"] = len(json_data)
        result["sample_item"] = analyze_json_structure(json_data[0])
    
    return result

def build_schema_generation_prompt(analysis_results, context):
    """Build a prompt for schema generation based on analysis results"""
    document_type = context.get('documentType', 'unknown')
    extraction_goals = context.get('extractionGoals', [])
    user_notes = context.get('userNotes', '')
    
    goals_text = "\n".join([f"- {goal}" for goal in extraction_goals])
    
    # Extract briefing context if available
    briefing_context = analysis_results.get('briefing_context', {})
    business_context = briefing_context.get('business_context', '') if briefing_context else ''
    
    # Prepare file analyses
    file_analyses = []
    for analysis in analysis_results.get('file_analyses', []):
        file_type = analysis.get('file_type', '')
        filename = analysis.get('filename', '')
        
        if file_type == 'document':
            file_analyses.append(f"""
            Document: {filename}
            ```
            {json.dumps(analysis.get('analysis', {}), indent=2)}
            ```
            """)
        elif file_type == 'structured_data':
            if 'columns' in analysis:
                # CSV analysis
                columns_info = "\n".join([
                    f"- {col} ({analysis['data_types'][col]})" 
                    for col in analysis.get('columns', [])
                ])
                
                sample_data = json.dumps(analysis.get('sample_data', []), indent=2)
                
                file_analyses.append(f"""
                CSV File: {filename}
                Columns:
                {columns_info}
                
                Sample Data:
                ```
                {sample_data}
                ```
                """)
            elif 'sheets' in analysis:
                # Excel analysis
                sheets_info = []
                for sheet_name, sheet_data in analysis.get('sheets', {}).items():
                    columns_info = "\n".join([
                        f"- {col} ({sheet_data['data_types'][col]})" 
                        for col in sheet_data.get('columns', [])
                    ])
                    
                    sample_data = json.dumps(sheet_data.get('sample_data', []), indent=2)
                    
                    sheets_info.append(f"""
                    Sheet: {sheet_name}
                    Columns:
                    {columns_info}
                    
                    Sample Data:
                    ```
                    {sample_data}
                    ```
                    """)
                
                file_analyses.append(f"""
                Excel File: {filename}
                {"\n".join(sheets_info)}
                """)
            elif 'structure' in analysis:
                # JSON analysis
                file_analyses.append(f"""
                JSON File: {filename}
                Structure:
                ```
                {json.dumps(analysis.get('structure', {}), indent=2)}
                ```
                """)
    
    prompt = f"""
    You are an expert data architect tasked with creating an optimal database schema for a document extraction system.
    
    DOCUMENT TYPE: {document_type}
    
    EXTRACTION GOALS:
    {goals_text if extraction_goals else "No specific goals provided."}
    
    USER NOTES:
    {user_notes}
    
    BUSINESS CONTEXT:
    {business_context}
    
    FILE ANALYSES:
    {"\n".join(file_analyses)}
    
    Based on the analysis of these documents, create a comprehensive schema that captures all relevant information.
    The schema should be optimized for the document type and extraction goals specified above.
    
    Your schema should follow these guidelines:
    1. Use snake_case for field names
    2. Choose appropriate data types (string, number, boolean, date, object, array, etc.)
    3. Include descriptions for each field
    4. Mark required fields appropriately
    5. Use nested objects where it makes logical sense
    6. Consider relationships between different entities
    
    Return a JSON object with the following structure:
    {{
      "name": "schema_name",
      "description": "Overall schema description",
      "fields": [
        {{
          "name": "field_name",
          "type": "data_type",
          "description": "Field description",
          "required": true/false,
          "format": "optional format specification",
          "fields": [] // For nested objects
        }}
      ]
    }}
    
    Return only the JSON schema object with no additional text.
    """
    
    return prompt

def extract_schema_from_response(response):
    """Extract and parse schema JSON from Claude's response"""
    return extract_json_from_response(response)

def extract_json_from_response(response):
    """Extract and parse JSON from Claude's response text"""
    # If response is already a dict (happens with Anthropic API sometimes)
    if isinstance(response, dict):
        return response
        
    # Handle string response
    try:
        # Look for JSON code block
        import re
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, response)
        
        if match:
            json_str = match.group(1).strip()
            return json.loads(json_str)
        
        # If no JSON code block, try to find JSON in the raw text
        # Find the first { and the last }
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx+1].strip()
            return json.loads(json_str)
        
        # If we can't find JSON, return the whole response as an error
        return {"error": "Could not extract JSON from response", "raw_response": response}
    
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON: {str(e)}", "raw_response": response}
    except Exception as e:
        return {"error": f"Error extracting JSON: {str(e)}", "raw_response": response}

@schema_blueprint.route('/save-schema/<project_id>', methods=['POST'])
def save_schema_endpoint(project_id):
    """Save schema for a specific project"""
    try:
        schema_data = request.json
        if not schema_data:
            return jsonify({"error": "No schema data provided"}), 400
            
        success = save_schema(project_id, schema_data)
        if success:
            return jsonify({"message": "Schema saved successfully"}), 200
        else:
            return jsonify({"error": "Failed to save schema"}), 500
    except Exception as e:
        return jsonify({"error": f"Error saving schema: {str(e)}"}), 500
    
    
    
    
@schema_blueprint.route('/schema/<project_id>', methods=['GET'])
def get_schema(project_id):
    """Retrieve schema for a specific project"""
    try:
        is_development = current_app.config.get('FLASK_ENV') == 'development'
        
        if is_development and os.environ.get('USE_LOCAL_FILES', 'False').lower() == 'true':
            schema_path = os.path.abspath(os.path.join(
                current_app.root_path, '..', '..', 'static', 'data',
                'schemas', project_id, 'schema.json'
            ))
            try:
                with open(schema_path, 'r') as f:
                    schema_data = json.load(f)
            except FileNotFoundError:
                return jsonify({"error": "Schema not found"}), 404
        else:
            # S3 path for production
            try:
                response = s3_client.get_object(
                    Bucket='juniperproductiondata',
                    Key=f'data/schemas/{project_id}/schema.json'
                )
                schema_data = json.loads(response['Body'].read().decode('utf-8'))
            except s3_client.exceptions.NoSuchKey:
                return jsonify({"error": "Schema not found"}), 404
                
        return jsonify(schema_data)
    except Exception as e:
        current_app.logger.error(f"Error retrieving schema: {str(e)}")
        return jsonify({"error": f"Error retrieving schema: {str(e)}"}), 500 
