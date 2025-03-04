# app/schema_blueprint.py
from flask import Blueprint, request, jsonify, current_app
import json
import yaml
import os
import tempfile
import boto3
import re
import copy
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
        # Debug prints
        flask_env = current_app.config.get('FLASK_ENV') or os.environ.get ('FLASK_ENV')
        print(flask_env)
        use_local_files = os.environ.get('USE_LOCAL_FILES', 'False')
        
        print(f"FLASK_ENV: {flask_env}")
        print(f"USE_LOCAL_FILES: {use_local_files}")
        
        is_development = flask_env == 'development'
        should_use_local = use_local_files.lower() == 'true'
        
        print(f"is_development: {is_development}")
        print(f"should_use_local: {should_use_local}")
        
        # Force local file saving for testing
        if True:  # Temporarily force local saving
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
            print(f"Created directory: {schema_dir}")
            
            schema_path = os.path.join(schema_dir, 'schema.json')
            with open(schema_path, 'w') as f:
                json.dump(schema_data, f, indent=2)
            print(f"Schema saved locally to: {schema_path}")
            return True
        elif is_development and should_use_local:
            # Original local file logic...
            # (keep this as a fallback)
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
            print(f"Created directory: {schema_dir}")
            
            schema_path = os.path.join(schema_dir, 'schema.json')
            with open(schema_path, 'w') as f:
                json.dump(schema_data, f, indent=2)
            print(f"Schema saved locally to: {schema_path}")
            return True
        else:
            # Save to S3 in production
            print("Would save to S3 in production, but skipping for now")
            # Comment out actual S3 code for safety during testing
            # s3_client = boto3.client('s3')
            # s3_client.put_object(
            #     Bucket='juniperproductiondata',
            #     Key=f'data/schemas/{project_id}/schema.json',
            #     Body=json.dumps(schema_data, indent=2)
            # )
            return True
    except Exception as e:
        current_app.logger.error(f"Error saving schema: {str(e)}")
        print(f"Error saving schema: {str(e)}")
        return False


















@schema_blueprint.route('/generate-schema', methods=['POST'])
def generate_schema():
    """Generate a schema proposal based on document analysis"""
    try:
        print("\n====== GENERATE SCHEMA ENDPOINT CALLED ======")
        
        # Print request information
        print(f"Request Method: {request.method}")
        print(f"Content Type: {request.content_type}")
        
        data = request.json
        if not data or 'analysisResults' not in data:
            print("ERROR: Missing analysis results")
            return jsonify({"error": "Missing analysis results"}), 400
        
        # Print analysis results summary
        analysis_results = data['analysisResults']
        print(f"Analysis Results: Document Type: {analysis_results.get('document_type')}")
        print(f"Number of file analyses: {len(analysis_results.get('file_analyses', []))}")
        
        # Get context and other settings
        context = data.get('context', {})
        model_settings = data.get('modelSettings', {})
        
        print(f"Context: {json.dumps(context, indent=2)}")
        print(f"Model Settings: {json.dumps(model_settings, indent=2)}")
        
        
        
        project_id = "3MAcquisition"  # Default project ID
        
        # Check for briefing context
        
        
        
        briefing_context = analysis_results.get('briefing_context') or context.get('briefingContext')
        if briefing_context:
            
            
            if 'project_id' in briefing_context:
                project_id = briefing_context.get('project_id') 
            print(f"\nBriefing Context Available:")
            
            if 'context' in briefing_context:
                project_info = briefing_context.get('context', {}).get('project', {})
                print(f"  Project Name: {project_info.get('name', 'Not provided')}")
                print(f"  Project Type: {project_info.get('type', 'Not provided')}")
                business_context = briefing_context.get('context', {}).get('business_context', '')
                print(f"  Business Context: {business_context[:100]}..." if business_context else "  No Business Context")
            
            # Include information from scope if available
            scope = briefing_context.get('scope', '')
            if scope:
                print(f"  Scope available ({len(scope)} characters)")
            else:
                print("  No scope information")
        else:
            print("No briefing context available")
        
        # For M&A Due Diligence, create a more detailed schema
        if context.get('documentType') == 'ma_due_diligence':
            schema = {
                "name": "ma_due_diligence_schema",
                "description": "Schema for M&A due diligence document extraction",
                "fields": [
                    {
                        "name": "document_id",
                        "type": "string",
                        "description": "Unique identifier for the document",
                        "required": True
                    },
                    {
                        "name": "document_metadata",
                        "type": "object",
                        "description": "Metadata about the document",
                        "required": True,
                        "fields": [
                            {
                                "name": "title",
                                "type": "string",
                                "description": "Document title",
                                "required": True
                            },
                            {
                                "name": "date",
                                "type": "date",
                                "description": "Document date",
                                "required": False,
                                "format": "YYYY-MM-DD"
                            },
                            {
                                "name": "document_type",
                                "type": "string",
                                "description": "Type of document (e.g., financial statement, contract)",
                                "required": True
                            },
                            {
                                "name": "source",
                                "type": "string",
                                "description": "Source of the document",
                                "required": False
                            }
                        ]
                    },
                    {
                        "name": "financial_data",
                        "type": "object",
                        "description": "Financial information extracted from the document",
                        "required": False,
                        "fields": [
                            {
                                "name": "revenue",
                                "type": "number",
                                "description": "Annual revenue figure",
                                "required": False
                            },
                            {
                                "name": "profit",
                                "type": "number",
                                "description": "Annual profit/loss figure",
                                "required": False
                            },
                            {
                                "name": "assets",
                                "type": "number",
                                "description": "Total assets value",
                                "required": False
                            },
                            {
                                "name": "liabilities",
                                "type": "number",
                                "description": "Total liabilities value",
                                "required": False
                            },
                            {
                                "name": "period",
                                "type": "string",
                                "description": "Financial reporting period",
                                "required": False
                            }
                        ]
                    },
                    {
                        "name": "legal_obligations",
                        "type": "array",
                        "description": "Legal obligations or commitments identified",
                        "required": False,
                        "items": {
                            "type": "object",
                            "fields": [
                                {
                                    "name": "obligation_type",
                                    "type": "string",
                                    "description": "Type of obligation (e.g., contract, regulatory)",
                                    "required": True
                                },
                                {
                                    "name": "description",
                                    "type": "string",
                                    "description": "Description of the obligation",
                                    "required": True
                                },
                                {
                                    "name": "parties",
                                    "type": "array",
                                    "description": "Parties involved in the obligation",
                                    "required": False,
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                {
                                    "name": "value",
                                    "type": "number",
                                    "description": "Monetary value of the obligation, if applicable",
                                    "required": False
                                },
                                {
                                    "name": "expiration_date",
                                    "type": "date",
                                    "description": "Expiration or due date of the obligation",
                                    "required": False,
                                    "format": "YYYY-MM-DD"
                                }
                            ]
                        }
                    },
                    {
                        "name": "risks",
                        "type": "array",
                        "description": "Risks identified in the document",
                        "required": False,
                        "items": {
                            "type": "object",
                            "fields": [
                                {
                                    "name": "risk_type",
                                    "type": "string",
                                    "description": "Type of risk (e.g., financial, operational, legal)",
                                    "required": True
                                },
                                {
                                    "name": "description",
                                    "type": "string",
                                    "description": "Description of the risk",
                                    "required": True
                                },
                                {
                                    "name": "severity",
                                    "type": "string",
                                    "description": "Severity of the risk (low, medium, high)",
                                    "required": False
                                },
                                {
                                    "name": "potential_impact",
                                    "type": "string",
                                    "description": "Potential impact of the risk",
                                    "required": False
                                },
                                {
                                    "name": "mitigation_measures",
                                    "type": "array",
                                    "description": "Measures to mitigate the risk",
                                    "required": False,
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            ]
                        }
                    },
                    {
                        "name": "intellectual_property",
                        "type": "array",
                        "description": "Intellectual property assets mentioned",
                        "required": False,
                        "items": {
                            "type": "object",
                            "fields": [
                                {
                                    "name": "ip_type",
                                    "type": "string",
                                    "description": "Type of IP (patent, trademark, copyright)",
                                    "required": True
                                },
                                {
                                    "name": "name",
                                    "type": "string", 
                                    "description": "Name or title of the IP asset",
                                    "required": True
                                },
                                {
                                    "name": "registration_number",
                                    "type": "string",
                                    "description": "Registration or application number",
                                    "required": False
                                },
                                {
                                    "name": "jurisdiction",
                                    "type": "string",
                                    "description": "Jurisdiction where the IP is registered",
                                    "required": False
                                },
                                {
                                    "name": "expiration_date",
                                    "type": "date",
                                    "description": "Expiration date of the IP protection",
                                    "required": False,
                                    "format": "YYYY-MM-DD"
                                }
                            ]
                        }
                    }
                ]
            }
            
            # If we have a specific client/acquisition from briefing context, add to schema name
            if briefing_context and 'context' in briefing_context:
                project_info = briefing_context.get('context', {}).get('project', {})
                client = project_info.get('client', '')
                if client:
                    schema['description'] = f"Schema for {client} due diligence document extraction"
        else:
            # Generic schema for other document types
            schema = {
                "name": f"{context.get('documentType', 'document')}_schema",
                "description": f"Schema for {context.get('documentType', 'document')} data extraction",
                "fields": [
                    {
                        "name": "document_id",
                        "type": "string",
                        "description": "Unique identifier for the document",
                        "required": True
                    },
                    {
                        "name": "title", 
                        "type": "string",
                        "description": "Document title",
                        "required": True
                    },
                    {
                        "name": "date",
                        "type": "date",
                        "description": "Document date",
                        "required": False
                    },
                    {
                        "name": "content_summary",
                        "type": "string",
                        "description": "Summary of document content",
                        "required": False
                    },
                    {
                        "name": "key_entities",
                        "type": "array",
                        "description": "Key entities mentioned in the document",
                        "required": False,
                        "items": {
                            "type": "object",
                            "fields": [
                                {
                                    "name": "entity_name",
                                    "type": "string",
                                    "description": "Name of the entity",
                                    "required": True
                                },
                                {
                                    "name": "entity_type",
                                    "type": "string",
                                    "description": "Type of entity (person, organization, location)",
                                    "required": False
                                }
                            ]
                        }
                    }
                ]
            }
        
# Now that schema is defined, print and save it
        print("\nGenerated Schema:")
        print(json.dumps(schema, indent=2))
        
        # Save the generated schema
        save_success = save_schema(project_id, schema)
        if save_success:
            print(f"Schema successfully saved to project: {project_id}")
        else:
            print(f"Warning: Failed to save schema to project: {project_id}")
        
        print("Sending response schema:")
        print(f"Schema Name: {schema['name']}")
        print(f"Number of Fields: {len(schema['fields'])}")
        print("====== GENERATE SCHEMA ENDPOINT COMPLETE ======\n")
        
        return jsonify(schema)
        
    except Exception as e:
        import traceback
        print("\n====== ERROR IN GENERATE SCHEMA ENDPOINT ======")
        print(f"Exception: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("================================================\n")
        
        return jsonify({"error": f"Error generating schema: {str(e)}"}), 500
    
    
























@schema_blueprint.route('/briefing/<project_id>', methods=['GET'])
def get_briefing(project_id):
    """Get the briefing context for a project including both context.yaml and scope.txt"""
    try:
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
        
        # For debugging
        current_app.logger.info(f"Attempting to read from context path: {context_yaml_path}")
        current_app.logger.info(f"Attempting to read from scope path: {scope_txt_path}")
        
        # Initialize response object
        briefing_data = {}
        
        # Read context.yaml file
        try:
            with open(context_yaml_path, 'r') as f:
                yaml_context = yaml.safe_load(f)
                briefing_data["context"] = yaml_context
        except Exception as e:
            current_app.logger.warning(f"Error reading context.yaml: {str(e)}")
            briefing_data["context"] = None
        
        # Read scope.txt file
        try:
            with open(scope_txt_path, 'r') as f:
                scope_text = f.read()
                briefing_data["scope"] = scope_text
        except Exception as e:
            current_app.logger.warning(f"Error reading scope.txt: {str(e)}")
            briefing_data["scope"] = None
        
        # Add project ID to the response
        briefing_data["project_id"] = project_id
        print(briefing_data)
        return jsonify(briefing_data)
    except Exception as e:
        current_app.logger.error(f"Error loading briefing context: {str(e)}")
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
        response = anthropic_client.messages.create(
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






@schema_blueprint.route('/analyze-schema-documents', methods=['POST'])
def analyze_documents():
    """Simplified version with print statements for debugging"""
    try:
        print("\n====== ANALYZE DOCUMENTS ENDPOINT CALLED ======")
        
        # Print request information
        print(f"Request Method: {request.method}")
        print(f"Content Type: {request.content_type}")
        print(f"Request Form Keys: {list(request.form.keys())}")
        print(f"Request Files Keys: {list(request.files.keys())}")
        
        if 'context' not in request.form:
            print("ERROR: Missing context information")
            return jsonify({"error": "Missing context information"}), 400
        
        # Print context information
        context = json.loads(request.form['context'])
        print(f"Context: {json.dumps(context, indent=2)}")
        
        document_type = context.get('documentType', 'unknown')
        extraction_goals = context.get('extractionGoals', [])
        briefing_context = context.get('briefingContext')
        
        print(f"Document Type: {document_type}")
        print(f"Extraction Goals: {extraction_goals}")
        
        # Print briefing context if available
        if briefing_context:
            print(f"\nBriefing Context Received:")
            print(f"  Project ID: {briefing_context.get('project_id', 'Not provided')}")
            if 'context' in briefing_context:
                project_info = briefing_context.get('context', {}).get('project', {})
                print(f"  Project Name: {project_info.get('name', 'Not provided')}")
                print(f"  Project Type: {project_info.get('type', 'Not provided')}")
                print(f"  Business Context Available: {'Yes' if briefing_context.get('context', {}).get('business_context') else 'No'}")
            print(f"  Scope Text Available: {'Yes' if briefing_context.get('scope') else 'No'}")
        else:
            print("No briefing context provided")
        
        # Print info about each file
        print("\nFiles received:")
        file_analyses = []
        for key in request.files:
            if key.startswith('file_'):
                file = request.files[key]
                filename = secure_filename(file.filename)
                file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
                
                print(f"  - File: {filename}, Type: {file_extension}, Size: {len(file.read())} bytes")
                file.seek(0)  # Reset file pointer after reading
                
                # Create basic analysis based on file type
                if file_extension in ['csv', 'xlsx', 'xls']:
                    file_type = "structured_data"
                    file_analyses.append({
                        "file_type": file_type,
                        "filename": filename,
                        "columns": ["Sample Column 1", "Sample Column 2"],
                        "sample_data": [{"Sample Column 1": "Value 1", "Sample Column 2": "Value 2"}]
                    })
                else:
                    file_type = "document"
                    file_analyses.append({
                        "file_type": file_type,
                        "filename": filename,
                        "analysis": {
                            "identified_fields": [
                                {"name": "sample_field", "type": "string", "description": "Sample field"}
                            ]
                        }
                    })
        
        # Create response
        response_data = {
            "document_type": document_type,
            "extraction_goals": extraction_goals,
            "file_analyses": file_analyses,
            "briefing_context": briefing_context
        }
        
        print("\nSending response data:")
        print(f"Number of file analyses: {len(file_analyses)}")
        print(f"Including briefing context: {'Yes' if briefing_context else 'No'}")
        print("====== ANALYZE DOCUMENTS ENDPOINT COMPLETE ======\n")
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print("\n====== ERROR IN ANALYZE DOCUMENTS ENDPOINT ======")
        print(f"Exception: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("================================================\n")
        
        current_app.logger.error(f"Error in analyze_documents: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Error processing files: {str(e)}"}), 500
    
    
@schema_blueprint.route('/suggest-schema-enhancements', methods=['POST'])
def suggest_schema_enhancements():
    """
    Analyze documents and briefing context to suggest schema enhancements for star schema
    The response format is designed to work with the DimensionEnhancementPanel component
    """
    try:
        print("\n====== SUGGEST SCHEMA ENHANCEMENTS ENDPOINT CALLED ======")
        
        # Print request information
        print(f"Request Method: {request.method}")
        print(f"Content Type: {request.content_type}")
        print(f"Request Form Keys: {list(request.form.keys())}")
        print(f"Request Files Keys: {list(request.files.keys())}")
        
        if 'context' not in request.form:
            print("ERROR: Missing context information")
            return jsonify({"error": "Missing context information"}), 400
        
        # Parse context information
        context = json.loads(request.form['context'])
        print(f"Context: {json.dumps(context, indent=2)}")
        
        document_type = context.get('documentType', 'unknown')
        project_name = context.get('projectName', 'Unnamed Project')
        briefing_context = context.get('briefingContext')
        
        print(f"Document Type: {document_type}")
        print(f"Project Name: {project_name}")
        
        # Determine which star schema to use based on document type
        # For now hardcoded to M&A due diligence
        constellation_type = "ma-due-diligence"
        
        # Read the current dimensions and attributes JSON files
        # We'll need to strip comments since JSON doesn't officially support them
        dimensions_path = os.path.abspath(os.path.join(
            current_app.root_path,  # This is backend/app
            '..',                   # Up to backend
            '..',                   # Up to transaction_platform_app
            'static',
            'ArchitecturalRegistries',
            'GalaxyParentClass',
            'Constellations',
            'AcquisitionProjects',
            'dimensions.json'
        ))
        
        attributes_path = os.path.abspath(os.path.join(
            current_app.root_path,  # This is backend/app
            '..',                   # Up to backend
            '..',                   # Up to transaction_platform_app
            'static',
            'ArchitecturalRegistries',
            'GalaxyParentClass',
            'Constellations',
            'AcquisitionProjects',
            'attributes.json'
        ))
        
        # Read the enhancement prompt
        prompt_path = os.path.abspath(os.path.join(
            current_app.root_path,  # This is backend/app
            '..',                   # Up to backend
            '..',                   # Up to transaction_platform_app
            'static',
            'data',
            'prompts',
            'vectorpromptpackage.txt'
        ))
        
        # Load and clean dimensions JSON (remove comments)
        try:
            with open(dimensions_path, 'r') as f:
                dimensions_text = f.read()
                # Remove JavaScript-style comments
                dimensions_text = re.sub(r'//.*?\n', '\n', dimensions_text)
                dimensions_data = json.loads(dimensions_text)
                print(f"Successfully loaded dimensions schema with {len(dimensions_data.get('dimensions', []))} dimensions")
        except Exception as e:
            print(f"Error loading dimensions JSON: {str(e)}")
            dimensions_data = {"dimensions": [], "parentDimensions": []}
        
        # Load and clean attributes JSON (remove comments)
        try:
            with open(attributes_path, 'r') as f:
                attributes_text = f.read()
                # Remove JavaScript-style comments
                attributes_text = re.sub(r'//.*?\n', '\n', attributes_text)
                attributes_data = json.loads(attributes_text)
                print(f"Successfully loaded attributes schema")
                
                # Print first few dimensions for which we have attributes
                sample_dims = list(attributes_data.keys())[:3]
                print(f"Attributes available for dimensions: {', '.join(sample_dims)}...")
        except Exception as e:
            print(f"Error loading attributes JSON: {str(e)}")
            attributes_data = {}
        
        # Read the enhancement prompt template
        try:
            with open(prompt_path, 'r') as f:
                prompt_template = f.read()
        except Exception as e:
            print(f"Error reading prompt file: {str(e)}")
            # Fallback prompt if file not found
            prompt_template = """
            You are an expert in data modeling and star schema design for M&A due diligence.
            
            Based on the uploaded documents and the current schema definition, suggest enhancements
            to the star schema. You should only suggest new dimensions or attributes to add - do not 
            suggest removing anything from the existing schema.
            
            Current dimensions: {dimensions_json}
            
            Current attributes: {attributes_json}
            
            Briefing context: {briefing_context}
            
            Return your recommendations in JSON format that can be used with the DimensionEnhancementPanel component.
            The format should be a list of enhancement objects with this structure:
            
            [
              {
                "name": "Dimension Name",
                "attributeGroups": [
                  {
                    "name": "Group Name", 
                    "count": 5, // Number of attributes in this group
                    "attributes": [
                      {
                        "id": "attribute_id",
                        "displayName": "Attribute Display Name",
                        "dataType": "text",
                        "required": true,
                        "maxLength": 2000,
                        "order": 1
                      },
                      // ... more attributes
                    ]
                  }
                ]
              }
            ]
            
            Remember, your suggestions should be in addition to the existing schema, not replacing it.
            """
            
        # Parse uploaded documents
        uploaded_documents_content = []
        for key in request.files:
            if key.startswith('file_'):
                file = request.files[key]
                filename = secure_filename(file.filename)
                content = file.read().decode('utf-8', errors='ignore')
                uploaded_documents_content.append({
                    "filename": filename,
                    "content": content[:10000]  # Limit content size for the prompt
                })
        
        # Build the prompt for Claude
        prompt = prompt_template.format(
            dimensions_json=json.dumps(dimensions_data, indent=2),
            attributes_json=json.dumps(attributes_data, indent=2),
            briefing_context=json.dumps(briefing_context, indent=2) if briefing_context else "Not provided",
            uploaded_documents=json.dumps(uploaded_documents_content, indent=2)
        )
        
        print(f"Calling Claude API to get schema enhancement suggestions")
        
        # Call Claude API
        response = anthropic_client.messages.create(
            model='claude-3-7-sonnet-20250219',
            max_tokens=4000,
            temperature=0.2,  # Lower temperature for more precise/predictable responses
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse Claude's response to get the suggested enhancements
        suggested_enhancements = extract_json_from_response(response.content)
        
        # Prepare current dimensions data for the frontend
        current_dimensions = []
        for dimension in dimensions_data.get('dimensions', []):
            attributes_count = len(attributes_data.get(dimension['id'], []))
            current_dimensions.append({
                "name": dimension['displayName'],
                "id": dimension['id'],
                "attributeCount": attributes_count
            })
        
        # Format the response for the DimensionEnhancementPanel
        response_data = {
            "projectName": project_name,
            "currentDimensions": current_dimensions,
            "suggestedEnhancements": suggested_enhancements
        }
        
        print("\nSending response data:")
        print(f"Current dimensions count: {len(current_dimensions)}")
        print(f"Suggested enhancements count: {len(suggested_enhancements)}")
        print("====== SUGGEST SCHEMA ENHANCEMENTS ENDPOINT COMPLETE ======\n")
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print("\n====== ERROR IN SUGGEST SCHEMA ENHANCEMENTS ENDPOINT ======")
        print(f"Exception: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("================================================\n")
        
        return jsonify({"error": f"Error suggesting schema enhancements: {str(e)}"}), 500


@schema_blueprint.route('/apply-schema-enhancements', methods=['POST'])
def apply_schema_enhancements():
    """
    Apply selected schema enhancements to create a new version of the schema
    The new schema is a deep copy of the original with the selected enhancements applied
    """
    try:
        print("\n====== APPLY SCHEMA ENHANCEMENTS ENDPOINT CALLED ======")
        
        data = request.json
        if not data or 'selectedEnhancements' not in data or 'projectId' not in data:
            print("ERROR: Missing selected enhancements or project ID")
            return jsonify({"error": "Missing required data"}), 400
        
        selected_enhancements = data['selectedEnhancements']
        project_id = data['projectId']
        
        print(f"Project ID: {project_id}")
        print(f"Selected enhancements count: {len(selected_enhancements)}")
        
        # Read the current dimensions and attributes JSON files
        dimensions_path = os.path.abspath(os.path.join(
            current_app.root_path,  # This is backend/app
            '..',                   # Up to backend
            '..',                   # Up to transaction_platform_app
            'static',
            'ArchitecturalRegistries',
            'GalaxyParentClass',
            'Constellations',
            'AcquisitionProjects',
            'dimensions.json'
        ))
        
        attributes_path = os.path.abspath(os.path.join(
            current_app.root_path,  # This is backend/app
            '..',                   # Up to backend
            '..',                   # Up to transaction_platform_app
            'static',
            'ArchitecturalRegistries',
            'GalaxyParentClass',
            'Constellations',
            'AcquisitionProjects',
            'attributes.json'
        ))
        
        # Load and clean dimensions JSON
        with open(dimensions_path, 'r') as f:
            dimensions_text = f.read()
            # Remove JavaScript-style comments
            dimensions_text = re.sub(r'//.*?\n', '\n', dimensions_text)
            dimensions_data = json.loads(dimensions_text)
        
        # Load and clean attributes JSON
        with open(attributes_path, 'r') as f:
            attributes_text = f.read()
            # Remove JavaScript-style comments
            attributes_text = re.sub(r'//.*?\n', '\n', attributes_text)
            attributes_data = json.loads(attributes_text)
        
        # Create deep copies to avoid modifying originals
        new_dimensions = copy.deepcopy(dimensions_data)
        new_attributes = copy.deepcopy(attributes_data)
        
        # Get the highest order value for existing dimensions
        max_order = 0
        for dimension in dimensions_data.get('dimensions', []):
            if dimension['order'] > max_order:
                max_order = dimension['order']
        
        # Apply the selected enhancements
        for enhancement in selected_enhancements:
            # Create a new dimension entry
            dimension_name = enhancement['name']
            dimension_id = re.sub(r'[^a-z0-9_]', '_', dimension_name.lower())
            
            # Check if dimension ID already exists
            existing_ids = [dim['id'] for dim in new_dimensions['dimensions']]
            if dimension_id in existing_ids:
                # Generate a unique ID by adding a suffix
                base_id = dimension_id
                suffix = 1
                while f"{base_id}_{suffix}" in existing_ids:
                    suffix += 1
                dimension_id = f"{base_id}_{suffix}"
            
            max_order += 1
            
            # Add the new dimension to dimensions.json
            new_dimension = {
                "id": dimension_id,
                "displayName": dimension_name,
                "tableName": f"{dimension_id}_dimension",
                "foreignKey": "target_id",
                "requiredForNewStar": True,
                "order": max_order
            }
            new_dimensions['dimensions'].append(new_dimension)
            
            # Add attributes to attributes.json
            all_attributes = []
            for group in enhancement.get('attributeGroups', []):
                all_attributes.extend(group.get('attributes', []))
            
            # Validate and update attribute data
            for i, attr in enumerate(all_attributes):
                # Make sure each attribute has required fields
                if 'id' not in attr:
                    attr['id'] = re.sub(r'[^a-z0-9_]', '_', attr['displayName'].lower())
                
                # Set default values for missing fields
                attr.setdefault('required', True)
                attr.setdefault('order', i + 1)
                if attr.get('dataType') == 'text':
                    attr.setdefault('maxLength', 2000)
            
            new_attributes[dimension_id] = all_attributes
        
        # Create directory for project-specific schema
        project_schema_dir = os.path.abspath(os.path.join(
            current_app.root_path,  # This is backend/app
            '..',                   # Up to backend
            '..',                   # Up to transaction_platform_app
            'static',
            'data',
            'schemas',
            project_id
        ))
        os.makedirs(project_schema_dir, exist_ok=True)
        
        # Save the enhanced dimensions.json
        enhanced_dimensions_path = os.path.join(project_schema_dir, 'dimensions.json')
        with open(enhanced_dimensions_path, 'w') as f:
            json.dump(new_dimensions, f, indent=2)
        
        # Save the enhanced attributes.json
        enhanced_attributes_path = os.path.join(project_schema_dir, 'attributes.json')
        with open(enhanced_attributes_path, 'w') as f:
            json.dump(new_attributes, f, indent=2)
        
        print(f"Enhanced schema saved to: {project_schema_dir}")
        print(f"New dimensions count: {len(new_dimensions['dimensions'])}")
        print(f"New attributes dimensions: {len(new_attributes.keys())}")
        
        # Return success response with counts
        response_data = {
            "success": True,
            "projectId": project_id,
            "originalDimensionsCount": len(dimensions_data['dimensions']),
            "newDimensionsCount": len(new_dimensions['dimensions']),
            "enhancedDimensionsCount": len(new_dimensions['dimensions']) - len(dimensions_data['dimensions']),
            "schemaLocation": project_schema_dir
        }
        
        print("====== APPLY SCHEMA ENHANCEMENTS ENDPOINT COMPLETE ======\n")
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print("\n====== ERROR IN APPLY SCHEMA ENHANCEMENTS ENDPOINT ======")
        print(f"Exception: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("================================================\n")
        
        return jsonify({"error": f"Error applying schema enhancements: {str(e)}"}), 500
