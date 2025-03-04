from flask import Blueprint, render_template, send_from_directory, g, request, session, send_file, abort, json, jsonify, current_app

import json

import boto3

from anthropic import Anthropic  # Now this will work after installation
from docxtpl import DocxTemplate  # Add this import at the top of your file
from io import BytesIO  # Add this at the top with other imports









import logging

import requests


import base64

import mammoth


import pythoncom
import win32com.client as win32
import os

import traceback

import re

from docx2pdf import convert

from pymongo import MongoClient
from pymongo.errors import CollectionInvalid

import csv
import io
from bson import json_util
# Add these imports at the top of your main_blueprint.py file


from threading import Thread




from bson import ObjectId


import time
import copy

import datetime
from pathlib import Path
#from ..auth.auth_engine import login_required  # Add this import

from botocore.exceptions import ClientError

from ..utilities.json_accessor import JsonAccessor
from ..utilities.classifier import Classifier
from .. utilities.tooljet_parser import TooljetParser
from ..utilities import template_converter
from .. utilities.templatemaker import DocumentProcessor

# Get AWS credentials from environment variables
AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.environ.get('S3_BUCKET')


# Initialize Anthropic client
anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))



# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)
client = MongoClient('mongodb://localhost:27017/')
db = client.litigationDatabase

company_matcher = Classifier()



main_blueprint = Blueprint('main', __name__)

# Base path configuration
APP_BASE = 'app/application_groups'

# Asset types configuration
ASSET_TYPES = ['js', 'css', 'data', 'handlers']

# Shared components configuration
SHARED_COMPONENTS = {
    'prototype': ['js', 'css'],
    'table': ['data', 'css', 'js'],
    'dashboardreporting': ['data', 'css', 'js'],
    'shared': ['css', 'js', 'handlers']
}

# =============================================================================
# Template Processing Constants and Configurations
# =============================================================================

# Template Processing Constants and Configurations
# =============================================================================

# Template processing configurations
TEMPLATE_CONFIG = {
    'ALLOWED_EXTENSIONS': {'docx'},
    'MAX_TEMPLATE_SIZE': 10 * 1024 * 1024  # 10MB max template size
}



def get_template_paths():
    """Get template paths using current_app context"""
    base_path = os.path.join(
        current_app.root_path,
        '..',
        '..',
        'static',
        'data',
        'SystemITTemplates',
        'TangibleITTemplates'
    )
    
    return {
        'BASE_PATH': base_path,
        'PARENT_TEMPLATES': os.path.join(base_path, 'Foundational', 'Parents'),
        'ORDER_TEMPLATES': os.path.join(base_path, 'Foundational', 'Orders')
    }


def load_runs_data():
    try:
        is_development = current_app.config.get('FLASK_ENV') == 'development'
        
        if is_development and os.environ.get('USE_LOCAL_FILES', 'False').lower() == 'true':
            json_path = os.path.abspath(os.path.join(
                current_app.root_path,
                '..',
                '..',
                'static',
                'data',
                'applicationRuns',
                'run_state.json'
            ))
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                return {"runs": {"MERGER_CONTROL": {}}}
        else:
            # S3 path for production
            s3_client = boto3.client('s3')
            try:
                response = s3_client.get_object(
                    Bucket='juniperproductiondata',
                    Key='data/applicationRuns/run_state.json'
                )
                return json.loads(response['Body'].read().decode('utf-8'))
            except ClientError:
                return {"runs": {"MERGER_CONTROL": {}}}
                
    except Exception as e:
        current_app.logger.error(f"Error loading runs data: {str(e)}")
        raise

def save_run_data(run_id, run_data):
    try:
        is_development = current_app.config.get('FLASK_ENV') == 'development'
        
        if is_development and os.environ.get('USE_LOCAL_FILES', 'False').lower() == 'true':
            json_path = os.path.abspath(os.path.join(
                current_app.root_path, '..', '..', 'static', 'data',
                'applicationRuns', 'run_state.json'
            ))
            # Read existing data
            try:
                with open(json_path, 'r') as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = {"runs": {"MERGER_CONTROL": {}}}
            
            # Update with new data
            existing_data["runs"]["MERGER_CONTROL"][run_id] = run_data
            
            # Write back to file
            with open(json_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
        else:
            # S3 handling for production
            s3_client = boto3.client('s3')
            try:
                response = s3_client.get_object(
                    Bucket='juniperproductiondata',
                    Key='data/applicationRuns/run_state.json'
                )
                existing_data = json.loads(response['Body'].read().decode('utf-8'))
            except ClientError:
                existing_data = {"runs": {"MERGER_CONTROL": {}}}
            
            existing_data["runs"]["MERGER_CONTROL"][run_id] = run_data
            
            s3_client.put_object(
                Bucket='juniperproductiondata',
                Key='data/applicationRuns/run_state.json',
                Body=json.dumps(existing_data, indent=2)
            )
        return True
    except Exception as e:
        current_app.logger.error(f"Error saving run data: {str(e)}")
        raise
    
    











@main_blueprint.route('/api/programs/exchange', methods=['GET'])
def get_exchange_programshold():
    try:
        is_development = current_app.config.get('FLASK_ENV') == 'development'
        
        if is_development and os.environ.get('USE_LOCAL_FILES', 'False').lower() == 'true':
            # Navigate up from backend/app to transaction_platform_app, then to static/data
            json_path = os.path.abspath(os.path.join(
                current_app.root_path,  # This is backend/app
                '..',                   # Up to backend
                '..',                   # Up to transaction_platform_app
                'static',
                'data',
                'exchange_programs.json'
            ))
            print(f"Attempting to read from path: {json_path}")  # Debug print
            with open(json_path, 'r') as f:
                programs_data = json.load(f)
        else:
            # S3 path for production
            s3_client = boto3.client('s3')
            response = s3_client.get_object(
                Bucket='juniperproductiondata',
                Key='data/programs/exchange_programs.json'
            )
            programs_data = json.loads(response['Body'].read().decode('utf-8'))
            
        return jsonify(programs_data)
        
    except Exception as e:
        current_app.logger.error(f"Error loading Exchang Review program data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@main_blueprint.route('/api/company-report/<customer_id>', methods=['GET'])
def get_customer_data(customer_id):
    try:
        # Add debug prints
        print(f"Root path: {current_app.root_path}")
        json_path = os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'companyreport_programs.json'
        )
        print(f"Constructed path: {json_path}")
        print(f"File exists: {os.path.exists(json_path)}")
        
        with open(json_path, 'r') as f:
            all_customer_data = json.load(f)
            
        # Find specific customer
        customer_data = next(
            (customer for customer in all_customer_data["customers"] 
             if customer["customer_id"] == customer_id),
            None
        )
        
        if not customer_data:
            return jsonify({'error': 'Customer not found'}), 404
            
        return jsonify(customer_data)
        
    except Exception as e:
        current_app.logger.error(f"Error loading customer data: {str(e)}")
        return jsonify({'error': str(e)}), 500



@main_blueprint.route('/api/programs/speakeasy', methods=['GET'])
def get_speakeasy_programshold():
    try:
        is_development = current_app.config.get('FLASK_ENV') == 'development'
        
        if is_development and os.environ.get('USE_LOCAL_FILES', 'False').lower() == 'true':
            # Navigate up from backend/app to transaction_platform_app, then to static/data
            json_path = os.path.abspath(os.path.join(
                current_app.root_path,  # This is backend/app
                '..',                   # Up to backend
                '..',                   # Up to transaction_platform_app
                'static',
                'data',
                'speakeasy_programs.json'
            ))
            print(f"Attempting to read from path: {json_path}")  # Debug print
            with open(json_path, 'r') as f:
                programs_data = json.load(f)
        else:
            # S3 path for production
            s3_client = boto3.client('s3')
            response = s3_client.get_object(
                Bucket='juniperproductiondata',
                Key='data/programs/speakeasy_programs.json'
            )
            programs_data = json.loads(response['Body'].read().decode('utf-8'))
            
        return jsonify(programs_data)
        
    except Exception as e:
        current_app.logger.error(f"Error loading Speakeasy Review program data: {str(e)}")
        return jsonify({'error': str(e)}), 500

    


@main_blueprint.route('/api/programs/houseapps', methods=['GET'])
def get_houseapps_programshold():
    try:
        is_development = current_app.config.get('FLASK_ENV') == 'development'
        
        if is_development and os.environ.get('USE_LOCAL_FILES', 'False').lower() == 'true':
            # Navigate up from backend/app to transaction_platform_app, then to static/data
            json_path = os.path.abspath(os.path.join(
                current_app.root_path,  # This is backend/app
                '..',                   # Up to backend
                '..',                   # Up to transaction_platform_app
                'static',
                'data',
                'houseapps_programs.json'
            ))
            print(f"Attempting to read from path: {json_path}")  # Debug print
            with open(json_path, 'r') as f:
                programs_data = json.load(f)
        else:
            # S3 path for production
            s3_client = boto3.client('s3')
            response = s3_client.get_object(
                Bucket='juniperproductiondata',
                Key='data/programs/houseapps_programs.json'
            )
            programs_data = json.loads(response['Body'].read().decode('utf-8'))
            
        return jsonify(programs_data)
        
    except Exception as e:
        current_app.logger.error(f"Error loading House App Review program data: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@main_blueprint.route('/api/programs/buildkits', methods=['GET'])
def get_buildkits_programshold():
    try:
        is_development = current_app.config.get('FLASK_ENV') == 'development'
        
        if is_development and os.environ.get('USE_LOCAL_FILES', 'False').lower() == 'true':
            # Navigate up from backend/app to transaction_platform_app, then to static/data
            json_path = os.path.abspath(os.path.join(
                current_app.root_path,  # This is backend/app
                '..',                   # Up to backend
                '..',                   # Up to transaction_platform_app
                'static',
                'data',
                'buildkits_programs.json'
            ))
            print(f"Attempting to read from path: {json_path}")  # Debug print
            with open(json_path, 'r') as f:
                programs_data = json.load(f)
        else:
            # S3 path for production
            s3_client = boto3.client('s3')
            response = s3_client.get_object(
                Bucket='juniperproductiondata',
                Key='data/programs/buildkits_programs.json'
            )
            programs_data = json.loads(response['Body'].read().decode('utf-8'))
            
        return jsonify(programs_data)
        
    except Exception as e:
        current_app.logger.error(f"Error loading Build Kit program data: {str(e)}")
        return jsonify({'error': str(e)}), 500
    




@main_blueprint.route('/api/programs/<program_type>', methods=['GET'])
def get_programs(program_type):
    try:
        # Get environment and customer path from headers
        react_env = request.headers.get('X-Environment', 'development')
        customer_path = request.headers.get('X-Customer-Path', 'hawkeyetest')
        
        print("\n=== Debug Information ===")
        print(f"Program Type: {program_type}")
        print(f"React Environment: {react_env}")
        print(f"Customer Path: {customer_path}")

        if react_env == 'development':
            # Local file path construction
            json_path = os.path.abspath(os.path.join(
                current_app.root_path,
                '..',
                '..',
                'static',
                'data',
                customer_path,  # Customer-specific folder
                f'{program_type}_programs.json'
            ))
            print(f"\n🔍 Using local path: {json_path}")
            print(f"Does path exist? {os.path.exists(json_path)}")
            
            with open(json_path, 'r') as f:
                programs_data = json.load(f)
        else:
            # S3 path construction
            s3_key = f'data/programs/{customer_path}/{program_type}_programs.json'
            print(f"\n🔍 Using S3 path: {s3_key}")
            
            s3_client = boto3.client('s3')
            response = s3_client.get_object(
                Bucket='juniperproductiondata',
                Key=s3_key
            )
            programs_data = json.loads(response['Body'].read().decode('utf-8'))
            
        return jsonify(programs_data)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': 'Failed to load program data'}), 500








    
@main_blueprint.route('/api/programs/r2d2', methods=['GET'])
def get_r2d2_programshold():
    try:
        is_development = current_app.config.get('FLASK_ENV') == 'development'
        
        if is_development and os.environ.get('USE_LOCAL_FILES', 'False').lower() == 'true':
            # Navigate up from backend/app to transaction_platform_app, then to static/data
            json_path = os.path.abspath(os.path.join(
                current_app.root_path,  # This is backend/app
                '..',                   # Up to backend
                '..',                   # Up to transaction_platform_app
                'static',
                'data',
                'r2d2_programs.json'
            ))
            print(f"Attempting to read from path: {json_path}")  # Debug print
            with open(json_path, 'r') as f:
                programs_data = json.load(f)
        else:
            # S3 path for production
            s3_client = boto3.client('s3')
            response = s3_client.get_object(
                Bucket='juniperproductiondata',
                Key='data/programs/r2d2_programs.json'
            )
            programs_data = json.loads(response['Body'].read().decode('utf-8'))
            
        return jsonify(programs_data)
        
    except Exception as e:
        current_app.logger.error(f"Error loading Insight Review program data: {str(e)}")
        return jsonify({'error': str(e)}), 500



@main_blueprint.route('/api/programs/tangibleteams', methods=['GET'])
def get_tangibleteams_programs():
    try:
        is_development = current_app.config.get('FLASK_ENV') == 'development'
        
        if is_development and os.environ.get('USE_LOCAL_FILES', 'False').lower() == 'true':
            # Navigate up from backend/app to transaction_platform_app, then to static/data
            json_path = os.path.abspath(os.path.join(
                current_app.root_path,  # This is backend/app
                '..',                   # Up to backend
                '..',                   # Up to transaction_platform_app
                'static',
                'data',
                'tangibleteams_programs.json'
            ))
            print(f"Attempting to read from path: {json_path}")  # Debug print
            with open(json_path, 'r') as f:
                programs_data = json.load(f)
        else:
            # S3 path for production
            s3_client = boto3.client('s3')
            response = s3_client.get_object(
                Bucket='juniperproductiondata',
                Key='data/programs/tangibleteams_programs.json'
            )
            programs_data = json.loads(response['Body'].read().decode('utf-8'))
            
        return jsonify(programs_data)
        
    except Exception as e:
        current_app.logger.error(f"Error loading Tangible Teams Review program data: {str(e)}")
        return jsonify({'error': str(e)}), 500













    
@main_blueprint.route('/api/sampleagreementdata', methods=['GET'])   
def get_sample_agreement_data():
    try:
        is_development = current_app.config.get('FLASK_ENV') == 'development'
        
        if is_development and os.environ.get('USE_LOCAL_FILES', 'False').lower() == 'true':
            # Navigate up from backend/app to transaction_platform_app, then to static/data
            json_path = os.path.abspath(os.path.join(
                current_app.root_path,  # This is backend/app
                '..',                   # Up to backend
                '..',                   # Up to transaction_platform_app
                'static',
                'data',
                'sampleagreementdata.json'
            ))
            print(f"Attempting to read from path: {json_path}")  # Debug print
            with open(json_path, 'r') as f:
                programs_data = json.load(f)
        else:
            # S3 path for production
            s3_client = boto3.client('s3')
            response = s3_client.get_object(
                Bucket='juniperproductiondata',
                Key='data/programs/sampleagreementdata.json'
            )
            programs_data = json.loads(response['Body'].read().decode('utf-8'))
            
        return jsonify(programs_data)
        
    except Exception as e:
        current_app.logger.error(f"Error loading Sample AGreement Data for Import program data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@main_blueprint.route('/api/flightdeck/dashboard', methods=['GET'])   
def dashboardreporting_data():
    try:
        is_development = current_app.config.get('FLASK_ENV') == 'development'
        
        if is_development and os.environ.get('USE_LOCAL_FILES', 'False').lower() == 'true':
            # Navigate up from backend/app to transaction_platform_app, then to static/data
            json_path = os.path.abspath(os.path.join(
                current_app.root_path,  # This is backend/app
                '..',                   # Up to backend
                '..',                   # Up to transaction_platform_app
                'static',
                'data',
                'dashboardreporting.json'
            ))
            print(f"Attempting to read from path: {json_path}")  # Debug print
            with open(json_path, 'r') as f:
                programs_data = json.load(f)
        else:
            # S3 path for production
            s3_client = boto3.client('s3')
            response = s3_client.get_object(
                Bucket='juniperproductiondata',
                Key='data/programs/dashboardreporting.json'
            )
            programs_data = json.loads(response['Body'].read().decode('utf-8'))
            
        return jsonify(programs_data)
        
    except Exception as e:
        current_app.logger.error(f"Error loading Sample Dashboard Reporting Data for Import program data: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
     
    
    
    
    
    
    
    
@main_blueprint.route('/api/speakeasy/questions', methods=['GET'])

def get_questions():
    try:
        is_development = current_app.config.get('FLASK_ENV') == 'development'
        
        if is_development and os.environ.get('USE_LOCAL_FILES', 'False').lower() == 'true':
            # Navigate up from backend/app to transaction_platform_app, then to static/data
            json_path = os.path.abspath(os.path.join(
                current_app.root_path,  # This is backend/app
                '..',                   # Up to backend
                '..',                   # Up to transaction_platform_app
                'static',
                'data',
                'sesamepairs.json'
            ))
            print(f"Attempting to read from path: {json_path}")  # Debug print
            with open(json_path, 'r') as f:
                all_questions = json.load(f)
        else:
            # S3 path for production
            s3_client = boto3.client('s3')
            response = s3_client.get_object(
                Bucket='juniperproductiondata',
                Key='data/programs/sesamepairs.json'
            )
            all_questions = json.loads(response['Body'].read().decode('utf-8'))
            print("Successfully read from S3")
            
        return jsonify({
            'success': True,
            'questions': all_questions['questionPairs']
        })
        
    except Exception as e:
        print(f"Error in get_questions: {str(e)}")
        current_app.logger.error(f"Error loading questions: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
        
@main_blueprint.route('/api/config/customer-instances', methods=['GET'])
def get_customer_instances():
    try:
        is_development = os.environ.get('FLASK_ENV') == 'development'
        
        if is_development:
            print(f"Root path: {current_app.root_path}")
            json_path = os.path.join(
                current_app.root_path,
                '..',
                '..',
                'static',
                'data',
                'customerInstances.json'
            )
            print(f"Constructed path: {json_path}")
            print(f"File exists: {os.path.exists(json_path)}")
            with open(json_path, 'r') as f:
                config_data = json.load(f)
        else:
            # Production S3 path
            s3_client = boto3.client('s3')
            response = s3_client.get_object(
                Bucket='juniperproductiondata',
                Key='data/configs/customerInstances.json'
            )
            config_data = json.loads(response['Body'].read().decode('utf-8'))
        
        return jsonify(config_data)
        
    except Exception as e:
        current_app.logger.error(f"Error loading customer instances: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@main_blueprint.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200



















@main_blueprint.route('/api/projects/search', methods=['GET'])
def find_project():
    try:
        buying_company = request.args.get('buyingCompany')
        target_company = request.args.get('targetCompany')
        
        if not buying_company or not target_company:
            return jsonify({"error": "Missing required query parameters"}), 400

        # Construct path to mergerControlProjects.json
        json_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'MergerControlDataPackages',
            'mergerControlProjects.json'
        ))

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"projects": {}}

        # Find matching projects
        matching_projects = [
            project for project in data["projects"].values()
            if project["buyingCompany"] == buying_company 
            and project["targetCompany"] == target_company
        ]

        return jsonify(matching_projects)

    except Exception as e:
        print(f"Error finding project: {str(e)}")
        return jsonify({'error': str(e)}), 500



@main_blueprint.route('/api/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    try:
        json_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'MergerControlDataPackages',
            'mergerControlProjects.json'
        ))
    

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            return jsonify({"error": "Project not found"}), 404

        project = data["projects"].get(project_id)
        if not project:
            return jsonify({"error": "Project not found"}), 404

        return jsonify(project)

    except Exception as e:
        print(f"Error getting project: {str(e)}")
        return jsonify({'error': str(e)}), 500
    





@main_blueprint.route('/api/projects', methods=['POST'])
def create_project():
    try:
        project_data = request.json
        print(f"Received project data: {project_data}")
        
        if not all(k in project_data for k in ['projectName', 'buyingCompany', 'targetCompany']):
            return jsonify({"error": "Missing required fields"}), 400

        # Base directory for all merger control data
        base_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'MergerControlDataPackages'
        ))

        # Create clean directory names
        buyer_dir_name = project_data['buyingCompany'].lower().replace(' ', '_')
        project_dir_name = project_data['projectName'].lower().replace(' ', '_')

        # Construct paths
        buyer_path = os.path.join(base_path, buyer_dir_name)
        project_path = os.path.join(buyer_path, project_dir_name)
        runs_path = os.path.join(project_path, 'application_runs')

        # Create directory structure if it doesn't exist
        os.makedirs(buyer_path, exist_ok=True)
        os.makedirs(project_path, exist_ok=True)
        os.makedirs(runs_path, exist_ok=True)

        print(f"Created directory structure:")
        print(f"Buyer path: {buyer_path}")
        print(f"Project path: {project_path}")
        print(f"Runs path: {runs_path}")

        # Create or update the main projects index
        index_path = os.path.join(base_path, 'mergerControlProjects.json')
        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            index_data = {
                "projects": {},
                "indices": {
                    "byCompanyPair": {}
                }
            }

        # Create project data structure
        timestamp = int(time.time() * 1000)
        project_id = project_dir_name
        new_project = {
            "projectId": project_id,
            "projectName": project_data['projectName'],
            "buyingCompany": project_data['buyingCompany'],
            "targetCompany": project_data['targetCompany'],
            "status": "active",
            "dateCreated": timestamp,
            "lastModified": timestamp,
            "metadata": {
                "createdBy": "system",
                "description": project_data.get('description', ''),
                "priority": "medium",
                "phase": "initial_filing"
            },
            "directoryPaths": {
                "base": project_path,
                "runs": runs_path
            },
            "applicationRuns": {}
        }

        # Update the main index
        index_data["projects"][project_id] = new_project
        company_pair_key = f"{project_data['buyingCompany']}_{project_data['targetCompany']}"
        if company_pair_key not in index_data["indices"]["byCompanyPair"]:
            index_data["indices"]["byCompanyPair"][company_pair_key] = []
        if project_id not in index_data["indices"]["byCompanyPair"][company_pair_key]:
            index_data["indices"]["byCompanyPair"][company_pair_key].append(project_id)

        # Write main index
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)

        # Create project-specific data file
        project_data_path = os.path.join(project_path, 'project_data.json')
        with open(project_data_path, 'w') as f:
            json.dump(new_project, f, indent=2)

        return jsonify(new_project)

    except Exception as e:
        print(f"Error creating project: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    
    
@main_blueprint.route('/api/setup/merger-control-directories', methods=['POST'])
def setup_merger_control_directories():
    try:
        # Base directory for all merger control data
        base_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'MergerControlDataPackages'
        ))
        
        print(f"Setting up directories in: {base_path}")

        # Create base directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)

        # Read existing project data
        json_path = os.path.join(base_path, 'mergerControlProjects.json')
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return jsonify({
                "error": f"Could not read project data: {str(e)}",
                "basePath": base_path
            }), 500

        # Track what we create
        created_dirs = []
        existing_dirs = []

        # Process each project
        for project_id, project in data['projects'].items():
            buyer_dir_name = project['buyingCompany'].lower().replace(' ', '_')
            project_dir_name = project_id.lower().replace(' ', '_')

            # Create paths
            buyer_path = os.path.join(base_path, buyer_dir_name)
            project_path = os.path.join(buyer_path, project_dir_name)
            runs_path = os.path.join(project_path, 'application_runs')

            # Create directories if they don't exist
            for path in [buyer_path, project_path, runs_path]:
                if not os.path.exists(path):
                    os.makedirs(path)
                    created_dirs.append(path)
                else:
                    existing_dirs.append(path)

            # Create or update project-specific data file
            project_data_path = os.path.join(project_path, 'project_data.json')
            with open(project_data_path, 'w') as f:
                json.dump(project, f, indent=2)

        return jsonify({
            "success": True,
            "basePath": base_path,
            "created": created_dirs,
            "existing": existing_dirs
        })

    except Exception as e:
        print(f"Error setting up directories: {str(e)}")
        return jsonify({
            "error": str(e),
            "basePath": base_path
        }), 500 


from datetime import datetime

@main_blueprint.route('/api/projects/<project_id>/runs', methods=['GET', 'POST'])
def create_run(project_id):
    try:
        print(f"🚀 Creating new run for project: {project_id}")
        
        # First load the main projects file to get the companies
        base_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'MergerControlDataPackages',
            'mergerControlProjects.json'
        ))

        # Load projects data
        with open(base_path, 'r') as f:
            all_projects = json.load(f)
            project_data = all_projects['projects'].get(project_id)
            if not project_data:
                raise ValueError(f"Project {project_id} not found")
            
            buying_company = project_data['buyingCompany']
            target_company = project_data['targetCompany']
            print(f"📊 Project details - Buying: {buying_company}, Target: {target_company}")

        # Define paths for both company data files
        target_data_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'MergerControlDataPackages',
            'targetCompanyData.json'
        ))

        buying_data_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'MergerControlDataPackages',
            'buyingCompanyData.json'
        ))

        print(f"📁 Loading buying and target company data from: {target_data_path} and {buying_data_path}")
        
        # Create consistent timestamp format for both display and file naming
        current_time = datetime.now()
        timestamp_compact = current_time.strftime("%Y%m%d_%H%M")
        timestamp_iso = current_time.strftime("%Y-%m-%dT%H:%M:%S")  # ISO format for JSON
        
        run_id = f"{project_id}_run_{timestamp_compact}"
        run_display_name = f"{project_data['projectName']} Analysis - {current_time.strftime('%B %d, %Y %I:%M %p')}"

        # Set up paths
        project_dir = os.path.join(os.path.dirname(base_path), buying_company.lower(), project_id)
        run_path = os.path.join(project_dir, 'application_runs', run_id)
        os.makedirs(run_path, exist_ok=True)
        print(f"📂 Created run directory: {run_path}")

        # Load and save target company data
        with open(target_data_path, 'r') as f:
            target_companies = json.load(f)
            if target_company not in target_companies:
                raise ValueError(f"Target company {target_company} data not found")
            run_data = target_companies[target_company]
            
            # Add project ID to the run data
            run_data['projectId'] = project_id

        run_data_path = os.path.join(run_path, 'run_data.json')
        with open(run_data_path, 'w') as f:
            json.dump(run_data, f, indent=2)
        print(f"💾 Saved target company data to: {run_data_path}")

        # Load and save buying company data
        with open(buying_data_path, 'r') as f:
            buying_companies = json.load(f)
            if buying_company not in buying_companies:
                raise ValueError(f"Buying company {buying_company} data not found")
            buying_data = buying_companies[buying_company]
            print(buying_data)
        buying_data_path = os.path.join(run_path, 'buyingCompanyDataRun.json')
        with open(buying_data_path, 'w') as f:
            json.dump(buying_data, f, indent=2)
        print(f"💾 Saved buying company data to: {buying_data_path}")

        # Update project's application runs index
        project_data.setdefault('applicationRuns', {})
        project_data['applicationRuns'][run_id] = {
            "runId": run_id,
            "displayName": run_display_name,
            "status": "setup_initiated",
            "dateCreated": timestamp_iso,  # Use consistent ISO format timestamp
            "lastModified": timestamp_iso   # Use consistent ISO format timestamp
        }

        # Save updated project data
        with open(base_path, 'w') as f:
            json.dump(all_projects, f, indent=2)

        print(f"✅ Successfully created new run: {run_id}")
        # Return both sets of data with consistent timestamps
        return jsonify({
            "runId": run_id,
            "projectId": project_id,
            "displayName": run_display_name,
            "dateCreated": timestamp_iso,
            "targetCompanyData": run_data,
            "buyingData": buying_data
        })

    except Exception as e:
        print(f"❌ Error creating new run: {str(e)}")
        return jsonify({'error': str(e)}), 500
 
 
 
@main_blueprint.route('/api/projects/<project_id>/runs/<run_id>/toggle_regional_blocks', methods=['POST'])
def toggle_regional_blocks(project_id, run_id):
    try:
        print(f"🔄 Processing presence updates for project: {project_id}, run: {run_id}")
        
        request_data = request.get_json()
        changes = request_data.get('changes', [])
        print(f"📥 Received {len(changes)} presence changes")

        # First load the main projects file to get the buying company
        base_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'MergerControlDataPackages',
            'mergerControlProjects.json'
        ))

        # Load projects data to get buying company
        with open(base_path, 'r') as f:
            all_projects = json.load(f)
            project_data = all_projects['projects'].get(project_id)
            if not project_data:
                raise ValueError(f"Project {project_id} not found")
            
            buying_company = project_data['buyingCompany']
            print(f"📊 Found buying company: {buying_company}")

        # Now load run data with correct buying company in path
        run_data_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'MergerControlDataPackages',
            buying_company,  # Use buying company instead of hardcoded 'artemis'
            project_id,
            'application_runs',
            run_id,
            'run_data.json'
        ))

        # Load and preserve existing data
        print(f"📝 Loading current run data from: {run_data_path}")
        with open(run_data_path, 'r') as f:
            run_data = json.load(f)
            
        # Get regional_blocks from root level
        regional_blocks = run_data.get('regional_blocks', {})
            
            
            
            
            
            
      

        # Process presence changes
        for change in changes:
            block_key = change['block_key']
            member_state_key = change['member_state_key']
            new_presence = change['presence']

            print(f"⚡ Processing presence change - Block: {block_key}, State: {member_state_key}, New Presence: {new_presence}")

            # Skip template
            if block_key == 'template':
                continue

            # Keep existing block if it exists, or create new one preserving the structure
            if block_key not in regional_blocks:
                if new_presence:
                    print(f"📝 Adding new block: {block_key}")
                    regional_blocks[block_key] = {
                        'presence': True,
                        'block_metrics': {
                            'revenue': None,
                            'revenue_formatted': None,
                            'employees': None,
                            'assets': None,
                            'registered_office': None,
                            'aggregated_market_share': None
                        },
                        'member_states': {}
                    }

            if block_key in regional_blocks:
                block = regional_blocks[block_key]
                member_states = block.get('member_states', {})

                # Just update presence for existing state
                if member_state_key in member_states:
                    print(f"🔄 Updating presence for existing state: {member_state_key}")
                    member_states[member_state_key]['presence'] = new_presence
                # Create new state if needed
                elif new_presence:
                    print(f"📝 Adding new state: {member_state_key}")
                    # Copy structure from template if it exists
                    if 'template' in regional_blocks['template']['member_states']:
                        member_states[member_state_key] = copy.deepcopy(
                            regional_blocks['template']['member_states']['template']
                        )
                        member_states[member_state_key]['presence'] = True
                    else:
                        member_states[member_state_key] = {
                            'presence': True,
                            'metrics': {
                                'revenue': None,
                                'employees': None,
                                'assets': None
                            },
                            'market_shares': {},
                            'merger_control_info': {
                                'filing_required': None,
                                'thresholds': {
                                    'revenue': None,
                                    'market_share': None
                                },
                                'review_period_days': None
                            }
                        }

        # Update block presence based on member states
        for block_key, block_data in regional_blocks.items():
            if block_key != 'template':
                has_active_states = any(
                    state.get('presence', False) 
                    for state in block_data.get('member_states', {}).values()
                    if isinstance(state, dict)  # Skip template
                )
                block_data['presence'] = has_active_states

        # Update run_data and save
        run_data['regional_blocks'] = regional_blocks
        
        current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        run_data['lastModified'] = current_time

        print(f"💾 Saving presence updates to: {run_data_path}")
        with open(run_data_path, 'w') as f:
            json.dump(run_data, f, indent=2)

        print(f"✅ Successfully updated presence states")
        return jsonify({
            'success': True,
            'message': "Successfully updated presence states",
            'data': run_data
        })

    except Exception as e:
        print(f"❌ Error updating presence states: {str(e)}")
        traceback.print_exc()
        return jsonify(error=str(e)), 500
 
 
 
 
 






@main_blueprint.route('/api/projects/<project_id>/runs/<run_id>/update_member_state_data', methods=['POST'])
def update_member_state_data(project_id, run_id):
    try:
        print(f"\n🚀 STARTING UPDATE PROCESS")
        print(f"📋 Project: {project_id}, Run: {run_id}")
        
       
        print(f"🔄 Processing metrics updates for project: {project_id}, run: {run_id}")
        
        request_data = request.get_json()
        changes = request_data.get('changes', [])
        print(f"📥 Received {len(changes)} presence changes")

        # First load the main projects file to get the buying company
        base_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'MergerControlDataPackages',
            'mergerControlProjects.json'
        ))

        # Load projects data to get buying company
        with open(base_path, 'r') as f:
            all_projects = json.load(f)
            project_data = all_projects['projects'].get(project_id)
            if not project_data:
                raise ValueError(f"Project {project_id} not found")
            
            buying_company = project_data['buyingCompany']
            print(f"📊 Found buying company: {buying_company}")

        # Now load run data with correct buying company in path
        run_data_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'MergerControlDataPackages',
            buying_company,  # Use buying company instead of hardcoded 'artemis'
            project_id,
            'application_runs',
            run_id,
            'run_data.json'
        ))

      
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       

        print(f"\n📝 Loading existing run data from: {run_data_path}")
        with open(run_data_path, 'r') as f:
            run_data = json.load(f)

        # Get updates and regional_blocks from root level
        updates = request_data.get('updates', {})
        regional_blocks = run_data.get('regional_blocks', {})

        # Process each update individually
        for region_key, region_updates in updates.items():
            region = regional_blocks.get(region_key)  # Changed from targetCompanyData
            if not region:
                continue

            for state_key, state_updates in region_updates.items():
                state = region.get('member_states', {}).get(state_key)
                if not state or not state.get('presence', False):
                    continue

                print(f"\n🔄 Processing update for {region_key}/{state_key}")
                
                # Handle metrics updates one at a time
                if 'metrics' in state_updates:
                    update_metrics = state_updates['metrics']
                    # Ensure metrics structure exists
                    if 'metrics' not in state:
                        state['metrics'] = {}
                    
                    # Process each metric individually
                    for metric_key in ['revenue', 'employees', 'assets', 'transaction_size']:
                        if metric_key in update_metrics:
                            print(f"  📊 Updating {metric_key}: {update_metrics[metric_key]}")
                            print(f"  📊 Previous value: {state['metrics'].get(metric_key)}")
                            state['metrics'][metric_key] = update_metrics[metric_key]
                            print(f"  📊 New value: {state['metrics'][metric_key]}")

                # Handle market shares updates
                if 'market_shares' in state_updates and region_key != 'united_states':
                    update_shares = state_updates['market_shares']
                    # Ensure market_shares structure exists
                    if 'market_shares' not in state:
                        state['market_shares'] = {}
                    
                    if 'clinical_software' in update_shares:
                        print(f"  📈 Updating market share: {update_shares['clinical_software']}")
                        print(f"  📈 Previous value: {state['market_shares'].get('clinical_software')}")
                        state['market_shares']['clinical_software'] = update_shares['clinical_software']
                        print(f"  📈 New value: {state['market_shares']['clinical_software']}")

        # Put updated blocks back into run_data
        run_data['regional_blocks'] = regional_blocks
        
        # Update timestamp
        run_data['lastModified'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # Create backup
        backup_path = f"{run_data_path}.bak"
        if os.path.exists(run_data_path):
            with open(run_data_path, 'r') as source:
                with open(backup_path, 'w') as target:
                    json.dump(json.load(source), target, indent=2)

        # Save updated file
        print(f"\n💾 Saving updates to: {run_data_path}")
        with open(run_data_path, 'w') as f:
            json.dump(run_data, f, indent=2)

        print(f"\n✅ Successfully saved updates")
        return jsonify({
            'success': True,
            'message': "Successfully updated worksheet data",
            'data': run_data
        })

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        traceback.print_exc()
        return jsonify(error=str(e)), 500











@main_blueprint.route('/api/litigation/cases', methods=['GET'])
def get_litigation_cases():
    try:
        # Get path to litigation packages
        base_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'PaceLitigationPackages'
        ))
        
        if not os.path.exists(base_path):
            return jsonify({
                'error': 'Litigation directory not found',
                'status': 'error'
            }), 404

        # Get all JSON files in directory
        json_files = [f for f in os.listdir(base_path) if f.endswith('.json')]
        
        if not json_files:
            return jsonify({
                'message': 'No litigation cases found',
                'data': [],
                'status': 'success'
            }), 200

        cases = []
        for json_file in json_files:
            try:
                file_path = os.path.join(base_path, json_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)
                    cases.append(case_data)
            except json.JSONDecodeError as e:
                print(f"Error reading {json_file}: {str(e)}")
                continue
            except Exception as e:
                print(f"Unexpected error reading {json_file}: {str(e)}")
                continue

        # Optional query parameters for filtering
        filters = {
            'jurisdiction': request.args.get('jurisdiction'),
            'status': request.args.get('status'),
            'disease': request.args.get('disease')
        }

        # Apply filters if they exist
        filtered_cases = filter_cases(cases, filters)

        return jsonify({
            'data': filtered_cases,
            'total': len(filtered_cases),
            'status': 'success'
        }), 200

    except Exception as e:
        print(f"Error getting litigation cases: {str(e)}")
        return jsonify(error=str(e)), 500

def filter_cases(cases, filters):
    """Filter cases based on query parameters"""
    filtered = cases
    
    if filters['jurisdiction']:
        filtered = [
            case for case in filtered 
            if any(
                lawsuit.get('lawsuit_jurisdiction') == filters['jurisdiction']
                for lawsuit in case.get('lawsuit_service', [])
            )
        ]
    
    if filters['disease']:
        filtered = [
            case for case in filtered 
            if any(
                any(
                    diagnosis.get('diagnosis_disease_name') == filters['disease']
                    for diagnosis in injured_party.get('diagnosis', [])
                )
                for lawsuit in case.get('lawsuit_service', [])
                for injured_party in lawsuit.get('injured_party', [])
            )
        ]
    
    if filters['status']:
        filtered = [
            case for case in filtered 
            if any(
                calculate_status(lawsuit.get('lawsuit_answer_due_date')) == filters['status']
                for lawsuit in case.get('lawsuit_service', [])
            )
        ]
    
    return filtered

def calculate_status(due_date):
    """Calculate status based on due date"""
    from datetime import datetime
    if not due_date:
        return 'PENDING'
    
    try:
        due = datetime.strptime(due_date, '%m/%d/%Y')
        days_until_due = (due - datetime.now()).days
        
        if days_until_due < 0:
            return 'OVERDUE'
        if days_until_due <= 7:
            return 'URGENT'
        return 'ACTIVE'
    except ValueError:
        return 'PENDING'







@main_blueprint.route('/api/projects/<project_id>/runs/<run_id>', methods=['GET'])
def get_run(project_id, run_id):
    try:
        print(f"🔍 Fetching run data for project: {project_id}, run: {run_id}")
        
        # First load the main projects file to get the companies
        base_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'MergerControlDataPackages',
            'mergerControlProjects.json'
        ))

        # Load projects data to get company info
        with open(base_path, 'r') as f:
            all_projects = json.load(f)
            project_data = all_projects['projects'].get(project_id)
            if not project_data:
                raise ValueError(f"Project {project_id} not found")
            
            buying_company = project_data['buyingCompany']
            print(f"📊 Found buying company: {buying_company}")

        # Get path to run data
        project_dir = os.path.join(os.path.dirname(base_path), buying_company.lower(), project_id)
        run_path = os.path.join(project_dir, 'application_runs', run_id)
        run_data_path = os.path.join(run_path, 'run_data.json')
        buying_data_path = os.path.join(run_path, 'buyingCompanyDataRun.json')

        print(f"📁 Loading run data from: {run_data_path}")

        # Load run data
        with open(run_data_path, 'r') as f:
            run_data = json.load(f)

      

        # Construct response with all necessary data
        response_data = {
            "runId": run_id,
            "projectId": project_id,
            "displayName": project_data['applicationRuns'][run_id]['displayName'],
            "dateCreated": project_data['applicationRuns'][run_id]['dateCreated'],
            "lastModified": project_data['applicationRuns'][run_id]['lastModified'],
            "buyingCompany": buying_company,
            "targetCompanyData": run_data,
           
            "status": project_data['applicationRuns'][run_id]['status']
        }

        print(f"✅ Successfully retrieved run data")
        print(run_data)
        return jsonify(response_data)

    except FileNotFoundError as e:
        print(f"❌ File not found error: {str(e)}")
        return jsonify(error=f"Could not find required file: {str(e)}"), 404
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {str(e)}")
        return jsonify(error="Invalid JSON data encountered"), 400
    except Exception as e:
        print(f"❌ Error retrieving run data: {str(e)}")
        return jsonify(error=str(e)), 500





@main_blueprint.route('/api/projects/<project_id>/runs/<run_id>/buying_data', methods=['GET'])
def get_buying_data(project_id, run_id):
    try:
        print(f"🔍 Fetching buying data for project: {project_id}, run: {run_id}")
        
        # First load the main projects file to get the companies
        base_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'MergerControlDataPackages',
            'mergerControlProjects.json'
        ))

        # Load projects data to get company info
        with open(base_path, 'r') as f:
            all_projects = json.load(f)
            project_data = all_projects['projects'].get(project_id)
            if not project_data:
                raise ValueError(f"Project {project_id} not found")
            
            buying_company = project_data['buyingCompany']
            print(f"📊 Found buying company: {buying_company}")

        # Get path to run data
        project_dir = os.path.join(os.path.dirname(base_path), buying_company.lower(), project_id)
        run_path = os.path.join(project_dir, 'application_runs', run_id)
     
        buying_data_path = os.path.join(run_path, 'buyingCompanyDataRun.json')

        print(f"📁 Loading buying data from: {buying_data_path}")

    
    

        # Load buying company data
        with open(buying_data_path, 'r') as f:
            buying_data = json.load(f)

        # Construct response with all necessary data
        response_data = {
            "runId": run_id,
            "projectId": project_id,
            "displayName": project_data['applicationRuns'][run_id]['displayName'],
            "dateCreated": project_data['applicationRuns'][run_id]['dateCreated'],
            "lastModified": project_data['applicationRuns'][run_id]['lastModified'],
            "buyingCompany": buying_company,
         
            "buyingCompanyData": buying_data,
            "status": project_data['applicationRuns'][run_id]['status']
        }

        print(f"✅ Successfully retrieved buying company data")
        print(buying_data)
        return jsonify(response_data)

    except FileNotFoundError as e:
        print(f"❌ File not found error: {str(e)}")
        return jsonify(error=f"Could not find required file: {str(e)}"), 404
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {str(e)}")
        return jsonify(error="Invalid JSON data encountered"), 400
    except Exception as e:
        print(f"❌ Error retrieving run data: {str(e)}")
        return jsonify(error=str(e)), 500








# Inside main_routes.py

@main_blueprint.route('/api/funds/registry', methods=['GET'])
def get_fund_registry():
    try:
        print("🔍 Fetching fund registry data")
        
        # Build path to registry file
        base_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'FundDataPackages',
            'fund_registry.json'
        ))
        
        print(f"📁 Loading registry from: {base_path}")
        
        with open(base_path, 'r') as f:
            registry = json.load(f)
        return jsonify(registry)
    except FileNotFoundError as e:
        print(f"❌ Registry file not found: {e}")
        return jsonify({"error": "Registry not found"}), 404
    except json.JSONDecodeError as e:
        print(f"❌ Invalid registry format: {e}")
        return jsonify({"error": "Invalid registry format"}), 500

@main_blueprint.route('/api/funds/<frn>', methods=['GET'])
def get_fund_data(frn):
    try:
        print(f"🔍 Fetching data for fund FRN: {frn}")
        
        # Build path to registry file
        base_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'FundDataPackages',
            'fund_registry.json'
        ))
        
        # Load registry to get file path
        with open(base_path, 'r') as f:
            registry = json.load(f)
        
        if frn not in registry['firms']:
            print(f"❌ Fund {frn} not found in registry")
            return jsonify({"error": "Fund not found"}), 404
            
        fund_file = registry['firms'][frn]['dataFile']
        
        # Build path to individual fund file
        fund_data_path = os.path.join(
            os.path.dirname(base_path),
            'FundDetails',
            fund_file
        )
        
        print(f"📁 Loading fund data from: {fund_data_path}")
        
        # Load individual fund data
        with open(fund_data_path, 'r') as f:
            fund_data = json.load(f)
            
        return jsonify(fund_data)
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return jsonify({"error": "Data not found"}), 404
    except json.JSONDecodeError as e:
        print(f"❌ Invalid data format: {e}")
        return jsonify({"error": "Invalid data format"}), 500












@main_blueprint.route('/api/funds/query/general-details', methods=['POST'])
def query_fund_details():
    try:
        data = request.get_json()
        print(f"🔍 Processing query request for funds: {data['funds']}")

        # Build path to FundDetails directory using the same pattern
        base_dir = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'FundDataPackages',
            'FundDetails'
        ))
        
        print(f"📁 Target directory: {base_dir}")
        
        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        results = []
        for frn in data['funds']:
            # Construct FCA API URL
            url = f"https://register.fca.org.uk/services/V0.1/Firm/{frn}"
            
            headers = {
                'x-auth-email': 'rreynolds@tangiblelabs.ai',
                'x-auth-key': '6085d3802c5a4a255e38fb30b914899a',
                'Content-Type': 'application/json'
            }
            
            print(f"📤 Requesting details for FRN: {frn}")
            response = requests.get(url, headers=headers)
            
            if response.ok:
                print(f"✅ Received data for FRN: {frn}")
                fund_data = response.json()
                
                # Debug the response structure
                print(f"📊 Response structure: {json.dumps(fund_data, indent=2)}")
                
                # Extract organization name from Data[0]
                org_name = fund_data['Data'][0]['Organisation Name']
                first_word = org_name.split()[0].lower()
                filename = f"{first_word}_{frn}.json"
                
                # Construct full file path
                file_path = os.path.join(base_dir, filename)
                print(f"📝 Writing to: {file_path}")
                
                # Save the data to a JSON file
                print(f"💾 Saving fund details to {filename}")
                with open(file_path, 'w') as f:
                    json.dump(fund_data, f, indent=2)
                
                results.append(fund_data)
            else:
                print(f"❌ Error fetching FRN {frn}: {response.status_code}")
            
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        print(f"Full error details: ", str(e.__class__), str(e))
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@main_blueprint.route('/api/utility/json-access', methods=['POST'])
def access_json():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['endpoint', 'action']
        if not all(field in data for field in required_fields):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields'
            }), 400

        # Build standardized base path ending with static/data
        base_path = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data'
        ))

        # Construct full path by joining base_path with provided endpoint
        full_path = os.path.join(base_path, data['endpoint'])
        print(f"🔍 Accessing JSON at path: {full_path}")

        # Initialize accessor
        accessor = JsonAccessor(file_path=full_path)

        # Handle different actions
        if data['action'] == 'get_value':
            result = accessor.get_value(data['path'], data.get('default'))
            return jsonify({
                'status': 'success',
                'value': result
            })

        elif data['action'] == 'get_schema':
            schema = accessor.get_schema()
            return jsonify({
                'status': 'success',
                'schema': schema
            })
            
        elif data['action'] == 'find_key':
            results = accessor.find_key_recursive(data['key'])
            return jsonify({
                'status': 'success',
                'results': results
            })   

    except Exception as e:
        print(f"❌ Error in JSON access: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500



def get_mongo_client():
    """Get MongoDB client using configuration from mongo config directory"""
    config_path = os.path.abspath(os.path.join(
        current_app.root_path,  # starts at backend where application.py is
        '..',                   # up one level to backend parent
        '..',                   # up another level to project root
        'mongo',               # into mongo directory
        'config',
        'config.json'  # the config file
    ))
    
    # Debug print to see the path
    print(f"Attempting to read config from: {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    return MongoClient(config['connection_string'])

def infer_schema_from_data(data):
    """
    Infer MongoDB schema from sample data
    Returns a JSON Schema compatible with MongoDB validation
    """
    if not data:
        return None
        
    sample = data[0] if isinstance(data, list) else data
    schema = {
        "bsonType": "object",
        "required": [],
        "properties": {}
    }
    
    for key, value in sample.items():
        if value is not None:  # Only add non-null fields
            field_type = type(value)
            
            if field_type == str:
                schema["properties"][key] = {"bsonType": "string"}
            elif field_type == int:
                schema["properties"][key] = {"bsonType": "int"}
            elif field_type == float:
                schema["properties"][key] = {"bsonType": "double"}
            elif field_type == bool:
                schema["properties"][key] = {"bsonType": "bool"}
            elif field_type == list:
                schema["properties"][key] = {"bsonType": "array"}
            elif field_type == dict:
                nested_schema = infer_schema_from_data(value)
                if nested_schema:
                    schema["properties"][key] = nested_schema
            
            schema["required"].append(key)
    
    return schema

@main_blueprint.route('/api/mongodb/databases', methods=['GET'])
def get_databases():
    """List all databases and their collections"""
    try:
        client = get_mongo_client()
        database_list = []
        
        # Print available databases for debugging
        print(f"Available databases: {client.list_database_names()}")
        
        for db_name in client.list_database_names():
            if db_name not in ['admin', 'local', 'config']:
                db = client[db_name]
                collections = []
                
                # Print collections for debugging
                print(f"Collections in {db_name}: {db.list_collection_names()}")
                
                for coll_name in db.list_collection_names():
                    try:
                        # Get collection info including validation rules
                        collection_info = db.command("listCollections", 
                                                  filter={"name": coll_name})
                        
                        # Get a sample document
                        sample = list(db[coll_name].find().limit(1))
                        
                        collections.append({
                            'name': coll_name,
                            'path': f'{db_name}/{coll_name}',
                            'sample': json.loads(json_util.dumps(sample[0])) if sample else None,
                            'schema': collection_info['cursor']['firstBatch'][0].get('options', {}).get('validator', {})
                        })
                    except Exception as e:
                        print(f"Error processing collection {coll_name}: {str(e)}")
                        continue
                
                database_list.append({
                    'name': db_name,
                    'collections': collections
                })
        
        # Return in format expected by React component
        return jsonify({
            'databases': database_list,
            'error': None
        })
        
    except Exception as e:
        print(f"Error in get_databases: {str(e)}")
        return jsonify({
            'databases': [],
            'error': str(e)
        })

@main_blueprint.route('/api/mongodb/databases', methods=['POST'])
def create_database():
    """Create a new database"""
    try:
        data = request.json
        db_name = data.get('name')
        
        if not db_name:
            return jsonify({'error': 'Database name is required'}), 400
            
        client = get_mongo_client()
        
        # MongoDB creates databases lazily - create a temporary collection
        db = client[db_name]
        db.create_collection('temp')
        db.drop_collection('temp')
        
        return jsonify({
            'message': f'Database {db_name} created successfully',
            'error': None
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@main_blueprint.route('/api/mongodb/collections', methods=['POST'])
def create_collection():
    """Create a new collection with optional schema validation"""
    try:
        data = request.json
        db_name = data.get('database')
        coll_name = data.get('name')
        schema = data.get('schema')  # Optional schema
        
        if not db_name or not coll_name:
            return jsonify({'error': 'Database and collection names are required'}), 400
            
        client = get_mongo_client()
        db = client[db_name]
        
        # If schema is provided, create collection with validation
        if schema:
            validator = {"$jsonSchema": schema}
            db.create_collection(
                coll_name,
                validator=validator,
                validationAction="error"  # or "warn"
            )
        else:
            db.create_collection(coll_name)
        
        return jsonify({
            'message': f'Collection {coll_name} created successfully',
            'error': None
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@main_blueprint.route('/api/mongodb/load-data', methods=['POST'])
def load_data():
    """Load data from JSON or CSV file into specified collection"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        db_name = request.form.get('database')
        coll_name = request.form.get('collection')
        infer_schema = request.form.get('infer_schema', 'false').lower() == 'true'
        
        if not db_name or not coll_name:
            return jsonify({'error': 'Database and collection names are required'}), 400
            
        client = get_mongo_client()
        db = client[db_name]
        collection = db[coll_name]
        
        # Process file and load data
        if file.filename.endswith('.json'):
            data = json.load(file)
        elif file.filename.endswith('.csv'):
            stream = io.StringIO(file.stream.read().decode("UTF8"))
            csv_reader = csv.DictReader(stream)
            data = list(csv_reader)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # If requested, infer and update schema from the data
        if infer_schema:
            schema = infer_schema_from_data(data)
            if schema:
                db.command("collMod", coll_name,
                          validator={"$jsonSchema": schema},
                          validationAction="error")
        
        # Insert the data
        if isinstance(data, list):
            collection.insert_many(data)
        else:
            collection.insert_one(data)
        
        return jsonify({
            'message': 'Data loaded successfully',
            'error': None
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@main_blueprint.route('/api/mongodb/schema', methods=['GET', 'PUT'])
def manage_schema():
    """Get or update schema for a collection"""
    try:
        db_name = request.args.get('database')
        coll_name = request.args.get('collection')
        
        if not db_name or not coll_name:
            return jsonify({'error': 'Database and collection names are required'}), 400
            
        client = get_mongo_client()
        db = client[db_name]
        
        if request.method == 'GET':
            # Get current schema
            collection_info = db.command("listCollections", 
                                      filter={"name": coll_name})
            schema = collection_info['cursor']['firstBatch'][0].get('options', {}).get('validator', {})
            return jsonify({'schema': schema})
            
        else:  # PUT
            # Update schema
            new_schema = request.json.get('schema')
            if not new_schema:
                return jsonify({'error': 'Schema is required'}), 400
                
            db.command("collMod", coll_name,
                      validator={"$jsonSchema": new_schema},
                      validationAction="error")
            return jsonify({'message': 'Schema updated successfully'})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500







@main_blueprint.route('/api/litigation/current', methods=['GET'])
def get_current_litigation():
    print("Starting get_current_litigation route handler")
    try:
        litigation_data = list(db.asbestosLitigation.find({}, {'_id': 0}))
        print(f"Retrieved {len(litigation_data)} documents from MongoDB")
        
        transformed_data = []
        for case in litigation_data:
            print(f"\nProcessing case...")
            
            # Get data from correct paths
            lawsuit_info = case.get('lawsuit_info', {})
            file_metadata = case.get('file_metadata', {})
            
            # Transform the case data
            transformed_case = {
                'log_number': file_metadata.get('log_number', lawsuit_info.get('docket_number')),  # Use docket as fallback
                'case_caption': lawsuit_info.get('case_caption'),
                'defendants': [d.get('company_name', d.get('defendant_company')) 
                             for d in case.get('defendants', [])],
                'jurisdiction': f"{lawsuit_info.get('jurisdiction', '')}, {lawsuit_info.get('jurisdiction_state', '')}".strip(', '),
                'disease_name': (case.get('injured_parties', [{}])[0]
                               .get('diagnoses', [{}])[0]
                               .get('disease_name', 'Unknown')),
                'file_date': lawsuit_info.get('file_date'),
                'answer_due_date': lawsuit_info.get('answer_due_date'),
                'status': 'NEW'  # Default status
            }
            
            # Debug print
            print(f"Transformed case data:")
            print(f"Log/Docket: {transformed_case['log_number']}")
            print(f"Caption: {transformed_case['case_caption']}")
            print(f"Jurisdiction: {transformed_case['jurisdiction']}")
            
            # Only add cases that have essential data
            if transformed_case['case_caption']:  # At minimum need a caption
                transformed_data.append(transformed_case)
                print(f"Successfully added case to transformed data")
        
        print(f"Transformation complete. Returning {len(transformed_data)} cases")
        return jsonify(transformed_data)
    except Exception as e:
        print(f"Error in get_current_litigation: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500





@main_blueprint.route('/api/litigation/production', methods=['GET'])
def get_production_litigation():
    print("Starting get_production_litigation route handler")
    try:
        # Changed collection to asbestosProduction
        litigation_data = list(db.asbestosProduction.find({}, {'_id': 0}))
        print(f"Retrieved {len(litigation_data)} documents from asbestosProduction")
        
        transformed_data = []
        for case in litigation_data:
            print(f"\nProcessing production case...")
            
            # Get data from correct paths
            lawsuit_info = case.get('lawsuit_info', {})
            file_metadata = case.get('file_metadata', {})
            
            # Transform the case data - same logic as before
            transformed_case = {
                'log_number': file_metadata.get('log_number', lawsuit_info.get('docket_number')),
                'case_caption': lawsuit_info.get('case_caption'),
                'defendants': [d.get('company_name', d.get('defendant_company')) 
                             for d in case.get('defendants', [])],
                'jurisdiction': f"{lawsuit_info.get('jurisdiction', '')}, {lawsuit_info.get('jurisdiction_state', '')}".strip(', '),
                'disease_name': (case.get('injured_parties', [{}])[0]
                               .get('diagnoses', [{}])[0]
                               .get('disease_name', 'Unknown')),
                'file_date': lawsuit_info.get('file_date'),
                'answer_due_date': lawsuit_info.get('answer_due_date'),
                'status': 'NEW'  # These are all new cases by definition
            }
            
            # Debug print
            print(f"Transformed production case data:")
            print(f"Log/Docket: {transformed_case['log_number']}")
            print(f"Caption: {transformed_case['case_caption']}")
            print(f"Jurisdiction: {transformed_case['jurisdiction']}")
            
            # Only add cases that have essential data
            if transformed_case['case_caption']:
                transformed_data.append(transformed_case)
                print(f"Successfully added production case")
        
        print(f"Production transformation complete. Returning {len(transformed_data)} cases")
        return jsonify(transformed_data)
    except Exception as e:
        print(f"Error in get_production_litigation: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500








@main_blueprint.route('/api/litigation/run-matching', methods=['POST'])
def run_matching():
    try:
        print("Starting matching process for production cases")
        
        # Get our customer list from the database
        customer_list = list(db.coreCustomers.find({}, {"official_name": 1}))
        customer_names = [c["official_name"] for c in customer_list]
        
        # Get unprocessed cases from asbestosConfirmed
        unprocessed_cases = list(db.asbestosConfirmed.find({
            "confirmation_data.processing_status": "NEW"
        }))
        
        processed_count = 0
        for case in unprocessed_cases:
            # Get defendants from the case
            defendants = case.get("defendants", [])
            
            # Store all matches for this case
            case_matches = []
            highest_confidence = 0
            best_match = None
            
            # Process each defendant
            for defendant in defendants:
                defendant_name = defendant.get("defendant_company", "")
                matches = company_matcher.find_matches(
                    defendant_name, 
                    customer_names, 
                    min_confidence=70  # Our minimum threshold
                )
                
                # Store matches and track highest confidence
                for match in matches:
                    case_matches.append(match)
                    if match['confidence'] > highest_confidence:
                        highest_confidence = match['confidence']
                        best_match = match
            
            # Determine match status based on confidence
            match_status = "NO_MATCH"
            if highest_confidence >= 90:
                match_status = "CONFIRMED"
            elif highest_confidence >= 70:
                match_status = "PROBABLE"
            
            # Update the case with matching results
            update_data = {
                "confirmation_data.confidence_score": highest_confidence,
                "confirmation_data.match_status": match_status,
                "confirmation_data.processing_status": "PROCESSING",
                "matching_results": {
                    "best_match": best_match,
                    "confidence_threshold": 70,
                    "potential_matches": case_matches,
                    "match_date": datetime.utcnow()
                },
                "audit_trail.last_modified": datetime.utcnow(),
                "$push": {
                    "audit_trail.modification_history": {
                        "date": datetime.utcnow(),
                        "action": "MATCHING_COMPLETE",
                        "details": f"Matching completed with {len(case_matches)} potential matches"
                    }
                }
            }
            
            if best_match:
                update_data["confirmation_data.matched_client"] = best_match['customer']
            
            db.asbestosConfirmed.update_one(
                {"_id": case["_id"]},
                {"$set": update_data}
            )
            
            processed_count += 1
            
            # If NO_MATCH, move to archived
            if match_status == "NO_MATCH":
                archive_case(str(case["_id"]), "NO_MATCH")
        
        return jsonify({
            "success": True,
            "message": f"Processed {processed_count} cases",
            "processed_count": processed_count
        })
        
    except Exception as e:
        print(f"Error in run_matching: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
        
        
        
@main_blueprint.route('/api/litigation/process-matches', methods=['POST'])
def process_matches():
    """
    Process matches after confirmation
    """
    try:
        data = request.get_json()
        case_ids = data.get('case_ids', [])
        action = data.get('action', 'CONFIRM')  # CONFIRM or REJECT
        
        processed = 0
        for case_id in case_ids:
            if action == 'CONFIRM':
                # Update case status
                db.asbestosConfirmed.update_one(
                    {"_id": ObjectId(case_id)},
                    {
                        "$set": {
                            "confirmation_data.match_status": "CONFIRMED",
                            "confirmation_data.processing_status": "REVIEWED",
                            "confirmation_data.confirmation_date": datetime.utcnow(),
                            "confirmation_data.confirmed_by": "user",  # Could be from session
                            "audit_trail.last_modified": datetime.utcnow()
                        },
                        "$push": {
                            "audit_trail.modification_history": {
                                "date": datetime.utcnow(),
                                "action": "MATCH_CONFIRMED",
                                "details": "Match confirmed by user"
                            }
                        }
                    }
                )
            else:  # REJECT
                # Archive the case
                archive_case(case_id, "REJECTED_MATCH")
            
            processed += 1
        
        return jsonify({
            "success": True,
            "message": f"Processed {processed} cases",
            "processed_count": processed
        })
        
    except Exception as e:
        print(f"Error in process_matches: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500     



# routes/litigation.py

@main_blueprint.route('/api/litigation/promote', methods=['POST'])
def promote_to_confirmation():
    try:
        print("Starting promotion to confirmation process")
        
        # Get cases from asbestosProduction that need to be processed
        production_cases = list(db.asbestosProduction.find({}))
        
        confirmed_cases = []
        for case in production_cases:
            # Create new document structure
            confirmed_case = {
                # Preserve original data structure
                "file_metadata": case.get("file_metadata", {}),
                "lawsuit_info": case.get("lawsuit_info", {}),
                "counsel": case.get("counsel", []),
                "injured_parties": case.get("injured_parties", []),
                "defendants": case.get("defendants", []),
                
                # Add confirmation-specific fields
                "confirmation_data": {
                    "original_case_id": str(case.get("_id")),
                    "confidence_score": None,  # Will be populated by matching algorithm
                    "match_status": "PENDING",  # PENDING, NO_MATCH, PROBABLE, CONFIRMED
                    "matched_client": None,
                    "confirmation_date": None,
                    "confirmed_by": None,
                    "processing_status": "NEW"  # NEW, PROCESSING, REVIEWED, ARCHIVED
                },
                
                # Add matching results (will be populated later)
                "matching_results": {
                    "best_match": None,
                    "confidence_threshold": 70,
                    "potential_matches": [],
                    "match_date": None
                },
                
                # Add audit trail
                "audit_trail": {
                    "created_at": datetime.utcnow(),
                    "created_by": "system",
                    "last_modified": datetime.utcnow(),
                    "modification_history": [
                        {
                            "date": datetime.utcnow(),
                            "action": "CREATED",
                            "details": "Initial promotion to confirmation queue"
                        }
                    ]
                }
            }
            
            confirmed_cases.append(confirmed_case)
        
        if confirmed_cases:
            # Insert into asbestosConfirmed collection
            result = db.asbestosConfirmed.insert_many(confirmed_cases)
            print(f"Successfully promoted {len(result.inserted_ids)} cases to confirmation")
            
            return jsonify({
                "success": True,
                "message": f"Promoted {len(result.inserted_ids)} cases to confirmation queue",
                "promoted_count": len(result.inserted_ids)
            })
        else:
            return jsonify({
                "success": False,
                "message": "No cases found to promote"
            }), 404
            
    except Exception as e:
        print(f"Error in promote_to_confirmation: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Helper function to archive cases
def archive_case(case_id, reason="NO_MATCH"):
    try:
        # Get the case from asbestosConfirmed
        case = db.asbestosConfirmed.find_one({"_id": ObjectId(case_id)})
        
        if not case:
            return False
            
        # Add archive-specific data
        case["archive_data"] = {
            "archive_date": datetime.utcnow(),
            "archive_reason": reason,
            "archived_by": "system",
            "original_confirmation_data": case.get("confirmation_data", {})
        }
        
        # Insert into archive collection
        db.asbestosArchived.insert_one(case)
        
        # Remove from confirmation collection
        db.asbestosConfirmed.delete_one({"_id": ObjectId(case_id)})
        
        return True
        
    except Exception as e:
        print(f"Error archiving case {case_id}: {str(e)}")
        return False





@main_blueprint.route('/api/litigation/confirmation', methods=['GET'])
def get_confirmation_queue():
    print("\n=== Starting Customer Collection Debug ===")
    
    try:
        # Get access to the customer database collection directly
        core_customers = db.client.customerDatabase.coreCustomers
        print("Attempting to fetch from customerDatabase.coreCustomers collection...")
        
        # Get raw customer documents and process into customer names
        customer_list = list(core_customers.find())
        print(f"\nRaw customer documents found: {len(customer_list)}")
        
        # Initialize customer names list and process all variations
        customer_names = set()  # Using a set for faster lookups and uniqueness
        
        for doc in customer_list:
            # Current name
            if doc.get('current_name'):
                customer_names.add(doc['current_name'])
            
            # Previous names
            if doc.get('previous_names'):
                for prev_name in doc['previous_names']:
                    if isinstance(prev_name, dict) and prev_name.get('name'):
                        customer_names.add(prev_name['name'])
                    elif isinstance(prev_name, str):
                        customer_names.add(prev_name)
            
            # Official name
            if doc.get('official_name'):
                customer_names.add(doc['official_name'])
            
            # Trade names
            if doc.get('trade_names'):
                for trade in doc['trade_names']:
                    if isinstance(trade, dict) and trade.get('name'):
                        customer_names.add(trade['name'])

        # Convert set to list for classifier
        customer_names = list(filter(None, customer_names))
        print(f"\nTotal unique customer names found: {len(customer_names)}")
        
        # Get cases from litigation database
        cases = list(db.asbestosConfirmed.find({
            "confirmation_data.match_status": "PENDING"
        }, {
            "_id": 1,
            "file_metadata.log_number": 1,
            "lawsuit_info.docket_number": 1,
            "lawsuit_info.case_caption": 1,
            "defendants": 1
        }))  # Only fetch the fields we need
        
        print(f"\nFound {len(cases)} total cases to process")
        
        transformed_cases = []
        classifier = Classifier()
        MIN_CONFIDENCE = 75.0
        MAX_MATCHES_PER_CASE = 5  # Limit matches per case for performance
        
        for case in cases:
            identifier = (
                case.get("file_metadata", {}).get("log_number") or 
                case.get("lawsuit_info", {}).get("docket_number")
            )
            
            if not identifier:
                continue
                
            # Get unique defendants
            defendants = list(set(
                d.get("company", "") for d in case.get("defendants", [])
                if d.get("company")
            ))
            
            case_matches = []
            
            for defendant in defendants:
                # Get matches for this defendant
                matches = classifier.find_matches(
                    defendant=defendant,
                    customer_list=customer_names
                )
                
                # Filter and limit matches
                valid_matches = [
                    match for match in matches 
                    if match['confidence'] >= MIN_CONFIDENCE
                ][:MAX_MATCHES_PER_CASE]
                
                for match in valid_matches:
                    case_matches.append({
                        "case_id": str(case["_id"]),
                        "log_number": identifier,
                        "case_caption": case.get("lawsuit_info", {}).get("case_caption", ""),
                        "confidence_score": f"{match['confidence']:.1f}%",
                        "matched_client": match['customer'],
                        "matched_defendant": defendant,
                        "match_details": match['match_details']
                    })
            
            # If we found matches, sort them and update the case
            if case_matches:
                # Sort matches by confidence
                case_matches.sort(
                    key=lambda x: float(x['confidence_score'].rstrip('%')), 
                    reverse=True
                )
                
                # Take top matches
                top_matches = case_matches[:MAX_MATCHES_PER_CASE]
                transformed_cases.extend(top_matches)
                
                # Update MongoDB with best match only
                best_match = top_matches[0]
                db.asbestosConfirmed.update_one(
                    {"_id": case["_id"]},
                    {"$set": {
                        "matching_results": {
                            "confidence_score": float(best_match['confidence_score'].rstrip('%')),
                            "best_match": best_match['matched_client'],
                            "matched_defendant": best_match['matched_defendant'],
                            "match_date": datetime.utcnow(),
                            "match_details": best_match['match_details']
                        }
                    }}
                )

        # Sort all cases by confidence
        transformed_cases.sort(
            key=lambda x: float(x['confidence_score'].rstrip('%')), 
            reverse=True
        )
        
        print(f"\nFound {len(transformed_cases)} valid matches across {len(cases)} cases")
        
        response_data = {
            "success": True,
            "count": len(transformed_cases),
            "cases": transformed_cases
        }
        
        return jsonify(response_data)
            
    except Exception as e:
        print(f"\nError in processing: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Error processing data: {str(e)}"
        }), 500














        
@main_blueprint.route('/api/litigation/archived-queue', methods=['GET'])
def get_archived_queue():
    try:
        # Get archived cases (<75% confidence)
        cases = list(db.asbestosArchived.find().sort("archive_data.archive_date", -1))
        
        transformed_cases = []
        for case in cases:
            transformed_case = {
                "case_id": str(case["_id"]),
                "log_number": case.get("log_number"),
                "case_caption": case.get("case_caption"),
                "confidence_score": case.get("matching_results", {}).get("confidence_score"),
                "archive_date": case.get("archive_data", {}).get("archive_date"),
                "archive_reason": case.get("archive_data", {}).get("archive_reason"),
                "jurisdiction": case.get("lawsuit_info", {}).get("jurisdiction"),
                "defendants": case.get("defendants", [])
            }
            transformed_cases.append(transformed_case)
            
        return jsonify({
            "success": True,
            "count": len(transformed_cases),
            "cases": transformed_cases
        })
        
    except Exception as e:
        print(f"Error getting archived queue: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500  








@main_blueprint.route('/api/litigation/promote-to-client', methods=['POST'])
def promote_to_client():
    try:
        data = request.get_json()
        case_strings = data.get('cases', [])
        
        if not case_strings:
            return jsonify({
                "success": False,
                "error": "No cases provided for promotion"
            }), 400
            
        print(f"Promoting {len(case_strings)} cases to client table")
        promoted_count = 0
        
        for case_string in case_strings:
            # Parse the combined string back into log_number and defendant
            try:
                log_number, defendant = case_string.split('|||')
            except ValueError:
                print(f"Invalid case string format: {case_string}")
                continue
            
            print(f"Processing log_number: {log_number}, defendant: {defendant}")
            
            # Find the case in asbestosConfirmed
            case_doc = db.asbestosConfirmed.find_one({
                "$or": [
                    {"file_metadata.log_number": log_number},
                    {"lawsuit_info.docket_number": log_number}
                ]
            })
            
            if case_doc:
                # Get the matching results
                matching_results = case_doc.get('matching_results', {})
                
                if matching_results:
                    # Create client table entry
                    client_entry = {
                        "log_number": log_number,
                        "case_caption": case_doc.get("lawsuit_info", {}).get("case_caption"),
                        "client": matching_results.get("best_match"),
                        "matched_defendant": defendant,
                        "confidence_score": matching_results.get("confidence_score"),
                        "promotion_date": datetime.utcnow(),
                        "original_case_id": str(case_doc["_id"]),
                        "lawsuit_info": case_doc.get("lawsuit_info", {}),
                        "file_metadata": case_doc.get("file_metadata", {}),
                        "defendants": case_doc.get("defendants", []),
                        "confirmation_data": {
                            "promotion_date": datetime.utcnow(),
                            "promoted_by": "system",
                            "original_match_results": matching_results
                        }
                    }
                    
                    try:
                        # Insert into asbestosClient
                        result = db.asbestosClient.insert_one(client_entry)
                        
                        if result.inserted_id:
                            # Remove from asbestosConfirmed only if successfully inserted
                            db.asbestosConfirmed.delete_one({"_id": case_doc["_id"]})
                            promoted_count += 1
                            print(f"Successfully promoted case {log_number} to client table")
                        
                    except Exception as insert_error:
                        print(f"Error inserting case {log_number}: {str(insert_error)}")
                        continue
        
        if promoted_count > 0:
            return jsonify({
                "success": True,
                "promoted_count": promoted_count,
                "message": f"Successfully promoted {promoted_count} cases to client table"
            })
        else:
            return jsonify({
                "success": False,
                "error": "No cases were successfully promoted"
            }), 400
        
    except Exception as e:
        print(f"Error promoting cases: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500






















@main_blueprint.route('/api/litigation/clear-confirmed', methods=['POST'])
def clear_confirmed_cases():
    try:
        # Get initial count
        initial_count = db.asbestosConfirmed.count_documents({})
        print(f"\nStarting clear operation. Current document count: {initial_count}")
        
        # Delete all documents from the collection
        result = db.asbestosConfirmed.delete_many({})
        
        # Verify deletion
        remaining_count = db.asbestosConfirmed.count_documents({})
        
        print(f"Deleted {result.deleted_count} documents")
        print(f"Remaining documents: {remaining_count}")
        
        return jsonify({
            "success": True,
            "deleted_count": result.deleted_count,
            "remaining_count": remaining_count
        })
        
    except Exception as e:
        print(f"Error clearing confirmed cases: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@main_blueprint.route('/api/prototyping/chat', methods=['POST'])
def chat_with_aida():
    try:
        data = request.get_json()
        user_message = data.get('message')
        conversation_history = data.get('history', [])
        current_stage = data.get('currentStage')
        selected_pattern = data.get('pattern')

        print("Received message:", user_message)
        print("Current stage:", current_stage)
        print("Selected pattern:", selected_pattern)

        # Build system prompt based on stage and pattern
        base_system_prompt = """You are AIDA (AI Design Assistant). You help users create structured workflows 
        based on established enterprise patterns.

        When analyzing workflow requirements, always structure your response as follows:

        Step X: [Descriptive Name]
        Components:
        - Primary: [Component Type] - [Purpose]
        - Secondary: [Component Type] - [Purpose]

        Data Requirements:
        - Data Sources: [Required Data Sources/APIs]
        - State Management: [State Requirements]

        User Flow:
        - Entry Point: [How users start this step]
        - Actions: [What users/system can do]
        - Exit Conditions: [When step is complete]

        Available Components:
        - Form: Input collection with validation
        - Table: Data display with sorting/filtering
        - Modal: Pop-up dialogs and forms
        - FileUpload: Document handling
        - Calendar: Date and scheduling
        - Kanban: Status and progress tracking
        - Timeline: Process visualization
        - Chart: Data visualization
        - Alert: User notifications
        """

        if selected_pattern:
            pattern_context = f"""
            The user has selected the {selected_pattern.get('name')} pattern, which is designed for 
            {selected_pattern.get('purpose')}.

            Key Pattern Features:
            {', '.join(selected_pattern.get('key_features', []))}

            Common Use Cases:
            {', '.join(selected_pattern.get('use_cases', []))}

            Please ensure the workflow structure follows {selected_pattern.get('name')} pattern best practices
            and leverages its core capabilities.
            """
            system_prompt = base_system_prompt + "\n\n" + pattern_context
        else:
            system_prompt = base_system_prompt

        # Format conversation history for Claude
        formatted_history = []
        for msg in conversation_history:
            role = "assistant" if msg['type'] == 'assistant' else "user"
            formatted_history.append({
                "role": role,
                "content": msg['content']
            })

        # Add the new user message
        formatted_history.append({
            "role": "user",
            "content": user_message
        })

        try:
            # Get response from Claude
            response = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.7,
                system=system_prompt,
                messages=formatted_history
            )

            if response and hasattr(response, 'content') and response.content:
                response_text = response.content[0].text
                print("Claude response:", response_text)
                
                # If we're in specification stage, ensure response is structured
                if current_stage == 'specification' and not any(line.startswith('Step') for line in response_text.split('\n')):
                    return jsonify({
                        "error": "Response not properly structured. Please try again."
                    }), 400

                return jsonify({
                    "response": response_text
                })
            else:
                raise Exception("No valid response content from Claude")

        except Exception as e:
            print(f"Error from Claude API: {str(e)}")
            raise

    except Exception as e:
        current_app.logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500


def validate_response_structure(response_text):
    """
    Validates that the response follows our required structure
    """
    required_sections = ['Step', 'Components', 'Data Requirements', 'User Flow']
    lines = response_text.split('\n')
    
    return all(any(section in line for line in lines) for section in required_sections)



















@main_blueprint.route('/api/prototyping/promote-to-canvas', methods=['POST'])
def promote_to_canvas():
    try:
        data = request.get_json()
        workflow_text = data.get('workflow_text')
        
        if not workflow_text:
            return jsonify({'error': 'No workflow text provided'}), 400
            
        parser = TooljetParser()
        canvas_json = parser.parse_workflow_text(workflow_text)
        
        # Here you could either:
        # 1. Return the JSON for the frontend to handle
        # 2. Make a direct API call to Tooljet to create the canvas
        # 3. Store the JSON in your database for later use
        
        return jsonify({
            'success': True,
            'canvas_json': canvas_json
        })
        
    except Exception as e:
        current_app.logger.error(f"Error converting to canvas: {str(e)}")
        return jsonify({'error': str(e)}), 500










    
    
    
    




@main_blueprint.route('/api/programs/winslow', methods=['GET'])
def get_winslow_programs():
    try:
       
      
        react_env = request.headers.get('X-Environment', 'development')
        if react_env == 'development':
            # Local file path construction
            json_path = os.path.abspath(os.path.join(
                current_app.root_path,
                '..',
                '..',
                'static',
                'data',
                'hawkeye',
                'winslow',
                'winslow_programs.json'
            ))
            print(f"\n🔍 Using local path: {json_path}")
            print(f"Does path exist? {os.path.exists(json_path)}")
            
            with open(json_path, 'r') as f:
                programs_data = json.load(f)
        else:
            # S3 path construction
            s3_key = f'data/programs/hawkeye/winslow/winslow_programs.json'
            print(f"\n🔍 Using S3 path: {s3_key}")
            
            s3_client = boto3.client('s3')
            response = s3_client.get_object(
                Bucket='juniperproductiondata',
                Key=s3_key
            )
            programs_data = json.loads(response['Body'].read().decode('utf-8'))
            
        return jsonify(programs_data)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': 'Failed to load program data'}), 500
















# Internal functions for scanning
def scan_source_orders_internal():
    """Internal function to scan order templates"""
    try:
        # Construct paths
        base_path = Path(current_app.root_path).parent.parent / 'static' / 'data'
        registry_path = base_path / 'SystemITTemplates' / 'TangibleITTemplates' / 'TemplateRegistry' / 'tangibletemplateregistry.json'
        source_orders_path = base_path / 'SystemITTemplates' / 'TangibleITTemplates' / 'Source' / 'Orders'
        foundational_orders_path = base_path / 'SystemITTemplates' / 'TangibleITTemplates' / 'Foundational' / 'Orders'

        # Load existing registry
        with open(registry_path, 'r') as f:
            registry_data = json.load(f)

        # Find sourcing group and order agreement type
        sourcing_group = next(
            (group for group in registry_data['programGroups']
             if group['id'] == 'sourcing'),
            None
        )

        if not sourcing_group:
            return False

        # Find order agreement type
        order_type = next(
            (atype for atype in sourcing_group['agreementTypes']
             if atype['id'] == 'order'),
            None
        )

        if not order_type:
            return False

        # Get list of actual files in both directories
        existing_source_files = set()
        if source_orders_path.exists():
            existing_source_files = {
                f.stem for f in source_orders_path.glob('*.docx')
                if not f.name.startswith('~')
            }

        existing_foundational_files = set()
        if foundational_orders_path.exists():
            existing_foundational_files = {
                f.stem for f in foundational_orders_path.glob('*.docx')
                if not f.name.startswith('~')
            }

        # Track changes
        for program_class in order_type.get('programClasses', []):
            for form in program_class.get('forms', []):
                old_status = form.get('status')
                
                # Check existence in both directories
                source_filename = Path(form['sourceTemplatePath']).stem
                foundational_filename = Path(form['templatePath']).stem
                
                source_exists = source_filename in existing_source_files
                foundational_exists = foundational_filename in existing_foundational_files
                
                # Update existence flags
                form['sourceFileExists'] = source_exists
                form['fileExists'] = foundational_exists
                
                # Set status based on file existence
                new_status = 'foundational' if foundational_exists else 'source' if source_exists else 'missing'
                form['status'] = new_status
                
                        # Add this section to update conversionState
                if source_exists:
                    form['conversionState'] = 'SOURCE'
                    form['lastConverted'] = None
                elif foundational_exists:
                    form['conversionState'] = 'CONVERTED'
                    form['lastConverted'] = datetime.now().isoformat()

        # Update timestamp
        registry_data['lastUpdated'] = datetime.now().isoformat()

        # Save updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)

        return True

    except Exception as e:
        print(f"❌ Error scanning orders: {str(e)}")
        return False

def scan_source_parents_internal():
    """Internal function to scan parent templates"""
    try:
        # Construct paths
        base_path = Path(current_app.root_path).parent.parent / 'static' / 'data'
        registry_path = base_path / 'SystemITTemplates' / 'TangibleITTemplates' / 'TemplateRegistry' / 'tangibletemplateregistry.json'
        source_parents_path = base_path / 'SystemITTemplates' / 'TangibleITTemplates' / 'Source' / 'Parents'
        foundational_parents_path = base_path / 'SystemITTemplates' / 'TangibleITTemplates' / 'Foundational' / 'Parents'

        # Load existing registry
        with open(registry_path, 'r') as f:
            registry_data = json.load(f)

        # Find sourcing group and parent agreement type
        sourcing_group = next(
            (group for group in registry_data['programGroups']
             if group['id'] == 'sourcing'),
            None
        )

        if not sourcing_group:
            return False

        # Find parent agreement type
        parent_type = next(
            (atype for atype in sourcing_group['agreementTypes']
             if atype['id'] == 'parent'),
            None
        )

        if not parent_type:
            return False

        # Get list of actual files in both directories
        existing_source_files = set()
        if source_parents_path.exists():
            existing_source_files = {
                f.stem for f in source_parents_path.glob('*.docx')
                if not f.name.startswith('~')
            }

        existing_foundational_files = set()
        if foundational_parents_path.exists():
            existing_foundational_files = {
                f.stem for f in foundational_parents_path.glob('*.docx')
                if not f.name.startswith('~')
            }

        # Check each program class and form in registry
        for program_class in parent_type.get('programClasses', []):
            for form in program_class.get('forms', []):
                old_status = form.get('status')
                
                # Check existence in both directories
                source_filename = Path(form['sourceTemplatePath']).stem
                foundational_filename = Path(form['templatePath']).stem
                
                source_exists = source_filename in existing_source_files
                foundational_exists = foundational_filename in existing_foundational_files
                
                # Update existence flags
                form['sourceFileExists'] = source_exists
                form['fileExists'] = foundational_exists
                
                # Set status based on file existence
                new_status = 'foundational' if foundational_exists else 'source' if source_exists else 'missing'
                form['status'] = new_status

                        # Add this section to update conversionState
                if source_exists:
                    form['conversionState'] = 'SOURCE'
                    form['lastConverted'] = None
                elif foundational_exists:
                    form['conversionState'] = 'CONVERTED'
                    form['lastConverted'] = datetime.now().isoformat()



        # Update timestamp
        registry_data['lastUpdated'] = datetime.now().isoformat()

        # Save updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)

        return True

    except Exception as e:
        print(f"❌ Error scanning parents: {str(e)}")
        return False

# Routes for external API access
@main_blueprint.route('/api/scan-source-orders', methods=['GET'])
def scan_source_orders():
    """Route endpoint for scanning orders"""
    success = scan_source_orders_internal()
    if success:
        return jsonify({
            'success': True,
            'message': 'Order templates scanned successfully'
        })
    return jsonify({
        'success': False,
        'error': 'Failed to scan order templates'
    }), 500

@main_blueprint.route('/api/scan-source-parents', methods=['GET'])
def scan_source_parents():
    """Route endpoint for scanning parents"""
    success = scan_source_parents_internal()
    if success:
        return jsonify({
            'success': True,
            'message': 'Parent templates scanned successfully'
        })
    return jsonify({
        'success': False,
        'error': 'Failed to scan parent templates'
    }), 500

   
 
 
 
@main_blueprint.route('/api/convert_template', methods=['POST'])
def convert_template_api():
    """API endpoint to convert a template and update registry"""
    try:
        # Initialize COM for the API thread
        pythoncom.CoInitialize()
        
        data = request.get_json()
        
        if not data or 'template' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing template information'
            }), 400
        
        template_info = data['template']
        source_path = template_info.get('sourceTemplatePath')
        
        if not source_path:
            return jsonify({
                'success': False,
                'error': 'Missing source template path'
            }), 400

        # Construct paths
        base_dir = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'SystemITTemplates',
            'TangibleITTemplates'
        ))

        source_full_path = os.path.join(base_dir, 'Source', 
            'Orders' if 'Orders' in source_path else 'Parents',
            os.path.basename(source_path))
            
        target_path = os.path.join(base_dir, 'Foundational',
            'Orders' if 'Orders' in source_path else 'Parents',
            os.path.basename(source_path))

        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        print(f"Beginning template conversion:")
        print(f"  Source: {source_full_path}")
        print(f"  Target: {target_path}")

        result_path = template_converter.convert_template(
            source_full_path,
            target_path
        )

        # Update registry (assuming conversion succeeded)
        registry_path = os.path.join(base_dir, 'TemplateRegistry', 'tangibletemplateregistry.json')
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
            
        # Update registry entry
        for group in registry.get('programGroups', []):
            for atype in group.get('agreementTypes', []):
                for pclass in atype.get('programClasses', []):
                    for form in pclass.get('forms', []):
                        if form.get('sourceTemplatePath') == source_path:
                            form['conversionState'] = 'CONVERTED'
                            form['lastConverted'] = datetime.now().isoformat()
                            form['fileExists'] = True
                            break

        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

        return jsonify({
            'success': True,
            'message': 'Template converted successfully',
            'path': result_path
        })

    except Exception as e:
        print(f"❌ Conversion error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
        
    finally:
        try:
            pythoncom.CoUninitialize()
        except:
            pass
 
 
 
   
   
   
   
     


@main_blueprint.route('/api/conversion_status/<job_id>', methods=['GET'])
def check_conversion_status(job_id):
    """
    API endpoint to check the status of a conversion job
    """
    status_dir = os.path.join(current_app.instance_path, 'conversion_status')
    status_file = os.path.join(status_dir, f"{job_id}.json")
    
    if not os.path.exists(status_file):
        return jsonify({
            'success': False,
            'error': f'No status found for job {job_id}'
        }), 404
    
    try:
        with open(status_file, 'r') as f:
            status_data = json.load(f)
            return jsonify({
                'success': True,
                'job_id': job_id,
                'status': status_data.get('status', 'unknown'),
                'message': status_data.get('last_message', ''),
                'log': status_data.get('log', []),
                'output_file': status_data.get('output_file')
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error reading status: {str(e)}'
        }), 500
        
        
        
        
        
        
        
        
        
        
        
        
@main_blueprint.route('/api7/files/view', methods=['GET'])
def view_file97():
    """
    Endpoint to view or download files
    """
    file_path = request.args.get('path')
    
    if not file_path:
        return jsonify({
            'success': False,
            'error': 'No file path provided'
        }), 400
    
    # Security check - ensure the path is within allowed directories
    allowed_dirs = [
        current_app.config.get('UPLOAD_FOLDER', ''),
        current_app.config.get('TEMPLATE_FOLDER', ''),
        current_app.instance_path,
        # Add any other allowed directories here
    ]
    
    # Normalize path and check if it's within allowed directories
    file_path = os.path.abspath(file_path)
    if not any(file_path.startswith(os.path.abspath(allowed_dir)) for allowed_dir in allowed_dirs if allowed_dir):
        return jsonify({
            'success': False,
            'error': 'Access to this file is not allowed'
        }), 403
    
    if not os.path.exists(file_path):
        return jsonify({
            'success': False,
            'error': 'File not found'
        }), 404
    
    # Determine if we should force download or try to display in browser
    download = request.args.get('download', 'false').lower() == 'true'
    filename = os.path.basename(file_path)
    
    try:
        return send_file(
            file_path,
            as_attachment=download,
            download_name=filename if download else None
        )
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error accessing file: {str(e)}'
        }), 500  
        
        
        
        
        
        
        
        
        

@main_blueprint.route('/api/files/view', methods=['GET'])
def view_files():
    """Simple endpoint to read and return file content"""
   
    
    file_path = request.args.get('path')
    print(file_path)
    
    if not file_path:
        return jsonify({
            'success': False,
            'error': 'No file path provided'
        }), 400
    
    allowed_dirs = [
        current_app.config.get('UPLOAD_FOLDER', ''),
        current_app.config.get('TEMPLATE_FOLDER', ''),
        'C:/Users/RobertReynolds/Python Projects/Python Development Projects/TransactionPlatformDevelopmentAlphaLocal/transaction_platform_app/static/data/SystemITTemplates'
    ]
    
    file_path = os.path.abspath(file_path)
    if not any(file_path.startswith(os.path.abspath(allowed_dir)) 
               for allowed_dir in allowed_dirs if allowed_dir):
        return jsonify({
            'success': False,
            'error': 'Access to this file is not allowed'
        }), 403
    
    if not os.path.exists(file_path):
        return jsonify({
            'success': False,
            'error': 'File not found'
        }), 404
        
    try:
        # Handle based on file type
        if file_path.lower().endswith('.docx'):
            with open(file_path, 'rb') as docx_file:
                result = mammoth.convert_to_html(docx_file)
                html = result.value
            return jsonify({
                'success': True,
                'content': html,
                'type': 'docx'
            })
        else:
            # For text files
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({
                'success': True,
                'content': content,
                'type': 'text'
            })
        
    except Exception as e:
        print(f"Error reading file: {str(e)}")  # Log the actual error
        return jsonify({
            'success': False,
            'error': f'Error reading file: {str(e)}'
        }), 500

























@main_blueprint.route('/api/templates', methods=['POST'])
def save_configured_template():
    try:
        data = request.get_json()
        program_class = data.get('programClass')
        customer_name = data.get('customerConstants', {}).get('customerName', 'unknown')
        
        base_dir = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'TangibleITTemplates',
            'Configured'
        ))

        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        template_filename = f"{customer_name}_{program_class}_configured_{timestamp}.docx"  # Changed to .docx
        template_path = os.path.join(base_dir, template_filename)

        print(f"💾 Saving configured template to: {template_path}")

        processor = DocumentProcessor()
        doc = processor.process_template(
            data.get('template_path'),
            data.get('variables', {})
        )
        doc.save(template_path)

        metadata_filename = f"{template_filename}.meta.json"
        metadata_path = os.path.join(base_dir, metadata_filename)
        
        with open(metadata_path, 'w', encoding='utf-8') as file:
            json.dump({
                'program_class': program_class,
                'customer_constants': data.get('customerConstants', {}),
                'metadata': data.get('metadata', {}),
                'created_at': datetime.now().isoformat(),
                'template_path': template_path,
                'variables_used': data.get('variables', {})
            }, file, indent=2)

        return jsonify({
            'message': 'Template saved successfully',
            'template_path': template_path,
            'metadata_path': metadata_path
        })

    except Exception as e:
        print(f"❌ Error saving template: {str(e)}")
        return jsonify({'error': str(e)}), 500




    
        

@main_blueprint.route('/api/housetemplates', methods=['GET'])
def list_house_templates():
    try:
        foundation = request.args.get('foundation')
        program_class = request.args.get('program_class')
        
        base_dir = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'TangibleITTemplates'
        ))

        templates = []
        for filename in os.listdir(base_dir):
            if filename.endswith('_template.docx'):
                template_info = {
                    'filename': filename,
                    'program_class': filename.split('_')[0],
                    'foundation': filename.split('_')[1],
                    'type': 'house',  # Explicitly mark as house template
                    'available_keys': [  # List available keys for each template
                        '[Customer]',
                        '[Customer Address]',
                        '[Governing Law]'
                    ]
                }
                
                if foundation and template_info['foundation'] != foundation:
                    continue
                if program_class and template_info['program_class'] != program_class:
                    continue
                    
                templates.append(template_info)

        return jsonify({
            'templates': templates
        })

    except Exception as e:
        print(f"❌ Error listing house templates: {str(e)}")
        return jsonify({'error': str(e)}), 500

# New separate route for PDF conversion
@main_blueprint.route('/api/housetemplates/convert-to-pdf', methods=['POST'])
def convert_to_pdf():
    try:
        data = request.get_json()
        docx_path = data.get('docx_path')
        
        if not docx_path or not os.path.exists(docx_path):
            return jsonify({'error': 'DOCX file not found'}), 404
            
        # Create PDF path
        pdf_path = docx_path.replace('.docx', '.pdf')
        
        print(f"\n--- Starting PDF Conversion ---")
        print(f"Source DOCX: {docx_path}")
        print(f"Target PDF: {pdf_path}")
        
        # Convert to PDF
        convert(docx_path, pdf_path)
        
        print("PDF conversion complete")
        
        return jsonify({
            'success': True,
            'message': 'PDF created successfully',
            'pdf_path': pdf_path
        })

    except Exception as e:
        print(f"\n❌ Error converting to PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Route to serve the PDF
@main_blueprint.route('/api/housetemplates/pdf/<platform_name>/<filename>', methods=['GET'])
def get_template_pdf(platform_name, filename):
    try:
        # Make sure we're looking for a PDF
        if not filename.endswith('.pdf'):
            filename = filename.replace('.docx', '.pdf')
        
        base_dir = os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'CustomerTemplates',
            platform_name,
            'ITTemplates'
        )
        
        pdf_path = os.path.join(base_dir, filename)
        print(f"\n--- PDF Request Details ---")
        print(f"Looking for PDF at: {pdf_path}")
        print(f"File exists: {os.path.exists(pdf_path)}")
        
        if os.path.exists(pdf_path):
            return send_file(pdf_path, mimetype='application/pdf')
        else:
            # List directory contents for debugging
            dir_path = os.path.dirname(pdf_path)
            if os.path.exists(dir_path):
                print(f"Directory contents of {dir_path}:")
                print(os.listdir(dir_path))
            return jsonify({'error': 'PDF not found'}), 404
            
    except Exception as e:
        print(f"Error serving PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    
    
    
    
    
@main_blueprint.route('/api/housetemplates', methods=['POST'])
def save_house_template():
    try:
        print("\n--- Starting Template Processing ---")
        data = request.get_json()
        
        # Extract key information
        program_class = data.get('programClass')
        customer_constants = data.get('customerConstants', {})
        platform_name = customer_constants.get('platformName', 'unknown')
        
        print("\n--- Customer Constants Received ---")
        print(json.dumps(customer_constants, indent=2))
        
        # Variables for replacement
        template_variables = {
            'Customer': customer_constants.get('customerName'),  # For {{Customer}}
            'CustomerAddress': customer_constants.get('customerAddress'),  # For {{CustomerAddress}}
            'GoverningLaw': customer_constants.get('governingLaw'),  # For {{Governing Law}}
            'EffectiveDate': datetime.now().strftime('%B %d, %Y'),  # For {{Effective Date}}
            'CustomerName': customer_constants.get('customerName'),  # For {{CustomerName}}
            'CUSTOMERSIGNATUREBLOCK': customer_constants.get('customerSignatureBlock')  # For {{CUSTOMERSIGNATUREBLOCK}}
                }
        
        print("\n--- Template Variables Prepared ---")
        print(json.dumps(template_variables, indent=2))
        
        # Build paths
        base_dir = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'CustomerTemplates',
            platform_name,
            'ITTemplates'
        ))
        os.makedirs(base_dir, exist_ok=True)
        
        # Source template path
        source_template = os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
            'TangibleITTemplates',
            f"{program_class}_tangible_template.docx"
        )
        
        # Generate new filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{platform_name}_{program_class}_{timestamp}.docx"
        output_path = os.path.join(base_dir, output_filename)
        
        # Process template
        processor = DocumentProcessor()
        processed_doc = processor.process_template(source_template, template_variables)
        processed_doc.save(output_path)
        
        return jsonify({
            'success': True,
            'message': 'Template processed successfully',
            'saved_path': output_path
        })
        
    except Exception as e:
        print(f"\n❌ Error processing template: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    
    

@main_blueprint.route('/api/template/registry/tangible', methods=['GET'])
def get_template_registry():
    try:
        orders_success = scan_source_orders_internal()
        parents_success = scan_source_parents_internal()

        if not orders_success or not parents_success:
            raise Exception("Failed to update registry")

        base_path = Path(current_app.root_path).parent.parent / 'static' / 'data'
        registry_path = base_path / 'SystemITTemplates' / 'TangibleITTemplates' / 'TemplateRegistry' / 'tangibletemplateregistry.json'

        with open(registry_path, 'r') as f:
            registry_data = json.load(f)

        # Return just the registry_data, not wrapped in success
        return jsonify(registry_data)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500
        
@main_blueprint.route('/api/templateforms/<path:template_path>', methods=['GET'])
def get_template_content(template_path):
    """Route for getting template content"""
    try:
        # Clean up the template path and ensure proper directory structure
      
        
        # Construct base path to static/data directory
        base_dir = os.path.abspath(os.path.join(
            current_app.root_path,
            '..',
            '..',
            'static',
            'data',
           
        ))

        # Construct the full path
        full_path = os.path.join(base_dir, template_path)
        print(full_path)

        print(f"🔍 Template request details:")
        print(f"  Base directory: {base_dir}")
        print(f"  Template path: {template_path}")
        print(f"  Full path: {full_path}")

        # Validate path is within allowed directory
        if not str(full_path).startswith(str(base_dir)):
            raise Exception(f"Invalid template path - must be within {base_dir}")

        # Check if file exists
        if not os.path.exists(full_path):
            print(f"❌ File not found at: {full_path}")
            return jsonify({
                'error': 'Template file not found'
            }), 404

        print(f"✅ Found template at: {full_path}")

        # Read the file content
        with open(full_path, 'rb') as f:
            content = f.read()
            
        # Convert to base64
        import base64
        content_b64 = base64.b64encode(content).decode('utf-8')
            
        return jsonify({
            'content': content_b64,
            'filename': os.path.basename(template_path)
        })

    except Exception as e:
        print(f"❌ Error reading template: {str(e)}")
        print(f"  Attempted path: {full_path if 'full_path' in locals() else 'not constructed'}")
        return jsonify({
            'error': str(e)
        }), 500  

def get_template_paths():
    """Get template paths using current_app context"""
    base_path = os.path.join(
        current_app.root_path,
        '..',
        '..',
        'static',
        'data',
        'SystemITTemplates',
        'TangibleITTemplates'
    )
    
    return {
        'BASE_PATH': base_path,
        'PARENT_TEMPLATES': os.path.join(base_path, 'Foundational', 'Parents'),
        'ORDER_TEMPLATES': os.path.join(base_path, 'Foundational', 'Orders')
    }
@main_blueprint.route('/api/fetch-template', methods=['POST'])
def fetch_template():
    """Flexible route to fetch any template from the template directory"""
    try:
        template_paths = get_template_paths()  # Get paths within route context
        
        # Get template path from request
        data = request.get_json()
        if not data or 'templatePath' not in data:
            return jsonify({'error': 'No template path provided'}), 400
            
        relative_path = data['templatePath']
        
        # Construct full path
        full_path = os.path.join(template_paths['BASE_PATH'], relative_path)
        
        # Debug prints
        print(f"🔍 Template request details:")
        print(f"  Base path: {template_paths['BASE_PATH']}")
        print(f"  Relative path: {relative_path}")
        print(f"  Full path: {full_path}")
        
        # Check if file exists
        if not os.path.exists(full_path):
            print(f"❌ Template not found at: {full_path}")
            return jsonify({'error': 'Template not found'}), 404
            
        print(f"✅ Found template at: {full_path}")
        
        # Read the file content
        with open(full_path, 'rb') as f:
            content = f.read()
            
        # Convert to base64
        content_b64 = base64.b64encode(content).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'content': content_b64,
            'filename': os.path.basename(full_path)
        })
        
    except Exception as e:
        print(f"❌ Error fetching template: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_template(template_path, substitutions):
    """
    Process a template with the given substitutions
    
    Args:
        template_path (str): Full path to the template file
        substitutions (dict): Dictionary of variables to substitute
        
    Returns:
        tuple: (processed_content_base64, filename)
    """
    try:
        print(f"🔄 Processing template at: {template_path}")
        print(f"📝 With substitutions:", substitutions)
        
        # Load the template
        doc = DocxTemplate(template_path)
        
        # Render with substitutions
        doc.render(substitutions)
        
        # Save to memory
        output = BytesIO()
        doc.save(output)
        output.seek(0)
        
        # Convert to base64
        content_b64 = base64.b64encode(output.read()).decode('utf-8')
        
        print("✅ Template processing complete")
        return content_b64, os.path.basename(template_path)
        
    except Exception as e:
        print(f"❌ Error processing template: {str(e)}")
        raise

@main_blueprint.route('/api/process-template', methods=['POST'])
def process_template_route():
    """Route to process template with substitutions"""
    try:
        template_paths = get_template_paths()
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        print('📦 Received process-template request:', data)
        
        # Get template path and substitutions
        template_path = data.get('templatePath')
        substitutions = data.get('substitutions', {})
        
        if not template_path:
            return jsonify({'error': 'No template path provided'}), 400
            
        # Construct full path
        full_path = os.path.join(template_paths['BASE_PATH'], template_path)
        
        if not os.path.exists(full_path):
            return jsonify({'error': 'Template not found'}), 404
            
        # Process the template
        content_b64, filename = process_template(full_path, substitutions)
        
        return jsonify({
            'status': 'success',
            'document': {
                'filename': filename,
                'content': content_b64
            }
        })
        
    except Exception as e:
        print(f"❌ Error in process-template route: {str(e)}")
        return jsonify({'error': str(e)}), 500
