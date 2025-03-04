# app/routes/dashboard_package.py
from flask import Blueprint, request, jsonify, current_app
import json
import os
import psycopg2
import datetime
from psycopg2.extras import DictCursor
from dotenv import load_dotenv

# Load environment variables
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
print("==== Dashboard Package PostgreSQL Connection Parameters ====")
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

# Create a Blueprint for dashboard routes - note the different name!
dashboard_package = Blueprint('dashboard', __name__, url_prefix='/api/dashboard')

# =========================================
# Routes for fetching dimensions
# =========================================

@dashboard_package.route('/dimensions', methods=['GET'])
def get_dimensions():
    """
    Get dimensions based on view status (on/off).
    
    Query parameters:
    - view: 'on' for current dimensions, 'off' for suggested dimensions
    """
    try:
        view_status = request.args.get('view', 'on')  # Default to 'on'
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
            
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Query dimensions based on view status
        cursor.execute("""
            SELECT id, dimension_id, display_name, description, program_type, order_num 
            FROM acquisitiondimensions 
            WHERE hawkeyeview = %s
            ORDER BY order_num
        """, (view_status,))
        
        dimensions = cursor.fetchall()
        
        # Format dimensions for the response
        formatted_dimensions = []
        for dim in dimensions:
            # Fetch attributes for this dimension
            cursor.execute("""
                SELECT id, attribute_id, display_name, data_type, required, max_length, order_num 
                FROM acquisitionattributes 
                WHERE dimension_id = %s
                ORDER BY order_num
            """, (dim['id'],))
            
            attributes = cursor.fetchall()
            
            # Format attributes
            formatted_attributes = []
            for attr in attributes:
                formatted_attributes.append({
                    "id": attr['id'],
                    "attributeId": attr['attribute_id'],
                    "displayName": attr['display_name'],
                    "dataType": attr['data_type'],
                    "required": attr['required'],
                    "maxLength": attr['max_length'],
                    "order": attr['order_num']
                })
            
            # Add dimension with its attributes
            formatted_dimensions.append({
                "id": dim['id'],
                "dimensionId": dim['dimension_id'],
                "displayName": dim['display_name'],
                "description": dim['description'],
                "programType": dim['program_type'],
                "order": dim['order_num'],
                "attributeCount": len(formatted_attributes),
                "attributes": formatted_attributes
            })
        
        return jsonify({"dimensions": formatted_dimensions})
        
    except Exception as e:
        current_app.logger.error(f"Error fetching dimensions: {str(e)}")
        return jsonify({"error": f"Error fetching dimensions: {str(e)}"}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@dashboard_package.route('/dimensions/update-view', methods=['POST'])
def update_dimensions_view():
    """
    Update the view status of dimensions.
    
    Expected JSON payload:
    {
        "dimensionIds": [1, 2, 3],
        "view": "on"  // or "off"
    }
    """
    try:
        data = request.json
        dimension_ids = data.get('dimensionIds', [])
        view_status = data.get('view', 'on')
        
        if not dimension_ids:
            return jsonify({"error": "No dimension IDs provided"}), 400
            
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
            
        cursor = conn.cursor()
        
        # Update dimensions
        for dim_id in dimension_ids:
            cursor.execute("""
                UPDATE acquisitiondimensions 
                SET hawkeyeview = %s
                WHERE id = %s
            """, (view_status, dim_id))
            
            # Also update related attributes to match the dimension's view status
            cursor.execute("""
                UPDATE acquisitionattributes 
                SET hawkeyeview = %s
                WHERE dimension_id = %s
            """, (view_status, dim_id))
        
        conn.commit()
        
        return jsonify({"success": True, "message": f"Updated {len(dimension_ids)} dimensions to '{view_status}'"})
        
    except Exception as e:
        current_app.logger.error(f"Error updating dimensions: {str(e)}")
        return jsonify({"error": f"Error updating dimensions: {str(e)}"}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
            
            
@dashboard_package.route('/test', methods=['GET'])
def test_dashboard_api():
    """
    A simple test endpoint to verify the dashboard API is working.
    """
    return jsonify({
        "status": "success",
        "message": "Way to go Claude and Robert! The dashboard API is set up and active.",
        "timestamp": datetime.datetime.now().isoformat()
    }), 200  



@dashboard_package.route('/api/msa/dashboard-data', methods=['GET'])
def get_service_orders():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Query service orders with joined data
        cursor.execute("""
            SELECT so.id, so.order_id, so.display_name as order_name,
                   p.display_name as provider_name, ma.msa_id,
                   so.status, so.start_date, so.end_date, so.value as order_value,
                   so.availability, so.hawkeyeview
            FROM service_orders so
            JOIN master_agreements ma ON so.msa_id = ma.id
            JOIN providers p ON ma.provider_id = p.id
            JOIN customers c ON ma.customer_id = c.id
            WHERE so.availability = 'All'
            ORDER BY so.created_at DESC
        """)
        
        orders = cursor.fetchall()
        
        # For each order, get the resources
        for order in orders:
            cursor.execute("""
                SELECT r.role, r.quantity, r.rate, r.utilization 
                FROM service_order_resources r
                WHERE r.order_id = %s
            """, (order['id'],))
            order['resources'] = cursor.fetchall()
            
            # Calculate fiscal year based on start and end dates
            start_date = datetime.strptime(order['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(order['end_date'], '%Y-%m-%d')
            
            # Set fiscal year based on your fiscal calendar (Jun-May)
            fy_start = start_date.year if start_date.month >= 6 else start_date.year - 1
            fy_end = end_date.year if end_date.month >= 6 else end_date.year - 1
            
            if fy_start == fy_end:
                order['fiscal_year'] = f"FY{str(fy_start)[-2:]}"
            else:
                order['fiscal_year'] = f"FY{str(fy_start)[-2:]}-{str(fy_end)[-2:]}"
            
        cursor.close()
        conn.close()
        
        return jsonify(orders)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
