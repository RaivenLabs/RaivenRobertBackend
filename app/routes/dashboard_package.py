# app/routes/dashboard_package.py
from flask import Blueprint, request, jsonify, current_app
import json
import os
import psycopg2
import datetime
from datetime import datetime, timedelta
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

# Create a Blueprint for dashboard routes
dashboard_package = Blueprint('dashboard', __name__, url_prefix='/api')

# =========================================
# Dashboard Data Routes
# =========================================

@dashboard_package.route('/msa/dashboard-data', methods=['GET'])
def get_dashboard_data():
    """
    Get comprehensive dashboard data including service orders, providers, and summary metrics.
    Optional query param: msa_id to filter for a specific master service agreement
    """
    try:
        # Get the MSA ID from query parameters if provided
        msa_id = request.args.get('msa_id')
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            # Use sample data if database connection fails
            current_app.logger.warning("Database connection failed, using sample data")
            return jsonify({
                "lastUpdated": datetime.now().isoformat(),
                "serviceOrders": get_sample_service_orders(),
                "providers": get_sample_providers(),
                "summary": get_sample_summary()
            })
            
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Query service orders with joined data
        # Use LEFT JOINs to include all service orders even if related records are missing
        query = """
            SELECT 
                so.id, 
                so.order_id, 
                so.display_name as order_name,
                COALESCE(p.display_name, 'Unknown Provider') as provider_name, 
                COALESCE(ma.msa_id, 'Unknown MSA') as master,
                so.model_type, 
                so.start_date, 
                so.end_date, 
                so.value as order_value,
                so.currency,
                so.status
            FROM service_orders so
            LEFT JOIN master_agreements ma ON so.msa_id = ma.id
            LEFT JOIN providers p ON ma.provider_id = p.id
        """
        
        # Add WHERE clause if msa_id is provided
        params = []
        if msa_id:
            query += " WHERE so.msa_id = %s"
            params.append(msa_id)
        
        query += " ORDER BY so.created_at DESC"
        
        cursor.execute(query, params)
        db_orders = cursor.fetchall()
        service_orders = []
        
        # For each order, get the resources and format the data
        for order in db_orders:
            # Check if service_order_resources table exists and query resources
            try:
                cursor.execute("""
                    SELECT r.role, r.quantity, r.rate, r.utilization 
                    FROM service_order_resources r
                    WHERE r.order_id = %s
                """, (order['id'],))
                resources = cursor.fetchall()
                
                # Format resources
                formatted_resources = []
                for resource in resources:
                    formatted_resources.append({
                        "role": resource['role'],
                        "quantity": resource['quantity'],
                        "rate": f"${resource['rate']}/hr" if resource['rate'] else "$0/hr",
                        "utilization": f"{resource['utilization']}%" if resource['utilization'] else "0%"
                    })
            except Exception as e:
                current_app.logger.warning(f"Error fetching resources for order {order['id']}: {e}")
                formatted_resources = []  # Use empty list if table doesn't exist yet
            
            # Format dates
            start_date = order['start_date']
            end_date = order['end_date'] if order['end_date'] else None
            
            formatted_start = start_date.strftime('%b %d, %Y') if start_date else 'N/A'
            formatted_end = end_date.strftime('%b %d, %Y') if end_date else 'TBD'
            
            # Set fiscal year based on fiscal calendar (Jun-May)
            if start_date:
                fy_start = start_date.year if start_date.month >= 6 else start_date.year - 1
                
                if end_date:
                    fy_end = end_date.year if end_date.month >= 6 else end_date.year - 1
                    fiscal_year = f"FY{str(fy_start)[-2:]}"
                    if fy_start != fy_end:
                        fiscal_year = f"FY{str(fy_start)[-2:]}-{str(fy_end)[-2:]}"
                else:
                    fiscal_year = f"FY{str(fy_start)[-2:]}"
            else:
                fiscal_year = "N/A"
            
            # Calculate completion score based on status and dates
            current_date = datetime.now().date()
            
            # For completed orders
            if order['status'] == 'Completed':
                completion_score = 100
            # For orders with end dates in the past
            elif end_date and current_date > end_date:
                completion_score = 85  # Assume nearly complete but not marked as completed
            # For active orders with start and end dates
            elif start_date and end_date and current_date >= start_date:
                days_elapsed = max(0, (current_date - start_date).days)
                total_days = max(1, (end_date - start_date).days)  # Prevent division by zero
                completion_score = min(100, max(0, int((days_elapsed / total_days) * 100)))
            # For future orders (start date in the future)
            elif start_date and current_date < start_date:
                completion_score = 0
            # Default for orders without proper dates
            else:
                completion_score = 0
            
            # Create formatted order object
            formatted_order = {
                "id": order['order_id'],
                "orderName": order['order_name'],
                "provider": order['provider_name'],
                "master": order['master'],
                "status": order['status'],
                "startDate": formatted_start,
                "endDate": formatted_end,
                "orderValue": f"${int(order['order_value']):,}" if order['order_value'] else "$0",
                "fiscalYear": fiscal_year,
                "completionScore": f"{completion_score}%",
                "resources": formatted_resources,
                "dimensions": get_sample_dimensions(completion_score)  # Generate dimensions based on completion
            }
            
            service_orders.append(formatted_order)
        
        # Calculate summary metrics
        total_active = sum(1 for order in service_orders if order['status'] == 'Active')
        total_value = sum(int(order['orderValue'].replace('$', '').replace(',', '')) for order in service_orders)
        
        # Group by provider and fiscal year
        spend_by_provider = {}
        spend_by_fy = {}
        
        for order in service_orders:
            provider = order['provider']
            value = int(order['orderValue'].replace('$', '').replace(',', ''))
            fy = order['fiscalYear']
            
            spend_by_provider[provider] = spend_by_provider.get(provider, 0) + value
            spend_by_fy[fy] = spend_by_fy.get(fy, 0) + value
        
        # Format providers data
        providers = [{"name": k, "spend": v} for k, v in spend_by_provider.items()]
        
        # Return dashboard data
        response_data = {
            "lastUpdated": datetime.now().isoformat(),
            "serviceOrders": service_orders,
            "providers": providers,
            "summary": {
                "totalOrders": len(service_orders),
                "activeOrders": total_active,
                "totalValue": total_value,
                "spendByFY": spend_by_fy
            }
        }
        
        cursor.close()
        conn.close()
        
        return jsonify(response_data)
        
    except Exception as e:
        current_app.logger.error(f"Error fetching dashboard data: {str(e)}")
        
        # Return sample data on error
        return jsonify({
            "lastUpdated": datetime.now().isoformat(),
            "serviceOrders": get_sample_service_orders(),
            "providers": get_sample_providers(),
            "summary": get_sample_summary()
        })

# =========================================
# Service Order Routes
# =========================================

@dashboard_package.route('/service-orders', methods=['GET'])
def get_service_orders():
    """
    Get all service orders, optionally filtered by MSA ID.
    """
    try:
        # Get the MSA ID from query parameters if provided
        msa_id = request.args.get('msa_id')
        
        conn = get_db_connection()
        if not conn:
            return jsonify(get_sample_service_orders())
            
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Basic query
        query = """
            SELECT * FROM service_orders
        """
        
        # Add WHERE clause if msa_id is provided
        params = []
        if msa_id:
            query += " WHERE msa_id = %s"
            params.append(msa_id)
            
        query += " ORDER BY created_at DESC"
        
        cursor.execute(query, params)
        orders = cursor.fetchall()
        
        # Format for response
        response = []
        for order in orders:
            response.append({
                "id": order['order_id'],
                "name": order['display_name'],
                "msa_id": order['msa_id'],
                "status": order['status'],
                "value": order['value'],
                "startDate": order['start_date'].strftime('%Y-%m-%d'),
                "endDate": order['end_date'].strftime('%Y-%m-%d') if order['end_date'] else None
            })
            
        cursor.close()
        conn.close()
        
        return jsonify(response)
        
    except Exception as e:
        current_app.logger.error(f"Error fetching service orders: {str(e)}")
        return jsonify(get_sample_service_orders())

@dashboard_package.route('/service-orders/<order_id>', methods=['PUT'])
def update_service_order(order_id):
    """
    Update a service order.
    """
    try:
        data = request.json
        
        conn = get_db_connection()
        if not conn:
            return jsonify({
                "success": False,
                "message": "Database connection failed"
            }), 500
            
        cursor = conn.cursor()
        
        # Build dynamic update query based on provided fields
        update_fields = []
        params = []
        
        if 'status' in data:
            update_fields.append("status = %s")
            params.append(data['status'])
            
        if 'display_name' in data:
            update_fields.append("display_name = %s")
            params.append(data['display_name'])
            
        if 'start_date' in data:
            update_fields.append("start_date = %s")
            params.append(data['start_date'])
            
        if 'end_date' in data:
            update_fields.append("end_date = %s")
            params.append(data['end_date'])
            
        if 'value' in data:
            update_fields.append("value = %s")
            params.append(data['value'])
            
        if not update_fields:
            return jsonify({
                "success": False,
                "message": "No fields to update"
            }), 400
            
        # Complete the query
        query = f"UPDATE service_orders SET {', '.join(update_fields)} WHERE order_id = %s"
        params.append(order_id)
        
        cursor.execute(query, params)
        rows_affected = cursor.rowcount
        conn.commit()
        
        cursor.close()
        conn.close()
        
        if rows_affected == 0:
            return jsonify({
                "success": False,
                "message": f"Order {order_id} not found"
            }), 404
            
        return jsonify({
            "success": True,
            "message": f"Order {order_id} updated successfully"
        })
        
    except Exception as e:
        current_app.logger.error(f"Error updating service order: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@dashboard_package.route('/service-orders/<order_id>', methods=['DELETE'])
def delete_service_order(order_id):
    """
    Delete a service order.
    """
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({
                "success": False,
                "message": "Database connection failed"
            }), 500
            
        cursor = conn.cursor()
        
        # First, delete related resources if they exist
        try:
            cursor.execute("""
                DELETE FROM service_order_resources
                WHERE order_id IN (
                    SELECT id FROM service_orders WHERE order_id = %s
                )
            """, (order_id,))
        except Exception as e:
            # Ignore errors if the resources table doesn't exist yet
            current_app.logger.warning(f"Could not delete resources for {order_id}: {e}")
        
        # Then delete the order
        cursor.execute("DELETE FROM service_orders WHERE order_id = %s", (order_id,))
        rows_affected = cursor.rowcount
        conn.commit()
        
        cursor.close()
        conn.close()
        
        if rows_affected == 0:
            return jsonify({
                "success": False,
                "message": f"Order {order_id} not found"
            }), 404
            
        return jsonify({
            "success": True,
            "message": f"Order {order_id} deleted successfully"
        })
        
    except Exception as e:
        current_app.logger.error(f"Error deleting service order: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@dashboard_package.route('/service-orders', methods=['POST'])
def create_service_order():
    """
    Create a new service order.
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['msa_id', 'display_name', 'model_type', 'status']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "message": f"Missing required field: {field}"
                }), 400
        
        # Generate a new order ID
        prefix = "SO"
        year = datetime.now().year
        random_suffix = str(datetime.now().microsecond)[:3]
        
        # Get customer and provider prefixes if possible
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=DictCursor)
            
            # Try to get customer and provider info for the order ID
            cursor.execute("""
                SELECT c.name_short, p.name_short
                FROM master_agreements ma
                JOIN customers c ON ma.customer_id = c.id
                JOIN providers p ON ma.provider_id = p.id
                WHERE ma.id = %s
            """, (data['msa_id'],))
            
            result = cursor.fetchone()
            if result:
                customer_short = result['name_short']
                provider_short = result['name_short']
                new_order_id = f"{prefix}-{customer_short}-{provider_short}-{random_suffix}"
            else:
                new_order_id = f"{prefix}-{year}-{random_suffix}"
                
        except Exception as e:
            # If anything fails, use the basic format
            current_app.logger.warning(f"Could not generate custom order ID: {e}")
            new_order_id = f"{prefix}-{year}-{random_suffix}"
        
        # Set defaults for optional fields
        start_date = data.get('start_date', datetime.now().date())
        end_date = data.get('end_date', datetime.now().date() + timedelta(days=90))
        value = data.get('value', 0)
        currency = data.get('currency', 'USD')
        
        # Insert the new order
        conn = get_db_connection()
        if not conn:
            return jsonify({
                "success": False,
                "message": "Database connection failed"
            }), 500
            
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO service_orders 
            (order_id, msa_id, display_name, model_type, start_date, end_date, 
            value, currency, status, availability) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            new_order_id,
            data['msa_id'],
            data['display_name'],
            data['model_type'],
            start_date,
            end_date,
            value,
            currency,
            data['status'],
            'All'  # Default availability
        ))
        
        new_id = cursor.fetchone()[0]
        conn.commit()
        
        # If resources are provided, add them
        if 'resources' in data and isinstance(data['resources'], list):
            for resource in data['resources']:
                try:
                    cursor.execute("""
                        INSERT INTO service_order_resources
                        (order_id, role, quantity, rate, utilization)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        new_id,
                        resource.get('role', 'Unspecified'),
                        resource.get('quantity', 1),
                        resource.get('rate', 0),
                        resource.get('utilization', 100)
                    ))
                except Exception as e:
                    current_app.logger.warning(f"Could not add resource to order: {e}")
                    
            conn.commit()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            "success": True,
            "message": "Service order created successfully",
            "id": new_order_id,
            "databaseId": new_id
        })
        
    except Exception as e:
        current_app.logger.error(f"Error creating service order: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

# =========================================
# Helper Functions for Sample Data
# =========================================

def get_sample_service_orders():
    """
    Return sample service order data for development and testing.
    """
    return [
        {
            "id": "SO-2025-42",
            "orderName": "Supply Chain Analytics Platform",
            "provider": "Midway Consulting",
            "master": "MSA-NIMITZ-MIDWAY-2023",
            "status": "Active",
            "startDate": "Jan 15, 2025",
            "endDate": "Jul 15, 2025",
            "orderValue": "$245,000",
            "fiscalYear": "FY25",
            "completionScore": "25%",
            "resources": [
                { "role": "Technical Lead", "quantity": 1, "rate": "$175/hr", "utilization": "100%" },
                { "role": "Senior Developer", "quantity": 2, "rate": "$150/hr", "utilization": "100%" },
                { "role": "Data Scientist", "quantity": 1, "rate": "$165/hr", "utilization": "75%" },
                { "role": "Project Manager", "quantity": 1, "rate": "$160/hr", "utilization": "50%" }
            ],
            "dimensions": [
                { "name": "Resource Allocation", "progress": 90, "color": "#48bb78" },
                { "name": "Budget Utilization", "progress": 30, "color": "#4299e1" },
                { "name": "Timeline Progress", "progress": 22, "color": "#f56565" },
                { "name": "Deliverables", "progress": 20, "color": "#f56565" }
            ]
        },
        {
            "id": "SO-2025-41",
            "orderName": "E-commerce Platform Optimization",
            "provider": "Midway Consulting",
            "master": "MSA-NIMITZ-MIDWAY-2023",
            "status": "Active",
            "startDate": "Feb 1, 2025",
            "endDate": "May 31, 2025",
            "orderValue": "$178,500",
            "fiscalYear": "FY25",
            "completionScore": "40%",
            "resources": [
                { "role": "UX Designer", "quantity": 1, "rate": "$155/hr", "utilization": "100%" },
                { "role": "Senior Developer", "quantity": 1, "rate": "$150/hr", "utilization": "100%" },
                { "role": "Frontend Developer", "quantity": 1, "rate": "$135/hr", "utilization": "100%" }
            ],
            "dimensions": [
                { "name": "Resource Allocation", "progress": 95, "color": "#48bb78" },
                { "name": "Budget Utilization", "progress": 42, "color": "#4299e1" },
                { "name": "Timeline Progress", "progress": 35, "color": "#f56565" },
                { "name": "Deliverables", "progress": 40, "color": "#f56565" }
            ]
        },
        {
            "id": "SO-2025-38",
            "orderName": "Data Migration & Integration",
            "provider": "Apex Systems",
            "master": "MSA-NIMITZ-APEX-2023",
            "status": "Pending Approval",
            "startDate": "Mar 15, 2025",
            "endDate": "Jun 30, 2025",
            "orderValue": "$215,000",
            "fiscalYear": "FY25",
            "completionScore": "0%",
            "resources": [
                { "role": "Solution Architect", "quantity": 1, "rate": "$185/hr", "utilization": "75%" },
                { "role": "Data Engineer", "quantity": 2, "rate": "$160/hr", "utilization": "100%" },
                { "role": "QA Specialist", "quantity": 1, "rate": "$125/hr", "utilization": "50%" }
            ],
            "dimensions": [
                { "name": "Resource Allocation", "progress": 0, "color": "#48bb78" },
                { "name": "Budget Utilization", "progress": 0, "color": "#4299e1" },
                { "name": "Timeline Progress", "progress": 0, "color": "#f56565" },
                { "name": "Deliverables", "progress": 0, "color": "#f56565" }
            ]
        },
        {
            "id": "SO-2025-35",
            "orderName": "Digital Marketing Automation",
            "provider": "Technica Solutions",
            "master": "MSA-NIMITZ-TECHNICA-2023",
            "status": "Planning",
            "startDate": "Apr 1, 2025",
            "endDate": "Sep 30, 2025",
            "orderValue": "$320,000",
            "fiscalYear": "FY25-26",
            "completionScore": "10%",
            "resources": [
                { "role": "Marketing Tech Lead", "quantity": 1, "rate": "$170/hr", "utilization": "100%" },
                { "role": "Developer", "quantity": 2, "rate": "$140/hr", "utilization": "100%" },
                { "role": "Content Specialist", "quantity": 1, "rate": "$130/hr", "utilization": "75%" },
                { "role": "Project Manager", "quantity": 1, "rate": "$160/hr", "utilization": "50%" }
            ],
            "dimensions": [
                { "name": "Resource Allocation", "progress": 80, "color": "#48bb78" },
                { "name": "Budget Utilization", "progress": 8, "color": "#4299e1" },
                { "name": "Timeline Progress", "progress": 5, "color": "#f56565" },
                { "name": "Deliverables", "progress": 10, "color": "#f56565" }
            ]
        },
        {
            "id": "SO-2025-32",
            "orderName": "Business Intelligence Dashboard",
            "provider": "Midway Consulting",
            "master": "MSA-NIMITZ-MIDWAY-2023",
            "status": "Active",
            "startDate": "Jan 10, 2025",
            "endDate": "Apr 30, 2025",
            "orderValue": "$145,000",
            "fiscalYear": "FY25",
            "completionScore": "65%",
            "resources": [
                { "role": "BI Specialist", "quantity": 1, "rate": "$165/hr", "utilization": "100%" },
                { "role": "Data Analyst", "quantity": 1, "rate": "$145/hr", "utilization": "100%" }
            ],
            "dimensions": [
                { "name": "Resource Allocation", "progress": 100, "color": "#48bb78" },
                { "name": "Budget Utilization", "progress": 60, "color": "#4299e1" },
                { "name": "Timeline Progress", "progress": 65, "color": "#48bb78" },
                { "name": "Deliverables", "progress": 70, "color": "#48bb78" }
            ]
        },
        {
            "id": "SO-2025-28",
            "orderName": "Mobile App Development",
            "provider": "Apex Systems",
            "master": "MSA-NIMITZ-APEX-2023",
            "status": "On Hold",
            "startDate": "Mar 1, 2025",
            "endDate": "Aug 31, 2025",
            "orderValue": "$275,000",
            "fiscalYear": "FY25-26",
            "completionScore": "15%",
            "resources": [
                { "role": "Mobile Lead Developer", "quantity": 1, "rate": "$175/hr", "utilization": "100%" },
                { "role": "iOS Developer", "quantity": 1, "rate": "$160/hr", "utilization": "100%" },
                { "role": "Android Developer", "quantity": 1, "rate": "$160/hr", "utilization": "100%" },
                { "role": "QA Specialist", "quantity": 1, "rate": "$125/hr", "utilization": "75%" }
            ],
            "dimensions": [
                { "name": "Resource Allocation", "progress": 30, "color": "#f56565" },
                { "name": "Budget Utilization", "progress": 12, "color": "#4299e1" },
                { "name": "Timeline Progress", "progress": 15, "color": "#f56565" },
                { "name": "Deliverables", "progress": 15, "color": "#f56565" }
            ]
        }
    ]

def get_sample_providers():
    """
    Return sample provider data.
    """
    return [
        { "name": "Midway Consulting", "spend": 568500 },
        { "name": "Apex Systems", "spend": 490000 },
        { "name": "Technica Solutions", "spend": 320000 }
    ]

def get_sample_summary():
    """
    Return sample summary metrics.
    """
    return {
        "totalOrders": 6,
        "activeOrders": 3,
        "totalValue": 1378500,
        "spendByFY": {
            "FY25": 783500,
            "FY25-26": 595000
        }
    }

def get_sample_dimensions(completion_score=0):
    """
    Return sample dimensions for a service order based on completion score.
    """
    # Convert completion score string to int if needed
    if isinstance(completion_score, str):
        completion_score = int(completion_score.rstrip('%'))
    
    # Generate realistic dimensions based on completion_score
    resource_allocation = min(100, completion_score + 20)
    budget_utilization = int(completion_score * 1.2) if completion_score < 50 else int(completion_score * 0.9)
    timeline_progress = completion_score
    deliverables = int(completion_score * 0.9)
    
    # Set colors based on values
    def get_color(value):
        if value >= 80:
            return "#48bb78"  # green
        if value >= 60:
            return "#68d391"  # light green
        if value >= 40:
            return "#4299e1"  # blue
        if value >= 20:
            return "#f6ad55"  # orange
        return "#f56565"  # red
    
    return [
        { "name": "Resource Allocation", "progress": resource_allocation, "color": get_color(resource_allocation) },
        { "name": "Budget Utilization", "progress": budget_utilization, "color": get_color(budget_utilization) },
        { "name": "Timeline Progress", "progress": timeline_progress, "color": get_color(timeline_progress) },
        { "name": "Deliverables", "progress": deliverables, "color": get_color(deliverables) }
    ]

# =========================================
# Test Route
# =========================================

@dashboard_package.route('/test', methods=['GET'])
def test_dashboard_api():
    """
    A simple test endpoint to verify the dashboard API is working.
    """
    # Get database connection status
    db_status = "connected"
    try:
        conn = get_db_connection()
        if not conn:
            db_status = "connection failed"
        else:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
    except Exception as e:
        db_status = f"error: {str(e)}"

    return jsonify({
        "status": "success",
        "message": "The dashboard API is set up and active.",
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    }), 200
