# app/routes/service_orders_blueprint.py
from flask import Blueprint, request, jsonify, current_app
import json
import os
import psycopg2
import datetime
from datetime import datetime, timedelta
from psycopg2.extras import RealDictCursor, DictCursor
from dotenv import load_dotenv

# PART 1: SETUP & CONFIGURATION
# ===================================================================

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
print("==== Service Orders Blueprint PostgreSQL Connection Parameters ====")
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

# Create a Blueprint for service order routes
dashboard_package = Blueprint('service_orders', __name__, url_prefix='/api')

# PART 2: SERVICE ORDER ROUTES
# ===================================================================

@dashboard_package.route('/service-orders', methods=['GET'])
def get_service_orders():
    """
    Get all service orders with optional filtering.
    
    Query Parameters:
    - provider_id: Filter by provider ID
    - customer_id: Filter by customer ID
    - status: Filter by order status
    - msa_id: Filter by master agreement ID
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "error": "Database connection failed"}), 500
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get optional filter parameters
        provider_id = request.args.get('provider_id')
        customer_id = request.args.get('customer_id')
        status = request.args.get('status')
        msa_id = request.args.get('msa_id')
        
        query = """
            SELECT so.*, 
                  ma.display_name as msa_name, 
                  p.display_name as provider_name, 
                  c.display_name as customer_name
            FROM public.service_orders so
            JOIN public.master_agreements ma ON so.msa_id = ma.id
            JOIN public.providers p ON ma.provider_id = p.id
            JOIN public.customers c ON ma.customer_id = c.id
            WHERE 1=1
        """
        params = []
        
        if provider_id:
            query += " AND ma.provider_id = %s"
            params.append(provider_id)
        
        if customer_id:
            query += " AND ma.customer_id = %s"
            params.append(customer_id)
            
        if status:
            query += " AND so.status = %s"
            params.append(status)
            
        if msa_id:
            query += " AND so.msa_id = %s"
            params.append(msa_id)
            
        query += " ORDER BY so.created_at DESC"
        
        cursor.execute(query, params)
        service_orders = cursor.fetchall()
        
        # Format dates for JSON serialization
        for order in service_orders:
            if 'start_date' in order and order['start_date']:
                order['start_date'] = order['start_date'].isoformat()
            if 'end_date' in order and order['end_date']:
                order['end_date'] = order['end_date'].isoformat()
            if 'created_at' in order and order['created_at']:
                order['created_at'] = order['created_at'].isoformat()
        
        return jsonify({"success": True, "data": service_orders}), 200
        
    except Exception as e:
        current_app.logger.error(f"Error fetching service orders: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@dashboard_package.route('/service-orders/<order_id>', methods=['GET'])
def get_service_order_details(order_id):
    """
    Get detailed information about a specific service order.
    
    Returns:
    - Basic order information
    - Deliverables associated with the order
    - Resources assigned to the order
    - Billing information for the order
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "error": "Database connection failed"}), 500
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get basic order information
        cursor.execute("""
            SELECT so.*, 
                  ma.display_name as msa_name, 
                  p.display_name as provider_name, 
                  c.display_name as customer_name
            FROM public.service_orders so
            JOIN public.master_agreements ma ON so.msa_id = ma.id
            JOIN public.providers p ON ma.provider_id = p.id
            JOIN public.customers c ON ma.customer_id = c.id
            WHERE so.order_id = %s
        """, (order_id,))
        
        order = cursor.fetchone()
        if not order:
            return jsonify({"success": False, "error": "Service order not found"}), 404
        
        # Format dates for JSON serialization
        if 'start_date' in order and order['start_date']:
            order['start_date'] = order['start_date'].isoformat()
        if 'end_date' in order and order['end_date']:
            order['end_date'] = order['end_date'].isoformat()
        if 'created_at' in order and order['created_at']:
            order['created_at'] = order['created_at'].isoformat()
        
        # Get deliverables
        try:
            cursor.execute("""
                SELECT * FROM public.service_order_deliverables
                WHERE service_order_id = %s
                ORDER BY deliverable_number
            """, (order['id'],))
            deliverables = cursor.fetchall()
            
            # Format dates for deliverables
            for deliverable in deliverables:
                if 'due_date' in deliverable and deliverable['due_date']:
                    deliverable['due_date'] = deliverable['due_date'].isoformat()
                if 'created_at' in deliverable and deliverable['created_at']:
                    deliverable['created_at'] = deliverable['created_at'].isoformat()
                if 'updated_at' in deliverable and deliverable['updated_at']:
                    deliverable['updated_at'] = deliverable['updated_at'].isoformat()
        except Exception as e:
            current_app.logger.warning(f"Error fetching deliverables: {str(e)}")
            deliverables = []
        
        # Get resources
        try:
            cursor.execute("""
                SELECT * FROM public.service_order_resources
                WHERE service_order_id = %s
            """, (order['id'],))
            resources = cursor.fetchall()
            
            # Format dates for resources
            for resource in resources:
                if 'start_date' in resource and resource['start_date']:
                    resource['start_date'] = resource['start_date'].isoformat()
                if 'end_date' in resource and resource['end_date']:
                    resource['end_date'] = resource['end_date'].isoformat()
                if 'created_at' in resource and resource['created_at']:
                    resource['created_at'] = resource['created_at'].isoformat()
                if 'updated_at' in resource and resource['updated_at']:
                    resource['updated_at'] = resource['updated_at'].isoformat()
        except Exception as e:
            current_app.logger.warning(f"Error fetching resources: {str(e)}")
            resources = []
        
        # Get billing information
        try:
            cursor.execute("""
                SELECT * FROM public.service_order_billing
                WHERE service_order_id = %s
                ORDER BY billing_period_start DESC
            """, (order['id'],))
            billing = cursor.fetchall()
            
            # Format dates for billing records
            for bill in billing:
                if 'invoice_date' in bill and bill['invoice_date']:
                    bill['invoice_date'] = bill['invoice_date'].isoformat()
                if 'billing_period_start' in bill and bill['billing_period_start']:
                    bill['billing_period_start'] = bill['billing_period_start'].isoformat()
                if 'billing_period_end' in bill and bill['billing_period_end']:
                    bill['billing_period_end'] = bill['billing_period_end'].isoformat()
                if 'payment_date' in bill and bill['payment_date']:
                    bill['payment_date'] = bill['payment_date'].isoformat()
                if 'created_at' in bill and bill['created_at']:
                    bill['created_at'] = bill['created_at'].isoformat()
                if 'updated_at' in bill and bill['updated_at']:
                    bill['updated_at'] = bill['updated_at'].isoformat()
        except Exception as e:
            current_app.logger.warning(f"Error fetching billing info: {str(e)}")
            billing = []
        
        # Combine all data
        result = {
            "order": order,
            "deliverables": deliverables,
            "resources": resources,
            "billing": billing
        }
        
        return jsonify({"success": True, "data": result}), 200
        
    except Exception as e:
        current_app.logger.error(f"Error fetching service order details: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@dashboard_package.route('/service-orders', methods=['POST'])
def create_service_order():
    """
    Create a new service order with optional deliverables and resources.
    
    Required fields:
    - msa_id: ID of the master agreement
    - display_name: Name of the service order
    - model_type: Type of the service order (e.g., 'SOW', 'PO')
    - start_date: Start date of the service order
    
    Optional fields:
    - order_id: Custom ID for the order (generated if not provided)
    - end_date: End date of the service order
    - value: Contract value
    - currency: Currency code (default: 'USD')
    - description: Detailed description
    - status: Status of the order (default: 'Active')
    - clm_number: Contract Lifecycle Management number
    - deliverables: List of deliverable objects
    - resources: List of resource objects
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "error": "Database connection failed"}), 500
    
    try:
        cursor = conn.cursor()
        data = request.json
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ['msa_id', 'display_name', 'model_type', 'start_date']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "success": False, 
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Generate order_id if not provided
        if 'order_id' not in data or not data['order_id']:
            # Get customer and provider prefixes if possible
            try:
                cursor.execute("""
                    SELECT c.short_name as customer_short, p.short_name as provider_short
                    FROM public.master_agreements ma
                    JOIN public.customers c ON ma.customer_id = c.id
                    JOIN public.providers p ON ma.provider_id = p.id
                    WHERE ma.id = %s
                """, (data['msa_id'],))
                
                result = cursor.fetchone()
                prefix = "SO"
                year = datetime.now().year
                random_suffix = str(datetime.now().microsecond)[:4]
                
                if result and result[0] and result[1]:
                    order_id = f"{prefix}-{result[0]}-{result[1]}-{random_suffix}"
                else:
                    order_id = f"{prefix}-{year}-{random_suffix}"
            except Exception as e:
                current_app.logger.warning(f"Could not generate detailed order ID: {str(e)}")
                order_id = f"SO-{year}-{random_suffix}"
        else:
            order_id = data['order_id']
        
        # Start a transaction
        conn.autocommit = False
        
        # Parse dates
        start_date = datetime.fromisoformat(data['start_date'].replace('Z', '+00:00')) if isinstance(data['start_date'], str) else data['start_date']
        end_date = datetime.fromisoformat(data['end_date'].replace('Z', '+00:00')) if 'end_date' in data and data['end_date'] and isinstance(data['end_date'], str) else data.get('end_date')
        
        # 1. Insert the service order
        cursor.execute("""
            INSERT INTO public.service_orders (
                order_id, msa_id, display_name, model_type, 
                start_date, end_date, value, currency, 
                description, status, clm_number
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            order_id,
            data['msa_id'],
            data['display_name'],
            data['model_type'],
            start_date,
            end_date,
            data.get('value'),
            data.get('currency', 'USD'),
            data.get('description'),
            data.get('status', 'Active'),
            data.get('clm_number')
        ))
        
        service_order_id = cursor.fetchone()[0]
        
        # 2. Insert deliverables if provided
        deliverables = data.get('deliverables', [])
        for deliverable in deliverables:
            due_date = datetime.fromisoformat(deliverable['due_date'].replace('Z', '+00:00')) if 'due_date' in deliverable and deliverable['due_date'] and isinstance(deliverable['due_date'], str) else deliverable.get('due_date')
            
            cursor.execute("""
                INSERT INTO public.service_order_deliverables (
                    service_order_id, deliverable_number, title,
                    description, due_date, status
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                service_order_id,
                deliverable.get('deliverable_number'),
                deliverable.get('title'),
                deliverable.get('description'),
                due_date,
                deliverable.get('status', 'Pending')
            ))
        
        # 3. Insert resources if provided
        resources = data.get('resources', [])
        for resource in resources:
            resource_start_date = datetime.fromisoformat(resource['start_date'].replace('Z', '+00:00')) if 'start_date' in resource and resource['start_date'] and isinstance(resource['start_date'], str) else resource.get('start_date')
            resource_end_date = datetime.fromisoformat(resource['end_date'].replace('Z', '+00:00')) if 'end_date' in resource and resource['end_date'] and isinstance(resource['end_date'], str) else resource.get('end_date')
            
            cursor.execute("""
                INSERT INTO public.service_order_resources (
                    service_order_id, resource_name, role_title,
                    role_level, location, hourly_rate, allocated_hours,
                    start_date, end_date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                service_order_id,
                resource.get('resource_name'),
                resource.get('role_title'),
                resource.get('role_level'),
                resource.get('location'),
                resource.get('hourly_rate'),
                resource.get('allocated_hours'),
                resource_start_date,
                resource_end_date
            ))
        
        # Commit the transaction
        conn.commit()
        
        return jsonify({
            "success": True, 
            "message": "Service order created successfully",
            "order_id": order_id,
            "id": service_order_id
        }), 201
        
    except Exception as e:
        conn.rollback()
        current_app.logger.error(f"Error creating service order: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@dashboard_package.route('/service-orders/<order_id>', methods=['PUT'])
def update_service_order(order_id):
    """
    Update an existing service order.
    
    Can update basic order information, but not deliverables or resources
    (use separate endpoints for those).
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "error": "Database connection failed"}), 500
    
    try:
        cursor = conn.cursor()
        data = request.json
        
        if not data:
            return jsonify({"success": False, "error": "No update data provided"}), 400
        
        # Build dynamic SQL update statement
        update_fields = []
        params = []
        
        # Map fields to database columns
        field_mapping = {
            'display_name': 'display_name',
            'status': 'status',
            'start_date': 'start_date',
            'end_date': 'end_date',
            'value': 'value',
            'currency': 'currency',
            'description': 'description',
            'clm_number': 'clm_number',
            'model_type': 'model_type'
        }
        
        # Process each field if present in the request
        for field, column in field_mapping.items():
            if field in data:
                # Special handling for date fields
                if field in ['start_date', 'end_date'] and data[field]:
                    try:
                        if isinstance(data[field], str):
                            # Parse ISO date string
                            data[field] = datetime.fromisoformat(data[field].replace('Z', '+00:00')).date()
                    except ValueError:
                        return jsonify({
                            "success": False, 
                            "error": f"Invalid date format for {field}"
                        }), 400
                
                update_fields.append(f"{column} = %s")
                params.append(data[field])
        
        if not update_fields:
            return jsonify({"success": False, "error": "No valid fields to update"}), 400
        
        # Build and execute the update query
        query = f"UPDATE public.service_orders SET {', '.join(update_fields)} WHERE order_id = %s RETURNING id"
        params.append(order_id)
        
        cursor.execute(query, params)
        result = cursor.fetchone()
        
        if not result:
            return jsonify({"success": False, "error": f"Service order with ID {order_id} not found"}), 404
        
        conn.commit()
        
        return jsonify({
            "success": True,
            "message": f"Service order {order_id} updated successfully",
            "id": result[0]
        }), 200
        
    except Exception as e:
        conn.rollback()
        current_app.logger.error(f"Error updating service order: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@dashboard_package.route('/service-orders/<order_id>', methods=['DELETE'])
def delete_service_order(order_id):
    """
    Delete a service order and its associated data (deliverables, resources, billing).
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "error": "Database connection failed"}), 500
    
    try:
        # Start a transaction
        conn.autocommit = False
        cursor = conn.cursor()
        
        # Get the service order's internal ID
        cursor.execute("SELECT id FROM public.service_orders WHERE order_id = %s", (order_id,))
        result = cursor.fetchone()
        
        if not result:
            return jsonify({"success": False, "error": f"Service order {order_id} not found"}), 404
            
        service_order_id = result[0]
        
        # Delete related deliverables (if the table exists)
        try:
            cursor.execute("DELETE FROM public.service_order_deliverables WHERE service_order_id = %s", (service_order_id,))
            deliverables_deleted = cursor.rowcount
        except Exception as e:
            current_app.logger.warning(f"Could not delete deliverables: {str(e)}")
            deliverables_deleted = 0
        
        # Delete related resources (if the table exists)
        try:
            cursor.execute("DELETE FROM public.service_order_resources WHERE service_order_id = %s", (service_order_id,))
            resources_deleted = cursor.rowcount
        except Exception as e:
            current_app.logger.warning(f"Could not delete resources: {str(e)}")
            resources_deleted = 0
        
        # Delete related billing records (if the table exists)
        try:
            cursor.execute("DELETE FROM public.service_order_billing WHERE service_order_id = %s", (service_order_id,))
            billing_deleted = cursor.rowcount
        except Exception as e:
            current_app.logger.warning(f"Could not delete billing records: {str(e)}")
            billing_deleted = 0
        
        # Delete the service order itself
        cursor.execute("DELETE FROM public.service_orders WHERE id = %s", (service_order_id,))
        
        # Commit the transaction
        conn.commit()
        
        return jsonify({
            "success": True,
            "message": f"Service order {order_id} and related data deleted successfully",
            "details": {
                "deliverables_deleted": deliverables_deleted,
                "resources_deleted": resources_deleted,
                "billing_records_deleted": billing_deleted
            }
        }), 200
        
    except Exception as e:
        conn.rollback()
        current_app.logger.error(f"Error deleting service order: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

# PART 3: DELIVERABLES, RESOURCES, AND BILLING ROUTES
# ===================================================================

@dashboard_package.route('/service-orders/<order_id>/deliverables', methods=['GET'])
def get_service_order_deliverables(order_id):
    """
    Get all deliverables for a specific service order.
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "error": "Database connection failed"}), 500
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # First, verify that the service order exists
        cursor.execute("SELECT id FROM public.service_orders WHERE order_id = %s", (order_id,))
        service_order = cursor.fetchone()
        
        if not service_order:
            return jsonify({"success": False, "error": f"Service order {order_id} not found"}), 404
        
        # Get deliverables for this service order
        cursor.execute("""
            SELECT * FROM public.service_order_deliverables
            WHERE service_order_id = %s
            ORDER BY deliverable_number
        """, (service_order['id'],))
        
        deliverables = cursor.fetchall()
        
        # Format dates for JSON serialization
        for deliverable in deliverables:
            if 'due_date' in deliverable and deliverable['due_date']:
                deliverable['due_date'] = deliverable['due_date'].isoformat()
            if 'created_at' in deliverable and deliverable['created_at']:
                deliverable['created_at'] = deliverable['created_at'].isoformat()
            if 'updated_at' in deliverable and deliverable['updated_at']:
                deliverable['updated_at'] = deliverable['updated_at'].isoformat()
        
        return jsonify({
            "success": True,
            "data": deliverables,
            "order_id": order_id
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error fetching deliverables: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@dashboard_package.route('/service-orders/<order_id>/deliverables', methods=['POST'])
def add_service_order_deliverable(order_id):
    """
    Add a new deliverable to a service order.
    
    Required fields:
    - title: Title of the deliverable
    
    Optional fields:
    - deliverable_number: Identifier for the deliverable (auto-assigned if not provided)
    - description: Detailed description
    - due_date: Due date for the deliverable
    - status: Status of the deliverable (default: 'Pending')
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "error": "Database connection failed"}), 500
    
    try:
        data = request.json
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Validate required fields
        if 'title' not in data:
            return jsonify({"success": False, "error": "Missing required field: title"}), 400
        
        cursor = conn.cursor()
        
        # First, verify that the service order exists
        cursor.execute("SELECT id FROM public.service_orders WHERE order_id = %s", (order_id,))
        service_order = cursor.fetchone()
        
        if not service_order:
            return jsonify({"success": False, "error": f"Service order {order_id} not found"}), 404
        
        service_order_id = service_order[0]
        
        # If deliverable_number is not provided, assign the next available number
        if 'deliverable_number' not in data or not data['deliverable_number']:
            cursor.execute("""
                SELECT MAX(CAST(deliverable_number AS INTEGER)) 
                FROM public.service_order_deliverables 
                WHERE service_order_id = %s
            """, (service_order_id,))
            
            max_number = cursor.fetchone()[0]
            deliverable_number = str(1 if max_number is None else max_number + 1)
        else:
            deliverable_number = data['deliverable_number']
        
        # Parse due_date if provided
        due_date = None
        if 'due_date' in data and data['due_date']:
            try:
                if isinstance(data['due_date'], str):
                    due_date = datetime.fromisoformat(data['due_date'].replace('Z', '+00:00')).date()
                else:
                    due_date = data['due_date']
            except ValueError:
                return jsonify({
                    "success": False, 
                    "error": "Invalid date format for due_date"
                }), 400
        
        # Insert the deliverable
        cursor.execute("""
            INSERT INTO public.service_order_deliverables (
                service_order_id, deliverable_number, title,
                description, due_date, status
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            service_order_id,
            deliverable_number,
            data['title'],
            data.get('description'),
            due_date,
            data.get('status', 'Pending')
        ))
        
        deliverable_id = cursor.fetchone()[0]
        conn.commit()
        
        return jsonify({
            "success": True,
            "message": "Deliverable added successfully",
            "deliverable_id": deliverable_id,
            "deliverable_number": deliverable_number,
            "order_id": order_id
        }), 201
        
    except Exception as e:
        conn.rollback()
        current_app.logger.error(f"Error adding deliverable: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@dashboard_package.route('/service-orders/<order_id>/resources', methods=['GET'])
def get_service_order_resources(order_id):
    """
    Get all resources assigned to a specific service order.
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "error": "Database connection failed"}), 500
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # First, verify that the service order exists
        cursor.execute("SELECT id FROM public.service_orders WHERE order_id = %s", (order_id,))
        service_order = cursor.fetchone()
        
        if not service_order:
            return jsonify({"success": False, "error": f"Service order {order_id} not found"}), 404
        
        # Get resources for this service order
        cursor.execute("""
            SELECT * FROM public.service_order_resources
            WHERE service_order_id = %s
        """, (service_order['id'],))
        
        resources = cursor.fetchall()
        
        # Format dates for JSON serialization
        for resource in resources:
            if 'start_date' in resource and resource['start_date']:
                resource['start_date'] = resource['start_date'].isoformat()
            if 'end_date' in resource and resource['end_date']:
                resource['end_date'] = resource['end_date'].isoformat()
            if 'created_at' in resource and resource['created_at']:
                resource['created_at'] = resource['created_at'].isoformat()
            if 'updated_at' in resource and resource['updated_at']:
                resource['updated_at'] = resource['updated_at'].isoformat()
        
        # Calculate totals
        total_hours = sum(resource.get('allocated_hours', 0) for resource in resources)
        total_cost = sum(resource.get('allocated_hours', 0) * resource.get('hourly_rate', 0) for resource in resources)
        
        return jsonify({
            "success": True,
            "data": resources,
            "order_id": order_id,
            "summary": {
                "total_resources": len(resources),
                "total_hours": total_hours,
                "total_cost": total_cost
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error fetching resources: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@dashboard_package.route('/service-orders/<order_id>/resources', methods=['POST'])
def add_service_order_resource(order_id):
    """
    Add a new resource to a service order.
    
    Required fields:
    - role_title: Title/role of the resource
    - hourly_rate: Hourly billing rate
    - allocated_hours: Number of hours allocated
    
    Optional fields:
    - resource_name: Name of the person assigned (if known)
    - role_level: Level/seniority of the role
    - location: Location of the resource
    - start_date: Start date for the resource
    - end_date: End date for the resource
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "error": "Database connection failed"}), 500
    
    try:
        data = request.json
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ['role_title', 'hourly_rate', 'allocated_hours']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "success": False, 
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        cursor = conn.cursor()
        
        # First, verify that the service order exists
        cursor.execute("SELECT id FROM public.service_orders WHERE order_id = %s", (order_id,))
        service_order = cursor.fetchone()
        
        if not service_order:
            return jsonify({"success": False, "error": f"Service order {order_id} not found"}), 404
        
        service_order_id = service_order[0]
        
        # Parse dates if provided
        start_date = None
        if 'start_date' in data and data['start_date']:
            try:
                if isinstance(data['start_date'], str):
                    start_date = datetime.fromisoformat(data['start_date'].replace('Z', '+00:00')).date()
                else:
                    start_date = data['start_date']
            except ValueError:
                return jsonify({
                    "success": False, 
                    "error": "Invalid date format for start_date"
                }), 400
        
        end_date = None
        if 'end_date' in data and data['end_date']:
            try:
                if isinstance(data['end_date'], str):
                    end_date = datetime.fromisoformat(data['end_date'].replace('Z', '+00:00')).date()
                else:
                    end_date = data['end_date']
            except ValueError:
                return jsonify({
                    "success": False, 
                    "error": "Invalid date format for end_date"
                }), 400
        
        # Insert the resource
        cursor.execute("""
            INSERT INTO public.service_order_resources (
                service_order_id, resource_name, role_title,
                role_level, location, hourly_rate, allocated_hours,
                start_date, end_date
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            service_order_id,
            data.get('resource_name'),
            data['role_title'],
            data.get('role_level'),
            data.get('location'),
            data['hourly_rate'],
            data['allocated_hours'],
            start_date,
            end_date
        ))
        
        resource_id = cursor.fetchone()[0]
        conn.commit()
        
        # Update the order value to include this resource if needed
        if data.get('update_order_value', False):
            resource_value = float(data['hourly_rate']) * float(data['allocated_hours'])
            
            cursor.execute("""
                UPDATE public.service_orders
                SET value = COALESCE(value, 0) + %s
                WHERE id = %s
            """, (resource_value, service_order_id))
            
            conn.commit()
        
        return jsonify({
            "success": True,
            "message": "Resource added successfully",
            "resource_id": resource_id,
            "order_id": order_id
        }), 201
        
    except Exception as e:
        conn.rollback()
        current_app.logger.error(f"Error adding resource: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@dashboard_package.route('/service-orders/<order_id>/billing', methods=['GET'])
def get_service_order_billing(order_id):
    """
    Get billing records for a specific service order.
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "error": "Database connection failed"}), 500
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # First, verify that the service order exists
        cursor.execute("SELECT id, value FROM public.service_orders WHERE order_id = %s", (order_id,))
        service_order = cursor.fetchone()
        
        if not service_order:
            return jsonify({"success": False, "error": f"Service order {order_id} not found"}), 404
        
        # Get billing records for this service order
        cursor.execute("""
            SELECT * FROM public.service_order_billing
            WHERE service_order_id = %s
            ORDER BY billing_period_start DESC
        """, (service_order['id'],))
        
        billing_records = cursor.fetchall()
        
        # Format dates for JSON serialization
        for record in billing_records:
            if 'invoice_date' in record and record['invoice_date']:
                record['invoice_date'] = record['invoice_date'].isoformat()
            if 'billing_period_start' in record and record['billing_period_start']:
                record['billing_period_start'] = record['billing_period_start'].isoformat()
            if 'billing_period_end' in record and record['billing_period_end']:
                record['billing_period_end'] = record['billing_period_end'].isoformat()
            if 'payment_date' in record and record['payment_date']:
                record['payment_date'] = record['payment_date'].isoformat()
            if 'created_at' in record and record['created_at']:
                record['created_at'] = record['created_at'].isoformat()
            if 'updated_at' in record and record['updated_at']:
                record['updated_at'] = record['updated_at'].isoformat()
        
        # Calculate summaries
        total_billed = sum(record.get('amount', 0) for record in billing_records)
        remaining_budget = service_order['value'] - total_billed if service_order['value'] else None
        
        return jsonify({
            "success": True,
            "data": billing_records,
            "order_id": order_id,
            "summary": {
                "total_invoices": len(billing_records),
                "total_billed": total_billed,
                "order_value": service_order['value'],
                "remaining_budget": remaining_budget,
                "budget_utilization_percentage": (total_billed / service_order['value'] * 100) if service_order['value'] else None
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error fetching billing records: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@dashboard_package.route('/service-orders/<order_id>/billing', methods=['POST'])
def add_service_order_billing(order_id):
    """
    Add a new billing record to a service order.
    
    Required fields:
    - amount: The billed amount
    - billing_period_start: Start date of the billing period
    - billing_period_end: End date of the billing period
    
    Optional fields:
    - invoice_number: Identifier for the invoice
    - invoice_date: Date of the invoice
    - hours_billed: Number of hours billed in this invoice
    - payment_status: Status of the payment (default: 'Unpaid')
    - payment_date: Date when payment was received
    - notes: Additional notes
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "error": "Database connection failed"}), 500
    
    try:
        data = request.json
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ['amount', 'billing_period_start', 'billing_period_end']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "success": False, 
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        cursor = conn.cursor()
        
        # First, verify that the service order exists
        cursor.execute("SELECT id FROM public.service_orders WHERE order_id = %s", (order_id,))
        service_order = cursor.fetchone()
        
        if not service_order:
            return jsonify({"success": False, "error": f"Service order {order_id} not found"}), 404
        
        service_order_id = service_order[0]
        
        # Parse dates
        try:
            billing_period_start = datetime.fromisoformat(data['billing_period_start'].replace('Z', '+00:00')).date() if isinstance(data['billing_period_start'], str) else data['billing_period_start']
            billing_period_end = datetime.fromisoformat(data['billing_period_end'].replace('Z', '+00:00')).date() if isinstance(data['billing_period_end'], str) else data['billing_period_end']
            
            invoice_date = None
            if 'invoice_date' in data and data['invoice_date']:
                invoice_date = datetime.fromisoformat(data['invoice_date'].replace('Z', '+00:00')).date() if isinstance(data['invoice_date'], str) else data['invoice_date']
                
            payment_date = None
            if 'payment_date' in data and data['payment_date']:
                payment_date = datetime.fromisoformat(data['payment_date'].replace('Z', '+00:00')).date() if isinstance(data['payment_date'], str) else data['payment_date']
        except ValueError as e:
            return jsonify({
                "success": False, 
                "error": f"Invalid date format: {str(e)}"
            }), 400
        
        # Check for duplicate invoice number if provided
        if 'invoice_number' in data and data['invoice_number']:
            cursor.execute("SELECT id FROM public.service_order_billing WHERE invoice_number = %s", (data['invoice_number'],))
            if cursor.fetchone():
                return jsonify({
                    "success": False, 
                    "error": f"Invoice number {data['invoice_number']} already exists"
                }), 400
        
        # Insert the billing record
        cursor.execute("""
            INSERT INTO public.service_order_billing (
                service_order_id, invoice_number, invoice_date,
                billing_period_start, billing_period_end, amount,
                hours_billed, payment_status, payment_date, notes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            service_order_id,
            data.get('invoice_number'),
            invoice_date,
            billing_period_start,
            billing_period_end,
            data['amount'],
            data.get('hours_billed'),
            data.get('payment_status', 'Unpaid'),
            payment_date,
            data.get('notes')
        ))
        
        billing_id = cursor.fetchone()[0]
        conn.commit()
        
        return jsonify({
            "success": True,
            "message": "Billing record added successfully",
            "billing_id": billing_id,
            "order_id": order_id
        }), 201
        
    except Exception as e:
        conn.rollback()
        current_app.logger.error(f"Error adding billing record: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()     
        
   # Additional utility routes and dashboard data

@dashboard_package.route('/dashboard/service-orders', methods=['GET'])
def get_dashboard_service_orders():
    """
    Get consolidated data for service orders dashboard, including summary metrics.
    
    Optional query parameters:
    - customer_id: Filter by customer
    - provider_id: Filter by provider
    - status: Filter by status
    - year: Filter by fiscal year
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "error": "Database connection failed"}), 500
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get filter parameters
        customer_id = request.args.get('customer_id')
        provider_id = request.args.get('provider_id')
        status = request.args.get('status')
        fiscal_year = request.args.get('year')
        
        # Build the query
        query = """
            SELECT 
                so.id, 
                so.order_id, 
                so.display_name as order_name,
                so.model_type,
                so.start_date, 
                so.end_date, 
                so.value as order_value,
                so.currency,
                so.status,
                so.clm_number,
                ma.id as msa_id,
                ma.display_name as msa_name,
                p.id as provider_id,
                p.display_name as provider_name,
                c.id as customer_id,
                c.display_name as customer_name
            FROM public.service_orders so
            JOIN public.master_agreements ma ON so.msa_id = ma.id
            JOIN public.providers p ON ma.provider_id = p.id
            JOIN public.customers c ON ma.customer_id = c.id
            WHERE 1=1
        """
        
        params = []
        
        if customer_id:
            query += " AND ma.customer_id = %s"
            params.append(customer_id)
            
        if provider_id:
            query += " AND ma.provider_id = %s"
            params.append(provider_id)
            
        if status:
            query += " AND so.status = %s"
            params.append(status)
            
        # Add fiscal year filter if provided (fiscal year runs June-May)
        if fiscal_year:
            # Convert to integer fiscal year
            fy = int(fiscal_year)
            # Start date range from June 1 of previous year
            fy_start = f"{fy-1}-06-01"
            # End date range to May 31 of current year
            fy_end = f"{fy}-05-31"
            
            query += " AND ((so.start_date BETWEEN %s AND %s) OR (so.end_date BETWEEN %s AND %s))"
            params.extend([fy_start, fy_end, fy_start, fy_end])
        
        query += " ORDER BY so.created_at DESC"
        
        cursor.execute(query, params)
        orders = cursor.fetchall()
        
        # Format dates and calculate additional metrics
        formatted_orders = []
        total_value = 0
        active_orders = 0
        
        current_date = datetime.now().date()
        
        for order in orders:
            # Format dates
            start_date = order['start_date']
            end_date = order['end_date']
            
            formatted_start = start_date.strftime('%b %d, %Y') if start_date else 'N/A'
            formatted_end = end_date.strftime('%b %d, %Y') if end_date else 'TBD'
            
            # Set fiscal year based on start date
            if start_date:
                fy = start_date.year if start_date.month >= 6 else start_date.year - 1
                fiscal_year = f"FY{str(fy)[-2:]}"
                
                if end_date and (end_date.year > start_date.year or 
                                (end_date.year == start_date.year and end_date.month >= 6 and start_date.month < 6)):
                    fy_end = end_date.year if end_date.month >= 6 else end_date.year - 1
                    if fy != fy_end:
                        fiscal_year = f"FY{str(fy)[-2:]}-{str(fy_end)[-2:]}"
            else:
                fiscal_year = "N/A"
            
            # Calculate completion percentage
            if order['status'] == 'Completed':
                completion_score = 100
            elif end_date and current_date > end_date:
                completion_score = 85  # Assume nearly complete but not marked as completed
            elif start_date and end_date and current_date >= start_date:
                days_elapsed = max(0, (current_date - start_date).days)
                total_days = max(1, (end_date - start_date).days)
                completion_score = min(100, max(0, int((days_elapsed / total_days) * 100)))
            elif start_date and current_date < start_date:
                completion_score = 0
            else:
                completion_score = 0
            
            # Get resources for this order (if table exists)
            resources = []
            try:
                cursor.execute("""
                    SELECT resource_name, role_title, role_level, hourly_rate, allocated_hours
                    FROM public.service_order_resources
                    WHERE service_order_id = %s
                """, (order['id'],))
                
                db_resources = cursor.fetchall()
                for resource in db_resources:
                    resources.append({
                        "name": resource['resource_name'] or "Unnamed Resource",
                        "role": resource['role_title'],
                        "level": resource['role_level'],
                        "rate": f"${resource['hourly_rate']}/hr" if resource['hourly_rate'] else "$0/hr",
                        "hours": resource['allocated_hours'],
                        "total": f"${resource['hourly_rate'] * resource['allocated_hours']:,.2f}" if resource['hourly_rate'] and resource['allocated_hours'] else "$0.00"
                    })
            except Exception as e:
                current_app.logger.warning(f"Could not fetch resources: {str(e)}")
            
            # Get billing information
            try:
                cursor.execute("""
                    SELECT SUM(amount) as billed_amount, COUNT(*) as invoice_count
                    FROM public.service_order_billing
                    WHERE service_order_id = %s
                """, (order['id'],))
                
                billing_info = cursor.fetchone()
                billed_amount = billing_info['billed_amount'] if billing_info and billing_info['billed_amount'] else 0
                invoice_count = billing_info['invoice_count'] if billing_info else 0
                
                # Calculate budget utilization
                if order['order_value']:
                    budget_utilization = (billed_amount / order['order_value']) * 100
                else:
                    budget_utilization = 0
            except Exception as e:
                current_app.logger.warning(f"Could not fetch billing info: {str(e)}")
                billed_amount = 0
                invoice_count = 0
                budget_utilization = 0
            
            # Create the formatted order
            formatted_order = {
                "id": order['order_id'],
                "orderName": order['order_name'],
                "provider": order['provider_name'],
                "customer": order['customer_name'],
                "master": order['msa_name'],
                "status": order['status'],
                "startDate": formatted_start,
                "endDate": formatted_end,
                "orderValue": f"${order['order_value']:,.2f}" if order['order_value'] else "$0.00",
                "billedAmount": f"${billed_amount:,.2f}" if billed_amount else "$0.00",
                "invoiceCount": invoice_count,
                "fiscalYear": fiscal_year,
                "completionScore": f"{completion_score}%",
                "resources": resources,
                "metrics": [
                    {"name": "Completion", "value": completion_score, "unit": "%", "target": 100},
                    {"name": "Budget Utilization", "value": round(budget_utilization, 1), "unit": "%", "target": 100},
                    {"name": "Resource Count", "value": len(resources), "unit": "", "target": None}
                ]
            }
            
            formatted_orders.append(formatted_order)
            
            # Update summary metrics
            if order['order_value']:
                total_value += float(order['order_value'])
            
            if order['status'] == 'Active':
                active_orders += 1
        
        # Group by provider for spending summary
        provider_spending = {}
        for order in orders:
            if not order['order_value']:
                continue
                
            provider_name = order['provider_name']
            if provider_name not in provider_spending:
                provider_spending[provider_name] = 0
                
            provider_spending[provider_name] += float(order['order_value'])
        
        # Group by fiscal year
        fy_spending = {}
        for order in formatted_orders:
            if 'fiscalYear' not in order or order['fiscalYear'] == 'N/A':
                continue
                
            fy = order['fiscalYear']
            value = float(order['orderValue'].replace('$', '').replace(',', ''))
            
            if fy not in fy_spending:
                fy_spending[fy] = 0
                
            fy_spending[fy] += value
        
        # Format provider spending for the response
        providers = [{"name": k, "spend": v} for k, v in provider_spending.items()]
        
        # Build the final response
        response = {
            "lastUpdated": datetime.now().isoformat(),
            "serviceOrders": formatted_orders,
            "providers": providers,
            "summary": {
                "totalOrders": len(orders),
                "activeOrders": active_orders,
                "totalValue": total_value,
                "spendByFY": fy_spending
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting dashboard data: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@dashboard_package.route('/service-orders/stats', methods=['GET'])
def get_service_order_stats():
    """
    Get statistical information about service orders.
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "error": "Database connection failed"}), 500
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Count orders by status
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM public.service_orders
            GROUP BY status
            ORDER BY count DESC
        """)
        
        status_counts = cursor.fetchall()
        
        # Get top providers by order count
        cursor.execute("""
            SELECT p.display_name as provider, COUNT(*) as order_count
            FROM public.service_orders so
            JOIN public.master_agreements ma ON so.msa_id = ma.id
            JOIN public.providers p ON ma.provider_id = p.id
            GROUP BY p.display_name
            ORDER BY order_count DESC
            LIMIT 5
        """)
        
        top_providers_by_count = cursor.fetchall()
        
        # Get top providers by order value
        cursor.execute("""
            SELECT p.display_name as provider, SUM(so.value) as total_value
            FROM public.service_orders so
            JOIN public.master_agreements ma ON so.msa_id = ma.id
            JOIN public.providers p ON ma.provider_id = p.id
            WHERE so.value IS NOT NULL
            GROUP BY p.display_name
            ORDER BY total_value DESC
            LIMIT 5
        """)
        
        top_providers_by_value = cursor.fetchall()
        
        # Get average order value
        cursor.execute("""
            SELECT AVG(value) as avg_value
            FROM public.service_orders
            WHERE value IS NOT NULL
        """)
        
        avg_value = cursor.fetchone()['avg_value'] or 0
        
        # Get average order duration
        cursor.execute("""
            SELECT AVG(end_date - start_date) as avg_duration
            FROM public.service_orders
            WHERE start_date IS NOT NULL AND end_date IS NOT NULL
        """)
        
        avg_duration_result = cursor.fetchone()
        avg_duration = avg_duration_result['avg_duration'].days if avg_duration_result and avg_duration_result['avg_duration'] else 0
        
        # Get monthly spending trend (last 12 months)
        cursor.execute("""
            SELECT 
                DATE_TRUNC('month', start_date) as month,
                SUM(value) as monthly_value
            FROM public.service_orders
            WHERE start_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '11 months')
            GROUP BY month
            ORDER BY month
        """)
        
        monthly_trend = cursor.fetchall()
        
        # Format the monthly trend for JSON response
        formatted_trend = []
        for item in monthly_trend:
            if item['month']:
                formatted_trend.append({
                    "month": item['month'].strftime('%b %Y'),
                    "value": float(item['monthly_value']) if item['monthly_value'] else 0
                })
        
        # Build the response
        stats = {
            "ordersByStatus": status_counts,
            "topProvidersByCount": top_providers_by_count,
            "topProvidersByValue": top_providers_by_value,
            "averageOrderValue": float(avg_value),
            "averageOrderDuration": avg_duration,
            "monthlyTrend": formatted_trend,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify({"success": True, "data": stats}), 200
        
    except Exception as e:
        current_app.logger.error(f"Error fetching service order stats: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

# Test route to verify the API is working
@dashboard_package.route('/test', methods=['GET'])
def test_service_orders_api():
    """
    A simple test endpoint to verify the service orders API is working.
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
        "message": "The service orders API is set up and active.",
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    }), 200        
