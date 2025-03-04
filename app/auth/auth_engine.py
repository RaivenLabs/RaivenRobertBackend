from flask import Blueprint, request, jsonify
import boto3

auth_blueprint = Blueprint('auth', __name__)  # No prefix

COGNITO_CONFIG = {
   'user_pool_id': 'us-west-2_JnwmUHLzc',
   'client_id': '47l1p0md5fpj3adg86no3h0sk6',
   'region': 'us-west-2'
}

def get_cognito_client():
   return boto3.client('cognito-idp', region_name=COGNITO_CONFIG['region'])

def get_user_groups(access_token):
    print("\n📋 Starting get_user_groups function...")
    try:
        print("🔄 Getting Cognito client for group fetch...")
        cognito_client = get_cognito_client()
        
        print("🔍 Getting user info using access token...")
        user_info = cognito_client.get_user(AccessToken=access_token)
        print(f"👤 Found user: {user_info.get('Username', 'Unknown')}")
        
        print(f"🔍 Fetching groups...")
        groups_response = cognito_client.admin_list_groups_for_user(
            Username=user_info['Username'],
            UserPoolId=COGNITO_CONFIG['user_pool_id']
        )
        print("📝 Raw groups response:", groups_response)
        
        groups = [group['GroupName'] for group in groups_response['Groups']]
        print(f"✨ Processed groups: {groups}")
        return groups
        
    except Exception as e:
        print(f"❌ Error in get_user_groups: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print("❌ Traceback:", traceback.format_exc())
        return []
    
@auth_blueprint.route('/api/auth/cognito-auth', methods=['POST'])
def cognito_auth():
    print("\n" + "="*50)
    print(f"🚀 FOUND ROUTE: {request.url}")
    print(f"🛣️  ENDPOINT: {request.endpoint}")
    print("="*50 + "\n")
    
    try:
        data = request.get_json()
        print("📝 DEBUG: Received data:", data)
        print("🔑 Attempting authentication...")
        
        identifier = data.get('identifier')
        password = data.get('password')
        
        if not identifier or not password:
            print("❌ Missing credentials")
            return jsonify({
                'error': 'Username/email and password are required'
            }), 400
            
        cognito_client = get_cognito_client()
        print("📞 Got Cognito client, initiating auth...")
        
        try:
            print(f"🔄 Starting auth flow for user: {identifier}")
            auth_response = cognito_client.initiate_auth(
                ClientId=COGNITO_CONFIG['client_id'],
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                    'USERNAME': identifier,
                    'PASSWORD': password
                }
            )
            print("✅ Initial Cognito auth successful!")
            print("🔍 Auth response received:", auth_response.get('AuthenticationResult', {}))
            
            try:
                access_token = auth_response['AuthenticationResult']['AccessToken']
                print(f"🎟️ Access token obtained: {access_token[:10]}...")
                
                print("👥 Starting group fetch process...")
                user_groups = get_user_groups(access_token)
                print(f"👥 Groups fetch completed. Groups: {user_groups}")
                
                response_data = {
                    'success': True,
                    'accessToken': access_token,
                    'idToken': auth_response['AuthenticationResult']['IdToken'],
                    'refreshToken': auth_response['AuthenticationResult']['RefreshToken'],
                    'expiresIn': auth_response['AuthenticationResult']['ExpiresIn'],
                    'userGroups': user_groups
                }
                print("📤 Sending response with groups:", response_data.get('userGroups'))
                return jsonify(response_data), 200
                
            except Exception as token_error:
                print(f"❌ Error processing tokens or groups: {str(token_error)}")
                print(f"❌ Error type: {type(token_error).__name__}")
                raise
            
        except Exception as auth_error:
            print(f"❌ Cognito authentication error: {str(auth_error)}")
            print(f"❌ Error type: {type(auth_error).__name__}")
            return jsonify({
                'error': f'Authentication failed: {str(auth_error)}'
            }), 500
            
    except Exception as e:
        print(f"❌ General server error: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

@auth_blueprint.route('/api/auth/logout', methods=['POST'])  # Standardized route pattern
def logout():
   print("\n" + "="*50)  # Create a visual separator line
   print(f"🚀 FOUND ROUTE: {request.url}")  # Shows full URL including host
   print(f"🛣️  ENDPOINT: {request.endpoint}")  # Shows route name
   print("="*50 + "\n")  # Close the visual separator
   
   try:
       return jsonify({'message': 'Successfully logged out'}), 200
   except Exception as e:
       print(f"DEBUG: Logout error: {str(e)}")
       return jsonify({
           'error': f'Logout failed: {str(e)}'
       }), 500
