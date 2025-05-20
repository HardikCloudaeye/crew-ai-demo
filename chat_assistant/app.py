# app.py with support for RAG functionality
from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash
from tenant_assistant import WebChatbot, TenantConfig
import os
import traceback

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize tenant config
tenant_config = TenantConfig()


@app.route('/')
def index():
    # Clear any existing session if user is returning to the tenant selection page
    if 'tenant_id' in session:
        session.pop('tenant_id')
    if 'tenant_name' in session:
        session.pop('tenant_name')

    tenant_ids = list(tenant_config.tenants.keys())
    tenant_names = [tenant_config.tenants[id]["name"] for id in tenant_ids]
    tenants = list(zip(tenant_ids, tenant_names))
    return render_template('index.html', tenants=tenants)


@app.route('/login', methods=['POST'])
def login():
    tenant_id = request.form.get('tenant_id')

    if tenant_id in tenant_config.tenants:
        # Store tenant_id in session but don't authenticate yet
        session['temp_tenant_id'] = tenant_id
        return jsonify({"success": True, "redirect": "/password"})

    return jsonify({"success": False, "error": "Invalid tenant"})


@app.route('/password', methods=['GET', 'POST'])
def password():
    if 'temp_tenant_id' not in session:
        return redirect(url_for('index'))

    tenant_id = session['temp_tenant_id']
    tenant_name = tenant_config.tenants[tenant_id]['name']

    if request.method == 'POST':
        password = request.form.get('password', '')
        if tenant_config.verify_tenant_password(tenant_id, password):
            # Password verified, set session variables for authenticated user
            session['tenant_id'] = tenant_id
            session['tenant_name'] = tenant_name
            # Remove temporary tenant ID
            session.pop('temp_tenant_id', None)
            return redirect(url_for('chat'))
        else:
            flash('Incorrect password. Please try again.')

    return render_template('password.html', tenant_name=tenant_name)


@app.route('/logout')
def logout():
    # Clear session variables
    session.pop('tenant_id', None)
    session.pop('tenant_name', None)
    session.pop('temp_tenant_id', None)
    return redirect(url_for('index'))


@app.route('/chat')
def chat():
    if 'tenant_id' not in session:
        return redirect(url_for('index'))

    return render_template('chat.html', tenant_name=session['tenant_name'])


@app.route('/api/query', methods=['POST'])
def handle_query():
    if 'tenant_id' not in session:
        return jsonify({"error": "Not logged in"})

    data = request.get_json()
    query = data.get('query', '')
    tenant_id = session['tenant_id']

    try:
        # Initialize chatbot for this tenant
        chatbot = WebChatbot(tenant_id)

        # Process query
        response = chatbot.process_query(query)

        return jsonify({
            "response": response,
            "tenant": session['tenant_name']
        })
    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        print(error_message)
        traceback.print_exc()

        return jsonify({
            "response": f"I apologize, but I'm experiencing technical difficulties. Error: {str(e)}",
            "tenant": session['tenant_name'],
            "error": True
        })


if __name__ == '__main__':
    app.run(debug=True)
