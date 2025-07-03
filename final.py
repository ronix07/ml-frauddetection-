import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import time
import re
import urllib.parse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import tldextract
import requests
from bs4 import BeautifulSoup
import socket
import datetime
import ssl
import hashlib

# Set page configuration
def main():
    st.set_page_config(
        page_title="Fraud Guardian | Payment Fraud Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for modern look
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .fraud-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar with navigation and logo
    with st.sidebar:
        st.markdown("# 🛡️ Fraud Guardian")
        st.markdown("---")
        
        tab = st.radio("Navigation", ["Dashboard", "Fraud Detection", "Analytics", "URL Phishing"])
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("Fraud Guardian protects your business from payment fraud using advanced machine learning algorithms.")
        
        # Add a fake "logged in" indicator for visual appeal
        st.markdown("---")
        st.markdown("👤 **Logged in as:** Admin User")
    
    if tab == "Dashboard":
        show_dashboard()
    elif tab == "Fraud Detection":
        show_fraud_detection()
    elif tab == "Analytics":
        show_analytics()
    elif tab == "URL Phishing":
        show_url_phishing()

def show_dashboard():
    st.markdown("<h1>Payment Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="fraud-card">
            <h3>Transactions</h3>
            <h2>14,382</h2>
            <p>+4.6% from last month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="fraud-card">
            <h3>Fraud Detected</h3>
            <h2>246</h2>
            <p>-2.3% from last month</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="fraud-card">
            <h3>Money Saved</h3>
            <h2>$189,245</h2>
            <p>+12.7% from last month</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="fraud-card">
            <h3>Detection Rate</h3>
            <h2>98.7%</h2>
            <p>+0.5% from last month</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Transaction volume chart
    st.markdown("<h2>Transaction Volume</h2>", unsafe_allow_html=True)
    
    # Sample data for charts
    chart_data = pd.DataFrame({
        'date': pd.date_range(start='2025-03-01', periods=30, freq='D'),
        'transactions': np.random.randint(300, 600, 30),
        'fraud': np.random.randint(5, 20, 30)
    })
    
    # Plotly chart
    fig = px.line(chart_data, x='date', y=['transactions', 'fraud'], 
                 title='Transaction Volume - March 2025',
                 labels={'value': 'Count', 'variable': 'Type'},
                 color_discrete_map={'transactions': '#1E3A8A', 'fraud': '#DC2626'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.markdown("<h2>Recent Activity</h2>", unsafe_allow_html=True)
    
    recent_data = pd.DataFrame({
        'Timestamp': pd.date_range(start='2025-03-31 09:00:00', periods=5, freq='30min'),
        'Transaction ID': ['TX-' + str(np.random.randint(10000, 99999)) for _ in range(5)],
        'Amount': np.random.randint(100, 5000, 5),
        'Type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH-OUT'], 5),
        'Status': np.random.choice(['✅ Legitimate', '⚠️ Suspicious', '❌ Fraudulent'], 5, p=[0.7, 0.2, 0.1])
    })
    
    st.dataframe(recent_data, use_container_width=True, hide_index=True)

def show_fraud_detection():
    st.markdown("<h1>Fraud Detection Service</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="fraud-card">
            <h3>Transaction Analysis</h3>
            <p>Enter the details of a transaction to analyze it for potential fraud.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load the trained Random Forest model (with error handling)
        try:
            model = joblib.load("fraud_detection_rf_model.pkl")
        except:
            st.warning("Model file not found. Using a mock prediction model instead.")
            # Create a mock model for demonstration
            class MockModel:
                def predict(self, X):
                    # Mock prediction based on amount and type
                    type_val = X[0][0]
                    amount = X[0][1]
                    if amount > 5000 and type_val in [1, 4]:  # CASH-OUT or TRANSFER with high amount
                        return np.array([1])
                    elif amount > 10000:  # Very high amount
                        return np.array([1])
                    else:
                        return np.array([0])
            model = MockModel()
        
        # User input form with improved layout
        with st.form("transaction_form"):
            # More intuitive form layout
            type_col, amount_col = st.columns(2)
            with type_col:
                transaction_type = st.selectbox(
                    "Transaction Type", 
                    ["PAYMENT", "TRANSFER", "CASH-IN", "CASH-OUT", "DEBIT"], 
                    index=0
                )
            
            with amount_col:
                amount = st.number_input(
                    "Transaction Amount ($)", 
                    min_value=0.0, 
                    max_value=1000000.0,
                    step=100.0, 
                    value=1000.0,
                    format="%.2f"
                )
            
            balance_col1, balance_col2 = st.columns(2)
            with balance_col1:
                oldbalanceOrg = st.number_input(
                    "Origin Account Balance ($)", 
                    min_value=0.0, 
                    step=100.0, 
                    value=5000.0,
                    format="%.2f"
                )
            
            with balance_col2:
                oldbalanceDest = st.number_input(
                    "Destination Account Balance ($)", 
                    min_value=0.0, 
                    step=100.0, 
                    value=2000.0,
                    format="%.2f"
                )
            
            # Automatically calculate newbalances (for realism)
            if transaction_type in ["TRANSFER", "PAYMENT", "CASH-OUT"]:
                newbalanceOrg = max(0, oldbalanceOrg - amount)
                newbalanceDest = oldbalanceDest + amount if transaction_type != "CASH-OUT" else oldbalanceDest
            elif transaction_type == "CASH-IN":
                newbalanceOrg = oldbalanceOrg + amount
                newbalanceDest = oldbalanceDest
            else:  # DEBIT
                newbalanceOrg = max(0, oldbalanceOrg - amount)
                newbalanceDest = oldbalanceDest
            
            # System flag based on amount threshold
            isFlaggedFraud = 1 if amount > 200000 else 0
            
            submitted = st.form_submit_button("Analyze Transaction")
    
    with col2:
        st.markdown("""
        <div class="fraud-card">
            <h3>Risk Factors</h3>
            <ul>
                <li>Transaction type</li>
                <li>Transaction amount</li>
                <li>Account balances</li>
                <li>Transaction pattern</li>
                <li>Time of transaction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if submitted:
        # Add a loading effect
        with st.spinner("Analyzing transaction..."):
            time.sleep(1.5)  # Simulate processing time
            
            # Prepare the input data
            type_mapping = {"CASH-IN": 0, "CASH-OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}
            type_encoded = type_mapping[transaction_type]
            features = np.array([[type_encoded, amount, oldbalanceOrg, newbalanceOrg, oldbalanceDest, newbalanceDest, isFlaggedFraud]])
            
            # For demo purposes, use only the first 5 features if the mock model is being used
            prediction_features = features[0][:5].reshape(1, -1)
            
            # Get prediction
            prediction = model.predict(prediction_features)[0]
            
            # Calculate a fake confidence score
            if prediction == 1:
                confidence = min(95, 65 + (amount / 1000))
            else:
                confidence = min(98, 80 - (amount / 10000))
            
            # Display result with animation
            result_col1, result_col2 = st.columns([3, 1])
            
            with result_col1:
                if prediction == 1:
                    st.error(f"⚠️ Fraudulent Transaction Detected! ({confidence:.1f}% confidence)")
                    st.markdown("""
                    <div style="background-color: #FEE2E2; padding: 15px; border-radius: 8px; border-left: 5px solid #DC2626;">
                        <h4 style="color: #DC2626; margin-top: 0;">Risk Factors Identified:</h4>
                        <ul>
                            <li>Unusual transaction amount</li>
                            <li>Suspicious balance changes</li>
                            <li>Pattern matches known fraud scenarios</li>
                        </ul>
                        <p><strong>Recommended Action:</strong> Block transaction and review account activity</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"✅ Legitimate Transaction ({confidence:.1f}% confidence)")
                    st.markdown("""
                    <div style="background-color: #ECFDF5; padding: 15px; border-radius: 8px; border-left: 5px solid #10B981;">
                        <h4 style="color: #10B981; margin-top: 0;">Transaction Appears Normal:</h4>
                        <p>All risk indicators are within acceptable ranges.</p>
                        <p><strong>Recommended Action:</strong> Process transaction normally</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with result_col2:
                # Create a gauge chart for confidence visualization
                fig = px.pie(values=[confidence, 100-confidence], 
                             names=['Confidence', ''],
                             hole=0.7, 
                             color_discrete_sequence=[
                                 '#10B981' if prediction == 0 else '#DC2626', 
                                 '#E5E7EB'
                             ])
                fig.update_layout(
                    showlegend=False,
                    annotations=[dict(text=f"{confidence:.1f}%", x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                st.plotly_chart(fig)
        
        # Transaction details summary
        st.markdown("<h3>Transaction Details</h3>", unsafe_allow_html=True)
        details_data = {
            "Parameter": ["Transaction Type", "Amount", "Origin Balance (Before)", "Origin Balance (After)", 
                         "Destination Balance (Before)", "Destination Balance (After)", "System Flag"],
            "Value": [transaction_type, f"${amount:,.2f}", f"${oldbalanceOrg:,.2f}", 
                     f"${newbalanceOrg:,.2f}", f"${oldbalanceDest:,.2f}", 
                     f"${newbalanceDest:,.2f}", "Yes" if isFlaggedFraud else "No"]
        }
        st.table(pd.DataFrame(details_data))

def show_analytics():
    st.markdown("<h1>Fraud Analytics</h1>", unsafe_allow_html=True)
    
    # Tabs for different analytics views
    tab1, tab2, tab3 = st.tabs(["Fraud Patterns", "Transaction Types", "Risk Scoring"])
    
    with tab1:
        st.markdown("<h3>Fraud Patterns by Time of Day</h3>", unsafe_allow_html=True)
        
        # Generate sample data for time-based fraud patterns
        hours = list(range(24))
        legitimate_counts = [100 + int(150 * np.sin(h/3)) + np.random.randint(-20, 20) for h in hours]
        fraud_counts = [5 + int(15 * np.sin((h-2)/3)) + np.random.randint(-3, 5) for h in hours]
        
        # Calculate fraud percentage
        fraud_percentage = [100 * f/(f+l) for f, l in zip(fraud_counts, legitimate_counts)]
        
        # Create DataFrame
        time_data = pd.DataFrame({
            'Hour': hours,
            'Legitimate Transactions': legitimate_counts,
            'Fraudulent Transactions': fraud_counts,
            'Fraud Percentage': fraud_percentage
        })
        
        # Create dual-axis chart
        fig = px.line(time_data, x='Hour', y=['Legitimate Transactions', 'Fraudulent Transactions'],
                     labels={'value': 'Count', 'variable': 'Type'},
                     color_discrete_map={
                         'Legitimate Transactions': '#1E3A8A',
                         'Fraudulent Transactions': '#DC2626'
                     })
        
        # Add percentage line on secondary y-axis
        fig2 = px.line(time_data, x='Hour', y='Fraud Percentage', color_discrete_sequence=['#10B981'])
        fig2.update_traces(yaxis="y2")
        
        # Add figure traces to first figure
        for trace in fig2.data:
            fig.add_trace(trace)
            
        # Update layout for dual y-axes
        fig.update_layout(
            yaxis2=dict(
                title="Fraud Percentage (%)",
                overlaying="y",
                side="right",
                range=[0, max(fraud_percentage) * 1.2]
            ),
            title="Transaction Patterns by Hour of Day",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="fraud-card">
            <h4>Key Insights:</h4>
            <ul>
                <li>Highest fraud rates occur between 1AM-4AM when monitoring is typically lower</li>
                <li>Legitimate transaction volume peaks during business hours</li>
                <li>Fraud attempts increase on weekends and holidays</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h3>Fraud by Transaction Type</h3>", unsafe_allow_html=True)
        
        # Sample data for transaction types
        type_data = pd.DataFrame({
            'Type': ['PAYMENT', 'TRANSFER', 'CASH-OUT', 'CASH-IN', 'DEBIT'],
            'Total': [4500, 3800, 2200, 2100, 1800],
            'Fraud Count': [120, 180, 170, 20, 60],
        })
        
        # Calculate fraud rate
        type_data['Fraud Rate'] = (type_data['Fraud Count'] / type_data['Total'] * 100).round(2)
        
        # Create horizontal bar chart
        fig = px.bar(
            type_data.sort_values('Fraud Rate', ascending=True), 
            y='Type', 
            x='Fraud Rate',
            color='Fraud Rate',
            color_continuous_scale=['#10B981', '#FBBF24', '#DC2626'],
            text='Fraud Rate'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(title="Fraud Rate by Transaction Type")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the data table
        st.dataframe(type_data, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("<h3>Risk Score Distribution</h3>", unsafe_allow_html=True)
        
        # Create sample data for risk scores
        np.random.seed(42)
        risk_scores = np.concatenate([
            np.random.normal(30, 15, 800),  # Low risk
            np.random.normal(70, 15, 200)   # High risk
        ])
        risk_scores = np.clip(risk_scores, 0, 100)
        
        # Create histogram
        fig = px.histogram(
            risk_scores, 
            nbins=20,
            color_discrete_sequence=['#1E40AF'],
            labels={'value': 'Risk Score', 'count': 'Number of Transactions'},
            title="Distribution of Risk Scores"
        )
        
        # Add vertical lines for risk thresholds
        fig.add_vline(x=40, line_dash="dash", line_color="#10B981", annotation_text="Low Risk Threshold")
        fig.add_vline(x=70, line_dash="dash", line_color="#DC2626", annotation_text="High Risk Threshold")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk score explanation
        st.markdown("""
        <div class="fraud-card">
            <h4>Risk Score Explanation:</h4>
            <ul>
                <li><strong>0-40:</strong> Low risk transactions - processed automatically</li>
                <li><strong>41-70:</strong> Medium risk - additional verification may be required</li>
                <li><strong>71-100:</strong> High risk - manual review required</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Define feature extraction functions for URL phishing detection
def extract_features(url):
    """Extract features from the URL for phishing detection"""
    features = {}
    
    # Parse the URL
    parsed_url = urllib.parse.urlparse(url)
    
    # Extract domain information
    domain_info = tldextract.extract(url)
    domain = domain_info.domain
    suffix = domain_info.suffix
    subdomain = domain_info.subdomain
    
    # 1. URL length
    features['url_length'] = len(url)
    
    # 2. Domain length
    features['domain_length'] = len(domain)
    
    # 3. Number of dots in URL
    features['dots_count'] = url.count('.')
    
    # 4. Number of special characters
    features['special_chars_count'] = len(re.findall(r'[^a-zA-Z0-9.]', url))
    
    # 5. Has IP address as hostname
    features['has_ip_address'] = 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', parsed_url.netloc) else 0
    
    # 6. Has '@' symbol in URL
    features['has_at_symbol'] = 1 if '@' in url else 0
    
    # 7. Has subdomain
    features['has_subdomain'] = 1 if len(subdomain) > 0 else 0
    
    # 8. URL contains 'https'
    features['has_https'] = 1 if url.startswith('https://') else 0
    
    # 9. Has suspicious words
    suspicious_words = ['login', 'signin', 'verify', 'secure', 'account', 'password', 'secure', 'ebay', 'paypal', 'bank', 'secure']
    features['has_suspicious_words'] = 1 if any(word in url.lower() for word in suspicious_words) else 0
    
    # 10. Domain age in days (if available)
    features['domain_age_days'] = -1  # Default if cannot be determined
    
    # 11. Path length
    features['path_length'] = len(parsed_url.path)
    
    # 12. Number of query parameters
    features['query_params_count'] = len(urllib.parse.parse_qs(parsed_url.query))
    
    # 13. Has port number
    features['has_port'] = 1 if parsed_url.port is not None else 0
    
    # 14. Hostname length
    features['hostname_length'] = len(parsed_url.netloc)
    
    # 15. Has hyphen in domain
    features['has_hyphen'] = 1 if '-' in domain else 0
    
    # 16. Uses 'bit.ly' or other shorteners
    shorteners = ['bit.ly', 'goo.gl', 't.co', 'tinyurl', 'tiny.cc', 'is.gd']
    features['is_shortened'] = 1 if any(shortener in parsed_url.netloc for shortener in shorteners) else 0
      
    return features

# Try to get domain age if whois is available
def get_domain_age(domain):
    try:
        import whois
        w = whois.whois(domain)
        if w.creation_date:
            # Handle cases where creation_date might be a list
            if isinstance(w.creation_date, list):
                creation_date = w.creation_date[0]
            else:
                creation_date = w.creation_date
            
            # Calculate the domain age in days
            days = (datetime.datetime.now() - creation_date).days
            return days
    except:
        pass
    return -1  # Return -1 if age cannot be determined

# Define the model or use a pre-trained one
def create_model():
    """Create and train a basic phishing detection model with sample data"""
    # You would ideally load a proper trained model here
    # This is a simplified model for demonstration
    
    # Create some synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features
    data = {
        'url_length': np.random.normal(60, 30, n_samples),
        'domain_length': np.random.normal(10, 5, n_samples),
        'dots_count': np.random.poisson(2, n_samples),
        'special_chars_count': np.random.poisson(3, n_samples),
        'has_ip_address': np.random.binomial(1, 0.1, n_samples),
        'has_at_symbol': np.random.binomial(1, 0.05, n_samples),
        'has_subdomain': np.random.binomial(1, 0.3, n_samples),
        'has_https': np.random.binomial(1, 0.7, n_samples),
        'has_suspicious_words': np.random.binomial(1, 0.2, n_samples),
        'domain_age_days': np.random.gamma(400, 100, n_samples),
        'path_length': np.random.poisson(5, n_samples),
        'query_params_count': np.random.poisson(1, n_samples),
        'has_port': np.random.binomial(1, 0.02, n_samples),
        'hostname_length': np.random.normal(20, 8, n_samples),
        'has_hyphen': np.random.binomial(1, 0.1, n_samples),
        'is_shortened': np.random.binomial(1, 0.05, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate synthetic labels with more sophisticated rules
    # Higher chance of phishing if it has suspicious characteristics
    p_phishing = (
        0.1 +  # base rate
        0.3 * df['has_ip_address'] +
        0.2 * df['has_at_symbol'] +
        0.1 * df['has_suspicious_words'] +
        0.1 * df['is_shortened'] +
        0.1 * df['has_hyphen'] +
        -0.2 * df['has_https'] +  # https reduces phishing probability
        -0.1 * np.clip(df['domain_age_days'] / 1000, 0, 1)  # older domains less likely to be phishing
    )
    p_phishing = np.clip(p_phishing, 0.05, 0.95)  # bound probabilities
    
    df['is_phishing'] = np.random.binomial(1, p_phishing)
    
    # Train a RandomForest model
    X = df.drop('is_phishing', axis=1)
    y = df['is_phishing']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Return model and feature names
    return {
        'model': model,
        'features': X.columns.tolist()
    }

@st.cache_resource
def load_model():
    """Load or create the model"""
    try:
        # Try to load a pre-trained model if it exists
        with open("phishing_detection_model.pkl", "rb") as f:
            model_package = pickle.load(f)
        return model_package
    except:
        # If no model exists, create one
        return create_model()

def check_url_security(url):
    """Perform additional security checks on the URL"""
    security_info = {}
    
    # Parse the URL
    parsed_url = urllib.parse.urlparse(url)
    domain_info = tldextract.extract(url)
    full_domain = f"{domain_info.domain}.{domain_info.suffix}"
    
    try:
        # Check if domain resolves (DNS check)
        ip_address = socket.gethostbyname(full_domain)
        security_info['resolves_to_ip'] = ip_address
        security_info['dns_resolution'] = True
    except:
        security_info['dns_resolution'] = False
        security_info['resolves_to_ip'] = None
    
    # Check SSL certificate
    security_info['has_valid_ssl'] = False
    if url.startswith('https://'):
        try:
            # Create an SSL context
            context = ssl.create_default_context()
            with socket.create_connection((full_domain, 443)) as sock:
                with context.wrap_socket(sock, server_hostname=full_domain) as ssock:
                    cert = ssock.getpeercert()
                    # Check certificate expiration
                    cert_expires = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    security_info['has_valid_ssl'] = True
                    security_info['ssl_expires'] = cert_expires.strftime('%Y-%m-%d')
        except:
            security_info['has_valid_ssl'] = False
    
    # Get domain age
    domain_age = get_domain_age(full_domain)
    security_info['domain_age_days'] = domain_age
    
    # Try to fetch the website (safely)
    security_info['site_accessible'] = False
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        response = requests.get(url, headers=headers, timeout=5, allow_redirects=True, verify=True)
        security_info['site_accessible'] = response.status_code == 200
        security_info['final_url'] = response.url  # Check if redirected
        security_info['redirected'] = response.url != url
        
        # Basic content analysis
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check for login forms (could indicate phishing)
            forms = soup.find_all('form')
            security_info['has_forms'] = len(forms) > 0
            
            # Check for password fields
            password_fields = soup.find_all('input', {'type': 'password'})
            security_info['has_password_field'] = len(password_fields) > 0
            
            # Calculate page text hash (useful for comparing with known phishing pages)
            text_content = soup.get_text()
            security_info['content_hash'] = hashlib.md5(text_content.encode()).hexdigest()
            
    except:
        security_info['site_accessible'] = False
    
    return security_info

def show_url_phishing():
    st.markdown("<h1>URL Phishing Detection</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="fraud-card">
        <h3>URL Phishing Scanner</h3>
        <p>Enter a URL to scan it for potential phishing indicators. Our ML model will analyze various features to determine if the URL is likely legitimate or fraudulent.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load or create the phishing detection model
    model_package = load_model()
    model = model_package['model']
    features = model_package['features']
    
    # URL input form
    with st.form("url_form"):
        url = st.text_input("Enter URL to scan", "https://example.com")
        advanced_scan = st.checkbox("Perform advanced security checks")
        submitted = st.form_submit_button("Scan URL")
    
    if submitted:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Add a loading effect
        with st.spinner("Scanning URL for phishing indicators..."):
            time.sleep(1.5)  # Simulate processing time
            
            # Extract features from the URL
            url_features = extract_features(url)
            
            # Prepare features for prediction
            feature_vector = [url_features.get(feature, 0) for feature in features]
            
            # Get prediction (0: legitimate, 1: phishing)
            prediction_proba = model.predict_proba([feature_vector])[0]
            phishing_probability = prediction_proba[1] * 100
            legitimate_confidence = 100 - phishing_probability
            
            # MODIFIED: Mark as phishing if confidence is below 90%
            is_phishing = legitimate_confidence < 90
            
            # Get additional security information if requested
            security_info = None
            if advanced_scan:
                security_info = check_url_security(url)
                # Update domain age feature with actual value if available
                if security_info['domain_age_days'] > 0:
                    url_features['domain_age_days'] = security_info['domain_age_days']
                    # Re-predict with updated features
                    feature_vector = [url_features.get(feature, 0) for feature in features]
                    prediction_proba = model.predict_proba([feature_vector])[0]
                    phishing_probability = prediction_proba[1] * 100
                    legitimate_confidence = 100 - phishing_probability
                    # MODIFIED: Update classification based on new confidence
                    is_phishing = legitimate_confidence < 90
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if is_phishing:
                st.error(f"⚠️ Warning: Potential Phishing URL Detected! (Confidence below 90%)")
                st.markdown("""
                <div style="background-color: #FEE2E2; padding: 15px; border-radius: 8px; border-left: 5px solid #DC2626;">
                    <h4 style="color: #DC2626; margin-top: 0;">Risk Factors Identified:</h4>
                    <p>This URL has characteristics commonly associated with phishing websites.</p>
                    <p><strong>Recommended Action:</strong> Avoid visiting this website and do not enter any personal information.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success(f"✅ URL Appears Legitimate ({legitimate_confidence:.1f}% confidence)")
                st.markdown("""
                <div style="background-color: #ECFDF5; padding: 15px; border-radius: 8px; border-left: 5px solid #10B981;">
                    <h4 style="color: #10B981; margin-top: 0;">Assessment:</h4>
                    <p>This URL does not show common phishing characteristics.</p>
                    <p><strong>Note:</strong> Always exercise caution when entering sensitive information online.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Create gauge chart for risk visualization
            fig = px.pie(values=[phishing_probability, legitimate_confidence], 
                         names=['Risk', 'Safe'],
                         hole=0.7, 
                         color_discrete_sequence=[
                             '#DC2626' if is_phishing else '#10B981', 
                             '#E5E7EB'
                         ])
            fig.update_layout(
                showlegend=False,
                annotations=[dict(text=f"{legitimate_confidence:.1f}%", x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            st.plotly_chart(fig)
        # Show extracted features
        st.markdown("<h3>URL Analysis</h3>", unsafe_allow_html=True)
        
        # Identify most important features for this prediction
        feature_importance = model.feature_importances_
        feature_values = {feature: url_features.get(feature, 0) for feature in features}
        feature_contributions = {feature: importance * feature_values[feature] for feature, importance in zip(features, feature_importance)}
        
        # Sort features by contribution to prediction
        sorted_features = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:5]
        
        # Create columns for feature display
        feature_cols = st.columns(5)
        
        for i, (feature, contribution) in enumerate(top_features):
            # Determine if feature contributes to phishing or legitimate prediction
            is_risk_factor = contribution > 0
            feature_display_name = feature.replace('_', ' ').title()
            feature_value = url_features.get(feature, 0)
            
            with feature_cols[i]:
                if is_risk_factor:
                    st.markdown(f"""
                    <div style="text-align:center; padding:10px; background-color:#FEF2F2; border-radius:5px;">
                        <h4 style="margin:0; color:#DC2626;">{feature_display_name}</h4>
                        <p style="font-size:1.2rem; font-weight:bold;">{feature_value}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="text-align:center; padding:10px; background-color:#ECFDF5; border-radius:5px;">
                        <h4 style="margin:0; color:#10B981;">{feature_display_name}</h4>
                        <p style="font-size:1.2rem; font-weight:bold;">{feature_value}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Display security information if advanced scan was performed
        if security_info:
            st.markdown("<h3>Advanced Security Analysis</h3>", unsafe_allow_html=True)
            
            sec_col1, sec_col2 = st.columns(2)
            
            with sec_col1:
                # Display DNS and IP information
                st.markdown("<h4>Domain Information</h4>", unsafe_allow_html=True)
                
                domain_info = {
                    "DNS Resolution": "✅ Success" if security_info['dns_resolution'] else "❌ Failed",
                    "IP Address": security_info['resolves_to_ip'] if security_info['resolves_to_ip'] else "N/A",
                    "Domain Age": f"{security_info['domain_age_days']} days" if security_info['domain_age_days'] > 0 else "Unknown"
                }
                
                for key, value in domain_info.items():
                    st.markdown(f"**{key}:** {value}")
            
            with sec_col2:
                # Display SSL and site information
                st.markdown("<h4>Website Security</h4>", unsafe_allow_html=True)
                
                security_details = {
                    "SSL Certificate": "✅ Valid" if security_info['has_valid_ssl'] else "❌ Invalid or Missing",
                    "SSL Expiry": security_info.get('ssl_expires', "N/A"),
                    "Site Accessible": "✅ Yes" if security_info['site_accessible'] else "❌ No",
                    "URL Redirection": "⚠️ Yes" if security_info.get('redirected', False) else "✅ No"
                }
                
                for key, value in security_details.items():
                    st.markdown(f"**{key}:** {value}")
            
            # Content analysis if site is accessible
            if security_info.get('site_accessible'):
                st.markdown("<h4>Content Analysis</h4>", unsafe_allow_html=True)
                
                content_col1, content_col2 = st.columns(2)
                
                with content_col1:
                    has_forms = security_info.get('has_forms', False)
                    has_password = security_info.get('has_password_field', False)
                    
                    # Determine risk level for content
                    content_risk = "Low"
                    if has_forms and has_password:
                        content_risk = "Medium to High"
                    elif has_forms or has_password:
                        content_risk = "Medium"
                    
                    content_details = {
                        "Login Forms": "⚠️ Present" if has_forms else "✅ None detected",
                        "Password Fields": "⚠️ Present" if has_password else "✅ None detected",
                        "Content Risk Level": content_risk
                    }
                    
                    for key, value in content_details.items():
                        st.markdown(f"**{key}:** {value}")
                
                with content_col2:
                    # Display final URL after any redirects
                    final_url = security_info.get('final_url', url)
                    st.markdown(f"**Final URL:** {final_url}")
                    
                    # Display content hash (useful for detecting known phishing templates)
                    content_hash = security_info.get('content_hash', "N/A")
                    st.markdown(f"**Content Hash:** {content_hash[:10]}...")
        
        # Educational information about phishing
        st.markdown("<h3>What is URL Phishing?</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="fraud-card">
            <p>Phishing is a cyber attack that uses disguised email addresses, websites, and text messages to steal personal information. 
            Attackers masquerade as trusted entities to trick users into providing sensitive data.</p>
            
            <h4>Common Phishing Indicators:</h4>
            <ul>
                <li>URLs with IP addresses instead of domain names</li>
                <li>Misspelled domain names (e.g., "g00gle" instead of "google")</li>
                <li>Excessive use of subdomains or unusual characters</li>
                <li>Recently registered domains</li>
                <li>Missing or invalid SSL certificates</li>
                <li>Requests for personal information</li>
            </ul>
            
            <h4>How to Stay Safe:</h4>
            <ul>
                <li>Check URLs carefully before clicking</li>
                <li>Look for HTTPS and valid certificates</li>
                <li>Be cautious with shortened URLs</li>
                <li>Don't enter personal information unless you're certain the site is legitimate</li>
                <li>Use browsers with built-in phishing protection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()