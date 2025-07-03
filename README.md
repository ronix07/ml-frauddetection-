# 🛡️ Fraud Guardian – Payment & URL Fraud Detection System

Fraud Guardian is a full-fledged interactive web application built with **Streamlit**, designed to detect and prevent payment fraud and phishing attacks using **machine learning** and **real-time URL analysis**. It includes dashboards, transaction risk evaluation, phishing URL scanning, and more.

---

## 🚀 Features

- 📊 **Dashboard Overview** with fraud metrics and visualizations
- 💸 **Real-Time Payment Fraud Detection** using ML (Random Forest)
- 🌐 **URL Phishing Detection** with deep URL feature extraction
- 🔬 **Analytics Tab** to observe fraud trends and risk distributions
- 📈 Interactive charts built with **Plotly**
- 🎨 Custom CSS for a clean and modern UI
- 🧠 Offline fallback with mock model in case of missing `.pkl`

---

## 🧠 Tech Stack

| Component            | Technology                      |
|---------------------|----------------------------------|
| Frontend UI         | Streamlit                        |
| Charts & Visuals    | Plotly, Matplotlib               |
| Machine Learning    | Random Forest (Scikit-learn)     |
| Phishing Detection  | Custom feature extractor         |
| Web Scraping        | Requests + BeautifulSoup         |
| Domain Info         | tldextract, socket, whois        |
| Styling             | Custom HTML/CSS in Streamlit     |

---1. Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/fraud-guardian.git
cd fraud-guardian
2. Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Streamlit app:
bash
Copy
Edit
streamlit run aae0cf5c-9c72-4f8d-b7dc-f61c23cebb2b.py
🧪 Models
fraud_detection_rf_model.pkl is used to classify payment transactions

phishing_detection_model.pkl is used for URL classification

If .pkl files are missing, mock models will be used with rule-based logic

🌐 URL Phishing Detection
Uses feature engineering on URLs like:

Length, subdomains, HTTPS usage

Special characters, IP address, redirection

Suspicious keywords (e.g., login, secure, paypal)

Also performs:

DNS resolution

SSL certificate check

WHOIS-based domain age

Password/login form detection via scraping

📊 Dashboard Insights
The Dashboard provides:

Total transactions

Fraud trends over time

Detection rates

Recently flagged activity

Analytics includes:

Hourly fraud patterns

Risk score distributions

Transaction type heatmaps

🔐 Security Notes
SSL validation and DNS checks are performed for URL safety

JSON models and external file usage are sandboxed within app logic

👨‍💻 Author
Ronit Rathod

## 📦 Folder Structure

```bash
.
├── aae0cf5c-...fraud.py           # Main Streamlit app (Fraud Guardian)
├── fraud_detection_rf_model.pkl   # Trained model for payment fraud (optional)
├── phishing_detection_model.pkl   # Trained model for phishing URLs (optional)
├── requirements.txt               # Python dependencies
└── README.md


