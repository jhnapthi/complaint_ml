import os
import sqlite3
import pickle
import datetime
from flask import Flask, request, jsonify, render_template
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize Flask App
app = Flask(__name__)

# --- Setup & Configuration ---
DB_NAME = "complaints.db"
MODEL_FILE = "complaint_classifier.pkl"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Ensure this is set in environment

# Download VADER lexicon if not present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Load ML Model
print("Loading ML model...")
try:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file not found. Please run train_model.py first.")
    model = None

# --- Database Helper Functions ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS complaints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    complaint_text TEXT NOT NULL,
                    category TEXT,
                    sentiment_score REAL,
                    priority TEXT,
                    status TEXT DEFAULT 'Pending',
                    admin_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# --- Logic Functions ---
def determine_priority(text, category, sentiment):
    # Logic: 
    # Negative sentiment + Critical Category -> High (Red)
    # Urgent Keywords -> High (Red)
    # Appreciation -> Low
    
    urgent_keywords = ['urgent', 'emergency', 'critical', 'immediately', 'medicine', 'passport', 'deadline']
    is_urgent = any(k in text.lower() for k in urgent_keywords)
    
    is_critical_category = category in ['Lost Parcel', 'Damaged Item', 'Staff Behavior', 'Refund / Compensation']
    
    if category == 'Appreciation':
        return "Low"
    elif is_urgent or (sentiment < -0.4 and is_critical_category):
        return "High"
    elif sentiment < -0.2:
        return "Medium"
    else:
        return "Low"

def get_template_response(category):
    templates = {
        'Delivery Delay': "We apologize for the delay in your delivery. We are tracking your consignment and will update you shortly.",
        'Lost Parcel': "We deeply regret to hear that your parcel is missing. We have initiated a search at our sorting hubs.",
        'Damaged Item': "We are sorry to hear about the damage. Please provide photographs of the package for insurance claim processing.",
        'Wrong Delivery': "We apologize for the misdelivery. We are coordinating with the delivery agent to retrieve and deliver your package correctly.",
        'Tracking Issue': "We are facing intermittent issues with the tracking server. Please try again in 2 hours.",
        'Refund / Compensation': "Your refund request has been noted. It will be processed within 7 working days.",
        'Staff Behavior': "We take such complaints seriously. An inquiry has been initiated against the concerned staff member."
    }
    return templates.get(category, "We have received your complaint and will look into it.")

# --- Routes ---

@app.route('/')
def home():
    return render_template('customer.html')

@app.route('/admin')
def admin_dashboard():
    return render_template('admin.html')

@app.route('/submit_complaint', methods=['POST'])
def submit_complaint():
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # 1. Predict Category
    category = "Uncategorized"
    if model:
        category = model.predict([text])[0]
    
    # 2. Analyze Sentiment
    sentiment = sia.polarity_scores(text)["compound"]
    
    # 3. Determine Priority
    priority = determine_priority(text, category, sentiment)
    
    # 4. Save to DB
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO complaints (complaint_text, category, sentiment_score, priority) VALUES (?, ?, ?, ?)",
              (text, category, sentiment, priority))
    conn.commit()
    complaint_id = c.lastrowid
    conn.close()
    
    return jsonify({
        "message": "Complaint submitted successfully",
        "id": complaint_id,
        "category": category,
        "priority": priority
    })

@app.route('/admin/complaints', methods=['GET'])
def get_complaints():
    conn = get_db_connection()
    complaints = conn.execute('SELECT * FROM complaints ORDER BY created_at DESC').fetchall()
    conn.close()
    
    # Convert to list of dicts
    result = []
    for row in complaints:
        result.append({
            "id": row['id'],
            "text": row['complaint_text'],
            "category": row['category'],
            "sentiment": row['sentiment_score'],
            "priority": row['priority'],
            "status": row['status'],
            "response": row['admin_response'],
            "created_at": row['created_at']
        })
    return jsonify(result)

@app.route('/admin/generate_response', methods=['POST'])
def generate_response():
    data = request.json
    complaint_id = data.get('id')
    
    conn = get_db_connection()
    complaint = conn.execute('SELECT * FROM complaints WHERE id = ?', (complaint_id,)).fetchone()
    conn.close()
    
    if not complaint:
        return jsonify({"error": "Complaint not found"}), 404
        
    category = complaint['category']
    text = complaint['complaint_text']
    
    # 1. Get Base Template
    base_template = get_template_response(category)
    
    # 2. Gemini Enhancement (Mocked if no key, or actual call logic)
    # In a real scenario, use google-generativeai library
    # import google.generativeai as genai
    # genai.configure(api_key=GEMINI_API_KEY)
    # model = genai.GenerativeModel('gemini-pro')
    # response = model.generate_content(f"Rewrite this politely for a customer complaining about '{text}': {base_template}")
    # refined_response = response.text
    
    # For Hackathon Demo (if key missing): simulate enhancement
    refined_response = f"[Draft] Dear Customer, regarding your concern: '{base_template}' We assure you we are working on it. (Enhanced by AI)"
    
    return jsonify({
        "original_template": base_template,
        "suggested_response": refined_response
    })

@app.route('/admin/resolve', methods=['POST'])
def resolve_complaint():
    data = request.json
    complaint_id = data.get('id')
    final_response = data.get('response')
    
    conn = get_db_connection()
    conn.execute('UPDATE complaints SET status = ?, admin_response = ? WHERE id = ?',
                 ('Resolved', final_response, complaint_id))
    conn.commit()
    conn.close()
    
    return jsonify({"message": "Complaint resolved"})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
