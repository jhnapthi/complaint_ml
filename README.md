# India Post AI Complaint Management System

This system automates the classification, prioritizing, and response generation for India Post complaints.

## Components
1. **ML Model**: `complaint_classifier.pkl` (Trained using `train_model.py` on `complaints_data.csv`).
2. **Backend**: `app.py` (Flask). Uses VADER for sentiment analysis and the ML model for classification.
3. **Frontend**:
   - Customer Portal: `http://localhost:5000/`
   - Admin Dashboard: `http://localhost:5000/admin`

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model** (Already done, but if you change data):
   ```bash
   python train_model.py
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Verify**:
   - Open [http://localhost:5000/](http://localhost:5000/) to submit a complaint.
   - Open [http://localhost:5000/admin](http://localhost:5000/admin) to view it, see the priority (Red/Yellow/Green), and generate a response.

## Gemini API
To enable real Gemini API calls, set the environment variable:
```bash
set GEMINI_API_KEY=your_api_key_here
```
(Currently mocked for demo purposes within `app.py`).
