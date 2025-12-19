import pickle
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the model
try:
    with open('complaint_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file 'complaint_classifier.pkl' not found. Run train_model.py first.")
    exit()

# Setup Sentiment Analysis
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

def predict_complaint(text):
    print("\n" + "="*50)
    print(f"Analyzing: \"{text}\"")
    print("="*50)
    
    # 1. Predict Category
    category = model.predict([text])[0]
    
    # 2. Get Confidence Score
    probabilities = model.predict_proba([text])[0]
    confidence = max(probabilities)
    
    # 3. Sentiment Analysis
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    
    # 4. Determine Logic (Mimicking backend logic)
    urgent_keywords = ['urgent', 'emergency', 'critical', 'immediately', 'medicine', 'passport', 'deadline']
    is_urgent = any(k in text.lower() for k in urgent_keywords)

    priority = "Low"
    if category == 'Appreciation':
        priority = "Low (Green)"
    elif is_urgent or (compound_score < -0.4 and category in ['Lost Parcel', 'Damaged Item', 'Staff Behavior']):
        priority = "High (Red)"
    elif compound_score < -0.2:
        priority = "Medium (Yellow)"
    else:
        priority = "Low (Green)"

    # Output
    print(f"📂 Predicted Category:  {category}")
    print(f"🎯 Confidence Score:    {confidence:.2%}")
    print(f"❤️ Sentiment Score:     {compound_score} (Range: -1.0 to 1.0)")
    print(f"🚨 Suggested Priority:  {priority}")
    print("-"*50)

if __name__ == "__main__":
    print("Interactive Complaint Classifier")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nEnter a complaint: ")
        if user_input.lower() == 'exit':
            break
        predict_complaint(user_input)
