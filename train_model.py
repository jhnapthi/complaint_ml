import pandas as pd
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

def train():
    print("Loading data...")
    try:
        df = pd.read_csv('complaints_data.csv')
    except FileNotFoundError:
        print("Error: complaints_data.csv not found.")
        return

    # Basic cleaning
    df = df.dropna(subset=['complaint_text', 'category'])
    
    X = df['complaint_text']
    y = df['category']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Setting up pipeline and grid search...")
    # Pipeline: TF-IDF -> LinearSVC
    # LinearSVC is generally better for text classification than Logistic Regression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3))),  # Removed stop_words='english' to keep negations
        ('clf', CalibratedClassifierCV(LinearSVC(random_state=42, dual='auto')))
    ])
    
    # Hyperparameters to tune
    parameters = {
        'tfidf__max_df': [0.75, 1.0],
        'tfidf__min_df': [1, 2],
        'clf__estimator__C': [0.1, 1, 10],
    }
    
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
    
    print("Training with GridSearch (finding best parameters)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.2%}")
    
    # Evaluate on Test Set
    print("\nEvaluating on Test Set...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
    # Retrain on ALL data with best parameters for the final product
    print("Retraining on full dataset with best parameters...")
    best_model.fit(X, y)
    
    # Save model
    with open('complaint_classifier.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    print("Model saved to 'complaint_classifier.pkl'")

if __name__ == "__main__":
    train()
