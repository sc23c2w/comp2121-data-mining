from llm_client import LLMClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class MisinformationDetector:
    def __init__(self):
        # Initialize the LLM client
        self.llm_client = LLMClient()
        
        # Check if the LLM API is running
        if not self.llm_client.health_check():
            raise RuntimeError("LLM API is not running. Execute ./start_llm_api.sh first.")
        
        # Initialize scikit-learn classifiers instead of Weka
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.is_trained = False
    
    def analyze_article(self, title, content):
        """Analyze an article for potential misinformation"""
        
        # Get analysis from LLM
        llm_analysis = self.llm_client.analyze_article(title, content)
        
        # For now, we'll just use the LLM analysis
        # When the classifier is trained, you could combine both approaches
        
        return {
            "llm_analysis": llm_analysis,
            "recommendation": "Potentially misleading" if llm_analysis.get("potentially_misleading") else "Likely reliable",
            "confidence": llm_analysis.get("combined_score", 0.5)
        }
    
    def train_classifier(self, data_df):
        """Train the classifier using scikit-learn
        
        Args:
            data_df: pandas DataFrame with 'text' and 'label' columns
        """
        try:
            X = self.vectorizer.fit_transform(data_df['text'])
            y = data_df['label']
            
            self.classifier.fit(X, y)
            self.is_trained = True
            
            print(f"Classifier trained on {len(data_df)} examples")
            return True
        except Exception as e:
            print(f"Error training classifier: {e}")
            return False
    
    def predict_with_ml(self, text):
        """Make a prediction using the trained ML classifier"""
        if not self.is_trained:
            return {"error": "Classifier not yet trained"}
        
        try:
            X = self.vectorizer.transform([text])
            prediction = self.classifier.predict(X)[0]
            probabilities = self.classifier.predict_proba(X)[0]
            
            return {
                "prediction": prediction,
                "confidence": float(max(probabilities)),
                "reliable_prob": float(probabilities[list(self.classifier.classes_).index("reliable")]) 
                                if "reliable" in self.classifier.classes_ else None
            }
        except Exception as e:
            return {"error": f"Prediction error: {e}"}

# Example usage
if __name__ == "__main__":
    # Create the detector
    detector = MisinformationDetector()
    
    # Analyze an example article
    result = detector.analyze_article(
        title="New Study Shows Amazing Results",
        content="A groundbreaking new study by researchers has shown incredible results that challenge all previous understanding. Experts are shocked by the findings which cannot be explained by current theories."
    )
    
    print("\nMisinformation Analysis:")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("\nDetailed LLM Analysis:")
    for key, value in result['llm_analysis'].items():
        print(f"{key}: {value}")
    
    # Example of how you might train the classifier with sample data
    print("\nTraining classifier with sample data...")
    sample_data = pd.DataFrame({
        'text': [
            "Scientists discover revolutionary new treatment",
            "BREAKING: Shocking conspiracy revealed by anonymous source",
            "Research study shows moderate benefits of new approach",
            "You won't BELIEVE what this celebrity did next!"
        ],
        'label': ['reliable', 'unreliable', 'reliable', 'unreliable']
    })
    
    detector.train_classifier(sample_data)
    
    # Test the ML classifier
    test_text = "New research published in a peer-reviewed journal suggests promising results"
    ml_result = detector.predict_with_ml(test_text)
    print("\nML Classification Result:")
    for key, value in ml_result.items():
        print(f"{key}: {value}")