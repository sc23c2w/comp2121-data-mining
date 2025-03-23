import requests
import json

class LLMClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def classify_text(self, text):
        response = requests.post(
            f"{self.base_url}/classify", 
            json={"text": text}
        )
        return response.json()
    
    def analyze_article(self, title, content):
        response = requests.post(
            f"{self.base_url}/analyze_article", 
            json={"title": title, "content": content}
        )
        return response.json()
    
    def health_check(self):
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False

# Example usage
if __name__ == "__main__":
    client = LLMClient()
    
    # Check if the API is running
    if client.health_check():
        # Test the API
        result = client.classify_text("This article contains misleading information.")
        print(json.dumps(result, indent=2))
        
        # Analyze an article
        article_result = client.analyze_article(
            title="Shocking new discovery proves everything wrong",
            content="Scientists were baffled yesterday when all previous theories were disproven by a groundbreaking discovery."
        )
        print(json.dumps(article_result, indent=2))
    else:
        print("API is not running. Execute ./start_llm_api.sh first.")