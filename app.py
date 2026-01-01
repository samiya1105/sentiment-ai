from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import spacy
import os
import re

app = Flask(__name__)
# Enable CORS so frontend can call backend
CORS(app)

@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

# Load SpaCy small English model
# Run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- CUSTOM SENTIMENT DICTIONARY (SpaCy-only compatibility) ---
# Since we are strictly not using NLTK/TextBlob, we use a lemma-based lookup.
POSITIVE_WORDS = {
    'good', 'great', 'happy', 'excellent', 'amazing', 'love', 'best', 'fantastic',
    'wonderful', 'perfect', 'beautiful', 'like', 'joy', 'excited', 'positive'
}
NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'sad', 'hate', 'worst', 'angry', 'poor',
    'disappointed', 'horrible', 'ugly', 'fail', 'negative', 'wrong', 'broken'
}

def get_rule_based_sentiment(lemmas):
    """
    Simple scoring based on lemma matching.
    Returns: (polarity_label, score, explanation)
    """
    score = 0
    matches = []
    
    for lemma in lemmas:
        if lemma.lower() in POSITIVE_WORDS:
            score += 1
            matches.append(f"+{lemma}")
        elif lemma.lower() in NEGATIVE_WORDS:
            score -= 1
            matches.append(f"-{lemma}")
            
    # Normalize score
    if score > 0:
        return "Positive", min(50 + (score * 10), 99), f"Found positive words: {', '.join(matches)}"
    elif score < 0:
        return "Negative", min(50 + (abs(score) * 10), 99), f"Found negative words: {', '.join(matches)}"
    else:
        return "Neutral", 85, "No strong sentiment words found."

def detect_intent(text, lemmas):
    """
    Detects user intent using keyword matching on lemmas and raw text.
    """
    text_lower = text.lower()
    lemma_set = set(l.lower() for l in lemmas)
    
    # Priority-based intent mapping
    intent_map = {
        # 1. Specific Business & Transactional
        "Payment Issue": ["payment failed", "card declined"],
        "Order Status": ["order status", "tracking", "shipped", "order"],
        "Delivery / Shipping": ["delivery date", "delay", "shipping cost"],
        "Return / Refund / Exchange": ["return", "refund", "exchange", "money", "replace"],
        "Cancellation": ["cancel order", "stop service", "cancel"],
        "Billing / Invoice": ["bill", "invoice", "charge", "fee"],
        "Availability Check": ["available", "in stock", "left"],
        "Purchase Inquiry": ["buy", "purchase", "price", "cost"],
        
        # 2. Support & Technical
        "Account Issue": ["login", "password", "account locked"],
        "Technical Issue": ["bug", "error", "not working", "crash"],
        "Escalation / Human Request": ["talk to agent", "customer care", "agent"],
        "Support Request": ["help", "support", "assist", "issue", "problem"],
        
        # 3. Specific Information Requests
        "How-to / Instruction Request": ["how to use", "steps", "guide", "tutorial"],
        "Recommendation / Suggestion Request": ["suggest", "recommend", "best option"],
        "Clarification Request": ["can you explain", "unclear", "didn't understand"],
        "Bot Capability Question": ["what can you do"],
        "Bot Feedback": ["slow", "smart bot", "good bot", "bad bot"],
        
        # 4. Security & Privacy
        "Privacy Concern": ["privacy", "data", "information safety"],
        "Security Issue": ["hacked", "fraud", "suspicious"],
        
        # 5. User Intent Control
        "Command / Action Request": ["do this", "generate", "analyze", "start"],
        "Comparison": ["compare", "vs", "difference between"],
        "Search Intent": ["find", "look for", "search"],
        
        # 6. Feedback & Opinions
        "Complaint": ["bad service", "unhappy", "disappointed"],
        "Suggestion / Feature Request": ["improve", "add feature", "suggestion", "add"],
        "Review / Rating": ["rate", "stars", "review"],
        "Positive Feedback": ["good", "nice", "loved it", "awesome"],
        "Negative Feedback": ["bad", "poor", "terrible"],
        "Error / Misunderstanding": ["that's wrong", "that is wrong", "incorrect response"],
        
        # 7. Core Social & Casual
        "Small Talk": ["how are you", "what's up", "how's it going"],
        "Joke / Fun": ["joke", "funny", "lol"],
        "Emotional Expression": ["happy", "sad", "angry", "frustrated"],
        "Greeting": ["hello", "hi", "hey", "greetings", "morning", "good morning"],
        "Farewell": ["bye", "goodbye", "see you", "take care"],
        "Thanks / Appreciation": ["thanks", "thank you", "appreciate", "grateful"],
        "Apology": ["sorry", "my bad", "apologize"],
        "Confirmation / Acknowledgement": ["okay", "ok", "got it", "understood", "yes", "sure"],
        "Negation / Disagreement": ["no", "not really", "don't agree", "never"],
        
        # 8. General Question (lowest priority specific intent)
        "General Question": ["what", "how", "when", "where", "why", "?"]
    }

    # Multi-pass matching for better precision
    for intent, keywords in intent_map.items():
        for kw in keywords:
            if kw == "?":
                if "?" in text: return intent
                continue
            
            # Escape keyword for regex and wrap in word boundaries
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, text_lower):
                return intent
            # Also check lemmas for more flexibility
            if kw in lemma_set:
                return intent
            
    return "General Statement"

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    
    # 1. SpaCy Processing Pipeline
    doc = nlp(text)
    
    # 2. Extract Lemmas (filtering stops/punct)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    # 3. Named Entity Recognition
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # 4. Sentiment Analysis (Rule-based)
    sentiment, confidence, explanation = get_rule_based_sentiment(lemmas)
    
    # 5. Intent Detection
    intent = detect_intent(text, lemmas)

    # 6. Construct Response
    response = {
        "sentiment": sentiment,
        "confidence_score": confidence,
        "explanation": explanation,
        "detected_intent": intent,
        "entities": entities,
        "processed_tokens": lemmas  # Debug info
    }
    
    return jsonify(response)

if __name__ == '__main__':
    # Bind to 0.0.0.0 and use PORT from environment for production deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
