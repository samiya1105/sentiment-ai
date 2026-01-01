from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import spacy
import os

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
    
    if any(w in text_lower for w in ['hello', 'hi', 'hey', 'greetings', 'morning']):
        return "Greeting"
    if any(w in lemma_set for w in ['help', 'support', 'assist', 'issue', 'problem', 'error']):
        return "Support Request"
    if any(w in lemma_set for w in ['return', 'refund', 'exchange', 'money']):
        return "Return/Refund"
    if '?' in text or any(w in lemma_set for w in ['what', 'how', 'when', 'where', 'why']):
        return "Question"
    if any(w in lemma_set for w in ['buy', 'purchase', 'order', 'price', 'cost']):
        return "Purchase Inquiry"
        
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
