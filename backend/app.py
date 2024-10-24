# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import asyncio
# from emotion_recommender import AdvancedEmotionRecommender  # Your previous emotion recommender class

# app = Flask(__name__)
# CORS(app)  # Enable CORS for Angular frontend

# # Initialize the recommender
# recommender = AdvancedEmotionRecommender()

# @app.route('/api/analyze', methods=['POST'])
# async def analyze_text():
#     try:
#         data = request.json
#         text = data.get('text', '')
        
#         # Get emotion analysis and recommendations
#         emotion_scores = await recommender.get_ensemble_emotion_scores(text)
#         recommendations = await recommender.get_recommendations(text)
        
#         # Convert recommendations to serializable format
#         recommendations_list = recommendations.to_dict('records')
        
#         return jsonify({
#             'emotions': emotion_scores,
#             'recommendations': recommendations_list
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)


from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load pre-trained models
emotion_model = pipeline('sentiment-analysis', model='bhadresh-savani/distilbert-base-uncased-emotion')
recommendation_model = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

@app.route('/analyze', methods=['POST'])
def analyze_emotion():
    data = request.json
    text = data['text']
    emotion_result = emotion_model(text)
    emotion = emotion_result[0]['label']
    return jsonify({'emotion': emotion})

@app.route('/recommend', methods=['POST'])
def recommend_content():
    data = request.json
    emotion = data['emotion']
    candidate_labels = ['happy', 'sad', 'angry', 'surprised', 'neutral']
    recommendation_result = recommendation_model(emotion, candidate_labels)
    recommended_content = recommendation_result['labels'][0]
    return jsonify({'recommendation': recommended_content})

if __name__ == '__main__':
    app.run(debug=True)