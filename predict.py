from flask import Flask, request, jsonify
from dotenv import load_dotenv
import boto3
import joblib
import io
import json
import os

load_dotenv()

app = Flask(__name__)

#configuration

AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")
BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

MODEL_KEY = "best_model.pkl"

#connexion à localstack s3

s3 = boto3.client(
    "s3",
    endpoint_url=AWS_ENDPOINT_URL,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=AWS_REGION
)

def load_model_from_s3():
    response = s3.get_object(
        Bucket=BUCKET_NAME,
        Key=MODEL_KEY
    )
    
    model_data = response["Body"].read()
    model = joblib.load(io.BytesIO(model_data))
    return model
def load_metrics_from_s3():
    response = s3.get_object(
        Bucket=BUCKET_NAME,
        Key="metrics.json"
    )

    metrics_data = response["Body"].read()
    metrics = json.loads(metrics_data)

    return metrics

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API de classification de sentiment opérationnel"
    })
    
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    }), 200

@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        metrics = load_metrics_from_s3()

        return jsonify(metrics), 200

    except Exception as e:
        return jsonify({
            "error": f"Erreur lors du chargement des métriques : {str(e)}"
        }), 500

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return jsonify({
            "message": "Endpoint de prédiction opérationnel",
            "method": "POST",
            "usage": "Envoyer un JSON avec le champ 'text'"
        }), 200

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({
            "error": "Le champ 'text' est obligatoire"
        }), 400

    text = data["text"]

    try:
        model = load_model_from_s3()
        prediction = model.predict([text])[0]

        return jsonify({
            "text": text,
            "sentiment": prediction
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"Erreur lors de la prediction : {str(e)}"
        }), 500

    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)