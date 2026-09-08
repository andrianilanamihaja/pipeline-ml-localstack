import os
import joblib
import boto3
import json
import pandas as pd
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score
)

load_dotenv()
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


def create_bucket_if_not_exists():
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)

        print(
            f"Le bucket '{BUCKET_NAME}' existe déjà."
        )

    except Exception:

        print(
            f"Le bucket '{BUCKET_NAME}' n'existe pas. Création..."
        )

        s3.create_bucket(
            Bucket=BUCKET_NAME
        )

        print(
            f"Bucket '{BUCKET_NAME}' créé avec succès."
        )


create_bucket_if_not_exists()

from datasets import load_dataset
ds = load_dataset("gaouehalim/French-reviews-with-prediction-of-their-feelings")
df = ds["train"].to_pandas()
print(df.head(6))


# suppression du colonne id

df = df.drop(columns = ["id"])


x = df["clean_content"]
y = df["sentiment"]




x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size = 0.2, 
    random_state = 42,
    stratify = y
    )


#MODEL LOGISTIC REGRESSION


pipeline_logistic = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features = 10000,
            ngram_range = (1,2)
        )
    ),
    (
        "model",
        LogisticRegression(
            max_iter = 1000
        )
    )
])
pipeline_logistic.fit(x_train, y_train)

y_pred = pipeline_logistic.predict(x_test)

accuracy_logistic = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy_logistic:.4f}")

print("Matrice de confusion: ")
print(confusion_matrix(y_test, y_pred))

print("\nRapport de classification: ")
print(classification_report(y_test, y_pred))

y_probs = pipeline_logistic.predict_proba(x_test)
roc_auc_logistic = roc_auc_score(
    y_test, 
    y_probs,
    multi_class= "ovr",
    average="weighted"
)

print(f"\nScore Roc auc: {roc_auc_logistic:.4f}")

# MODEL MULTINOMIALNB

pipeline_multinomial = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features = 10000,
                ngram_range = (1,2)
            )
        ),
        (
            "model",
            MultinomialNB()
        )
    ]
)

pipeline_multinomial.fit(x_train, y_train)
y_pred_nb = pipeline_multinomial.predict(x_test)


accuracy_multinomial = accuracy_score(y_test, y_pred_nb)

print(f"Accuracy : {accuracy_multinomial:.4f}")

print("Matrice de confusion: ")
print(confusion_matrix(
    y_test, 
    y_pred_nb,
    labels = ["negative", "neutral", "positive"]
))

print("\nRapport de classification: ")
print(classification_report(
    y_test, 
    y_pred_nb,
    
))

y_probs_nb = pipeline_multinomial.predict_proba(x_test)
roc_auc_multinomial = roc_auc_score(
    y_test, 
    y_probs_nb,
    multi_class= "ovr",
    average="weighted"
)

print(f"\nScore Roc auc: {roc_auc_multinomial:.4f}")

#comparaison des models

resultats = pd.DataFrame({
    "Modeles" : [
        "LogisticRegression",
        "MultinomialNB"
    ],
    "Accuracy" : [
        accuracy_logistic,
        accuracy_multinomial
    ],
    "Roc-auc" : [
        roc_auc_logistic,
        roc_auc_multinomial
    ]
})

print(resultats)


#choix du model

if accuracy_logistic >= accuracy_multinomial :
    best_model = pipeline_logistic
    best_model_name = "logistic"
    best_accuracy = accuracy_logistic
else:
    best_model = pipeline_multinomial
    best_model_name = "multinomial"
    best_accuracy = accuracy_multinomial

print("modele: ", best_model_name)
print(f"Accuracy:  {best_accuracy:.4f}")


joblib.dump(
    best_model,
    "best_model.pkl"
)

s3.upload_file(
    "best_model.pkl",
    BUCKET_NAME,
    MODEL_KEY
)

print(
    f"Modèle uploadé vers "
    f"s3://{BUCKET_NAME}/best_model.pkl"
)

# Sauvegarde des métriques du meilleur modèle
metrics = {
    "model": best_model_name,
    "accuracy": float(best_accuracy),
    "roc_auc": float(
        roc_auc_logistic
        if best_model_name == "logistic"
        else roc_auc_multinomial
    )
}
METRICS_KEY = "metrics.json"

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

s3.upload_file(
    "metrics.json",
    BUCKET_NAME,
    METRICS_KEY
)

print("Métriques uploadées vers :")
print(f"s3://{BUCKET_NAME}/metrics.json")