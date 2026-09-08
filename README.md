# French Sentiment Classification API

Projet de classification automatique de sentiments sur des avis en français, basé sur des modèles de Machine Learning et exposé via une API REST Flask.

Le projet intègre également Docker et LocalStack S3 afin de simuler un stockage cloud local pour sauvegarder et récupérer le modèle entraîné et ses métriques.

## Architecture

```text
                     Dataset Hugging Face
                              │
                              ▼
                         train.py
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
             TF-IDF + Logistic    TF-IDF + MultinomialNB
             Regression
                    │                   │
                    └─────────┬─────────┘
                              ▼
                     Comparaison modèles
                              │
                              ▼
                       best_model.pkl
                              │
                              ▼
                       LocalStack S3
                              │
                              ▼
                         predict.py
                              │
                              ▼
                        Flask REST API
                              │
                              ▼
                    Postman / cURL / Client
```

## Objectifs

* Préparer et nettoyer un dataset d'avis en français.
* Transformer les textes en variables numériques avec **TF-IDF**.
* Comparer plusieurs modèles de classification.
* Sélectionner automatiquement le meilleur modèle.
* Sauvegarder le modèle entraîné au format `.pkl`.
* Stocker le modèle et les métriques dans un bucket S3 simulé avec LocalStack.
* Développer une API REST avec Flask.
* Conteneuriser l'API avec Docker.
* Tester les prédictions avec Postman et cURL.
* Préparer une évolution du projet vers Google Cloud / Vertex AI.

## Dataset

Le projet utilise le dataset Hugging Face :

`gaouehalim/French-reviews-with-prediction-of-their-feelings`

Les principales colonnes utilisées sont :

| Colonne         | Description                |
| --------------- | -------------------------- |
| `id`            | Identifiant du commentaire |
| `clean_content` | Texte de l'avis            |
| `sentiment`     | Sentiment associé au texte |

La colonne `id` est supprimée avant l'entraînement.

La variable explicative est :

```python
X = df["clean_content"]
```

La variable cible est :

```python
y = df["sentiment"]
```

Les classes de sentiment sont :

* `positive`
* `negative`
* `neutral`

## Modèles utilisés

Deux modèles sont comparés.

### Logistic Regression

```text
TF-IDF → Logistic Regression
```

Configuration TF-IDF :

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)
```

### Multinomial Naive Bayes

```text
TF-IDF → MultinomialNB
```

Les modèles sont évalués avec :

* Accuracy
* Matrice de confusion
* Classification Report
* ROC-AUC multiclass

Le modèle obtenant la meilleure Accuracy est automatiquement sélectionné comme modèle final.

## Gestion des négations

La représentation TF-IDF utilise des unigrammes et bigrammes :

```python
ngram_range=(1, 2)
```

Cela permet notamment au modèle d'apprendre certaines expressions telles que :

```text
pas satisfait
pas bon
ne fonctionne
très satisfait
```

Une amélioration possible consiste à utiliser des trigrammes :

```python
TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 3)
)
```

Une autre évolution possible est l'ajout d'un traitement spécifique des négations françaises avant la vectorisation.

## Stockage avec LocalStack

LocalStack permet de simuler le service Amazon S3 localement.

Le bucket utilisé est :

```text
sentiment-model-bucket
```

Les fichiers stockés sont :

```text
best_model.pkl
metrics.json
```

Architecture :

```text
train.py
   │
   ├── best_model.pkl
   │
   └── metrics.json
           │
           ▼
     LocalStack S3
           │
           ▼
    sentiment-model-bucket
```

## Docker

L'API Flask est exécutée dans un conteneur Docker.

Le projet utilise :

```text
Python 3.14
```

Le service Flask est exposé sur :

```text
http://localhost:5000
```

LocalStack utilise :

```text
http://localhost:4566
```

À l'intérieur du réseau Docker, l'API communique avec LocalStack via :

```text
http://localstack:4566
```


### `train.py`

Responsable de :

* chargement du dataset ;
* préparation des données ;
* entraînement des modèles ;
* évaluation ;
* sélection du meilleur modèle ;
* sauvegarde du modèle ;
* upload du modèle vers S3 ;
* sauvegarde et upload des métriques.

### `predict.py`

Contient l'API Flask.

Elle récupère le modèle depuis LocalStack S3 et réalise les prédictions.

### `Dockerfile`

Permet de construire l'image Docker de l'API.

### `docker-compose.yml`

Permet de lancer :

* LocalStack ;
* l'API Flask.

## Installation

### 1. Cloner le projet

```bash
git clone <URL_DU_REPOSITORY>
cd pipeline_ml_SageMaker_Vertex_AI_sur_LocalStack
```

### 2. Créer l'environnement virtuel

Sous Windows :

```powershell
python -m venv venv
```

Activer l'environnement :

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```powershell
pip install -r requirements.txt
```

## Configuration

Créer un fichier `.env` :

```env
# Configuration de l'infrastructure Cloud (LocalStack)
AWS_ENDPOINT_URL=http://localhost:4566
BUCKET_NAME=sentiment-model-bucket
AWS_ACCESS_KEY_ID=mock_key
AWS_SECRET_ACCESS_KEY=mock_secret
AWS_DEFAULT_REGION=us-east-1

# Configuration de l'API Flask (Optionnel, utile pour le dev local hors Docker)
FLASK_ENV=development
FLASK_APP=app/main.py
```

Ces identifiants sont utilisés uniquement pour l'environnement LocalStack.

Ne jamais publier de véritables clés AWS dans le dépôt GitHub.

## Lancement de LocalStack

Démarrer LocalStack :

```powershell
docker compose up -d localstack
```

Vérifier les conteneurs :

```powershell
docker ps
```

Vérifier S3 :

```powershell
aws --endpoint-url=http://localhost:4566 s3 ls
```

## Entraînement du modèle

Lancer :

```powershell
python train.py
```

Le script :

1. charge le dataset ;
2. sépare les données en train/test ;
3. entraîne Logistic Regression ;
4. entraîne MultinomialNB ;
5. compare les performances ;
6. sélectionne le meilleur modèle ;
7. crée `best_model.pkl` ;
8. crée `metrics.json` ;
9. envoie les fichiers vers LocalStack S3.

Vérifier le contenu du bucket :

```powershell
aws --endpoint-url=http://localhost:4566 s3 ls s3://sentiment-model-bucket
```

Résultat attendu :

```text
best_model.pkl
metrics.json
```

## Lancer l'API

Construire l'image :

```powershell
docker compose build api
```

Démarrer l'API :

```powershell
docker compose up -d api
```

Vérifier :

```powershell
docker ps
```

Consulter les logs :

```powershell
docker compose logs api
```

## Tester l'API

### Vérifier l'API

```http
GET http://localhost:5000/
```

## Tests avec cURL

Sous Windows, utiliser `curl.exe` afin d'appeler le véritable programme cURL :

```powershell
curl.exe http://localhost:5000/health
curl.exe http://localhost:5000/metrics
```

Pour une prédiction :

```powershell
$body = '{"text":"Ce produit est excellent, je suis très satisfait !"}'

[System.IO.File]::WriteAllText(
    "body.json",
    $body,
    [System.Text.UTF8Encoding]::new($false)
)

curl.exe -X POST "http://localhost:5000/predict" `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@body.json"
```
## Tester dans Postman
URL
http://localhost:5000/predict
Méthode:
POST
## Configuration de Postman
Étape 1 — Sélectionner la méthode

Dans Postman, sélectionner :

POST

Puis saisir :

http://localhost:5000/predict
Étape 2 — Configurer le Body

Aller dans :

Body → raw → JSON

Puis saisir :

{
    "text": "Je suis très satisfait de ce produit, il est excellent !"
}

Étape 3 — Envoyer la requête

Cliquer sur : Send

## Réentraînement

Après une modification du modèle ou du preprocessing :

```powershell
python train.py
```

Le nouveau `best_model.pkl` remplace l'ancien modèle dans le bucket S3.

Il n'est pas nécessaire de reconstruire toute l'infrastructure Docker à chaque réentraînement.

Si le modèle est chargé au démarrage de l'API, redémarrer l'API après un nouvel entraînement :

```powershell
docker compose restart api
```


## Technologies utilisées

| Technologie               | Utilisation                         |
| ------------------------- | ----------------------------------- |
| Python                    | Langage de programmation            |
| Pandas                    | Manipulation des données            |
| Scikit-learn              | Machine Learning                    |
| TF-IDF                    | Vectorisation des textes            |
| Logistic Regression       | Classification                      |
| MultinomialNB             | Classification                      |
| Hugging Face Datasets     | Chargement du dataset               |
| Joblib                    | Sauvegarde du modèle                |
| Flask                     | API REST                            |
| Docker                    | Conteneurisation                    |
| LocalStack                | Simulation des services AWS         |
| Amazon S3 / LocalStack S3 | Stockage du modèle et des métriques |
| Postman                   | Tests de l'API                      |
| cURL                      | Tests de l'API                      |

## Bonnes pratiques

Ne pas versionner :

```text
.env
venv/
.venv/
localstack-data/
```

Ajouter ces éléments dans `.gitignore`.

Ne jamais publier de véritables clés AWS ou Google Cloud dans GitHub.

Projet réalisé à des fins d'apprentissage et de portfolio.
