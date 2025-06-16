import pandas as pd
from pymongo import MongoClient
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import random
from dotenv import load_dotenv
import os

# Carregar variáveis do arquivo .env
load_dotenv()

# 🔗 Conexão com MongoDB
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DATABASE = os.getenv("MONGO_DATABASE")
MONGO_CASES_COLLECTION = os.getenv("MONGO_CASES_COLLECTION")
MONGO_VICTIMS_COLLECTION = os.getenv("MONGO_VICTIMS_COLLECTION")

client = MongoClient(MONGO_URI)
db = client[MONGO_DATABASE]
cases = list(db[MONGO_CASES_COLLECTION].find({}, {"_id": 0}))
vitimas = list(db[MONGO_VICTIMS_COLLECTION].find({}, {"_id": 0}))

print(f"Total de casos: {len(cases)}")
print(f"Total de vítimas: {len(vitimas)}")

# 🔁 Associação simulada entre casos e vítimas
dados_combinados = []
for caso in cases:
    if not vitimas:
        continue
    vitima = random.choice(vitimas)

    try:
        dados_combinados.append({
            "idade": vitima.get("idadeAproximada"),
            "sexo": vitima.get("sexo"),
            "cidade": caso.get("cidade"),
            "estado": caso.get("estado"),
            "tipo_do_caso": caso.get("titulo")  # ou use `descricao` ou `status` se preferir
        })
    except Exception as e:
        print(f"Erro ao montar dados: {e}")
        continue

df = pd.DataFrame(dados_combinados)
df.dropna(inplace=True)

print(df.head())

# 🔤 Preparo para o modelo
X = df[["idade", "sexo", "cidade", "estado"]]
y = df["tipo_do_caso"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

categorical_features = ["sexo", "cidade", "estado"]
numeric_features = ["idade"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
])

pipeline.fit(X, y_encoded)

# 💾 Salvando modelo
MODEL_PATH = os.getenv("MODEL_PATH")
with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "pipeline": pipeline,
        "label_encoder": label_encoder
    }, f)

print("Modelo treinado e salvo com sucesso.")