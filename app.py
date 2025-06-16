from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from pymongo import MongoClient
import pandas as pd
from datetime import datetime
import pickle
import random
from dotenv import load_dotenv
import os

# Carregar variáveis do arquivo .env
load_dotenv()

# 🔧 Inicialização
app = Flask(__name__)
CORS(app)

# 🌐 Conexão com MongoDB Atlas
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DATABASE = os.getenv("MONGO_DATABASE")
MONGO_CASES_COLLECTION = os.getenv("MONGO_CASES_COLLECTION")
MONGO_VICTIMS_COLLECTION = os.getenv("MONGO_VICTIMS_COLLECTION")

client = MongoClient(MONGO_URI)
db = client[MONGO_DATABASE]
colecao_casos = db[MONGO_CASES_COLLECTION]
colecao_vitimas = db[MONGO_VICTIMS_COLLECTION]

# 🔍 Carregamento do modelo
MODEL_PATH = os.getenv("MODEL_PATH")
try:
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
        modelo = data["pipeline"]
        label_encoder = data["label_encoder"]
        print("Modelo carregado com sucesso")
except Exception as e:
    print(f"Erro ao carregar modelo: {str(e)}")
    modelo = None
    label_encoder = None

@app.route('/')
def home():
    return jsonify({"mensagem": "API ODX-Perícias com predição está funcionando!"}), 200

# 📦 Listar todos os casos
@app.route('/api/casos', methods=['GET'])
def listar_casos():
    casos = list(colecao_casos.find({}, {"_id": 0}))
    return jsonify(casos), 200

# 📦 Listar todas as vítimas
@app.route('/api/vitimas', methods=['GET'])
def listar_vitimas():
    vitimas = list(colecao_vitimas.find({}, {"_id": 0}))
    return jsonify(vitimas), 200

# 🧠 Predizer tipo de caso com base nas informações
@app.route('/api/predizer', methods=['POST'])
def predizer():
    if not modelo or not label_encoder:
        return jsonify({"erro": "Modelo não carregado"}), 500

    dados = request.get_json()
    if not dados or not all(k in dados for k in ("idade", "sexo", "cidade", "estado")):
        return jsonify({"erro": "JSON inválido. Esperado: idade, sexo, cidade, estado"}), 400

    try:
        df = pd.DataFrame([dados])
        y_prob = modelo.predict_proba(df)[0]
        y_pred_encoded = modelo.predict(df)[0]
        y_pred = label_encoder.inverse_transform([y_pred_encoded])[0]
        classes = label_encoder.classes_

        resultado = {
            "classe_predita": y_pred,
            "probabilidades": {classe: round(float(prob), 4) for classe, prob in zip(classes, y_prob)}
        }
        return jsonify(resultado), 200
    except Exception as e:
        return jsonify({"erro": f"Erro ao fazer predição: {str(e)}"}), 500

# 🔍 Testar importância das features (coeficientes)
@app.route('/api/modelo/coefficients', methods=['GET'])
def coefficients_modelo():
    if not modelo:
        return jsonify({"erro": "Modelo não carregado"}), 500

    try:
        preprocessor = modelo.named_steps['preprocessor']
        classifier = modelo.named_steps['classifier']

        cat_encoder = preprocessor.named_transformers_['cat']
        cat_features = cat_encoder.get_feature_names_out(["sexo", "cidade", "estado"])
        numeric_features = ["idade"]
        all_features = list(cat_features) + list(numeric_features)

        importancias = classifier.feature_importances_

        features_importances = {
            feature: float(importance)
            for feature, importance in zip(all_features, importancias)
        }
        return jsonify(features_importances), 200
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

# 🧪 Gerar dados mock (se necessário)
@app.route('/api/gerar-dados-mock', methods=['POST'])
def gerar_mock():
    cidades = ["São Paulo", "Recife", "Salvador", "Rio de Janeiro"]
    estados = ["SP", "PE", "BA", "RJ"]
    sexos = ["masculino", "feminino"]
    titulos = ["Assassinato", "Furto", "Violência Doméstica", "Tráfico"]

    vitimas_mock = []
    casos_mock = []

    for i in range(10):
        idade = random.randint(15, 90)
        sexo = random.choice(sexos)
        cidade = random.choice(cidades)
        estado = estados[cidades.index(cidade)]
        titulo = random.choice(titulos)

        vitima = {
            "nome": f"Pessoa {i}",
            "dataNascimento": str(datetime.now()),
            "idadeAproximada": idade,
            "nacionalidade": "Brasileira",
            "cidade": cidade,
            "sexo": sexo,
            "estadoCorpo": "inteiro",
            "lesoes": "lesão leve",
            "identificada": True
        }

        caso = {
            "titulo": titulo,
            "descricao": "Descrição gerada automaticamente",
            "status": "Em andamento",
            "responsavel": "mock_id",
            "dataCriacao": str(datetime.now()),
            "cidade": cidade,
            "estado": estado,
            "casoReferencia": f"CR-{random.randint(1000,9999)}"
        }

        vitimas_mock.append(vitima)
        casos_mock.append(caso)

    db[MONGO_VICTIMS_COLLECTION].insert_many(vitimas_mock)
    db[MONGO_CASES_COLLECTION].insert_many(casos_mock)

    return jsonify({"mensagem": "Dados mock inseridos com sucesso"}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)