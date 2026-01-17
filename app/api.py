from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Carregar o modelo treinado
MODEL_PATH = "classificadores/modelo_classificacao.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"Erro: Modelo não encontrado em {MODEL_PATH}")
    print("Execute primeiro: python treinamento_modelo.py")
    pipeline = None
else:
    pipeline = joblib.load(MODEL_PATH)
    print(f"Modelo carregado com sucesso de {MODEL_PATH}")


@app.route('/', methods=['GET'])
def home():
    """Retorna a página HTML para interface de testes"""
    return open('interface/index.html', 'r', encoding='utf-8').read()


@app.route('/api/classificar', methods=['POST'])
def classificar():
    """
    Recebe um texto e retorna a classificação
    
    Exemplo de requisição:
    {
        "texto": "preciso de sua assistência com o prazo do projeto"
    }
    """
    try:
        if pipeline is None:
            return jsonify({
                "erro": "Modelo não carregado. Execute primeiro: python treinamento_modelo.py"
            }), 500
        
        # Obter o JSON da requisição
        dados = request.get_json()
        
        if not dados or 'texto' not in dados:
            return jsonify({
                "erro": "Campo 'texto' é obrigatório"
            }), 400
        
        texto = dados['texto'].strip()
        
        if not texto:
            return jsonify({
                "erro": "Texto não pode estar vazio"
            }), 400
        
        # Fazer a predição
        predicao = pipeline.predict([texto])[0]
        
        # Obter probabilidades (se disponível)
        try:
            probabilidades = pipeline.predict_proba([texto])[0]
            classes = pipeline.named_steps['classificador'].classes_
            confianca = {
                classe: float(prob) 
                for classe, prob in zip(classes, probabilidades)
            }
        except:
            confianca = {}
        
        return jsonify({
            "texto": texto,
            "classificacao": predicao,
            "confianca": confianca,
            "sucesso": True
        }), 200
    
    except Exception as e:
        return jsonify({
            "erro": str(e),
            "sucesso": False
        }), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Retorna o status da API"""
    modelo_carregado = pipeline is not None
    return jsonify({
        "status": "ativo",
        "modelo_carregado": modelo_carregado,
        "modelo_arquivo": MODEL_PATH
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
