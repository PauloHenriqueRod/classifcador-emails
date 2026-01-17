from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Carregar o modelo treinado
MODEL_PATH = "classificadores/modelo_classificacao.pkl"

# Respostas sugeridas baseadas na classificação
RESPOSTAS_SUGERIDAS = {
    "Produtivo": [
        {
            "titulo": "Resposta Padrão de Priorização",
            "texto": "Prezado(a),\n\nRecebemos sua mensagem e já priorizamos seu atendimento. Nossa equipe está analisando a solicitação e retornaremos com uma posição em breve.\n\nEstamos à disposição para quaisquer esclarecimentos adicionais.\n\nAtenciosamente,"
        },
        {
            "titulo": "Confirmação de Recebimento com Prazo",
            "texto": "Olá,\n\nConfirmamos o recebimento de sua solicitação. Estamos trabalhando para resolver esta questão e você receberá nosso retorno em até [X] dias úteis.\n\nCaso necessite de informações urgentes, não hesite em nos contatar.\n\nCordialmente,"
        },
        {
            "titulo": "Encaminhamento para Equipe Responsável",
            "texto": "Prezado(a),\n\nSua mensagem foi recebida e encaminhada para a equipe responsável. Eles entrarão em contato em breve para dar continuidade ao seu atendimento.\n\nAgradecemos pela compreensão.\n\nAtenciosamente,"
        }
    ],
    "Improdutivo": [
        {
            "titulo": "Resposta Educada de Redirecionamento",
            "texto": "Prezado(a),\n\nAgradecemos pelo contato. Para melhor atendê-lo, sugerimos que envie sua solicitação através dos canais apropriados ou com mais detalhes sobre o que precisa.\n\nEstamos à disposição para ajudá-lo.\n\nCordialmente,"
        },
        {
            "titulo": "Resposta de Informação Adicional",
            "texto": "Olá,\n\nRecebemos sua mensagem. Para que possamos auxiliá-lo da melhor forma, precisaríamos de mais informações sobre sua necessidade específica.\n\nPor favor, nos forneça mais detalhes para que possamos direcionar adequadamente seu atendimento.\n\nAtenciosamente,"
        },
        {
            "titulo": "Resposta de Baixa Prioridade",
            "texto": "Prezado(a),\n\nSua mensagem foi registrada em nosso sistema. Responderemos assim que possível, de acordo com nossa ordem de prioridades.\n\nAgradecemos pela compreensão.\n\nCordialmente,"
        }
    ]
}

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
        
        # Obter respostas sugeridas para a classificação
        respostas_sugeridas = RESPOSTAS_SUGERIDAS.get(predicao, [])
        
        return jsonify({
            "texto": texto,
            "classificacao": predicao,
            "confianca": confianca,
            "respostas_sugeridas": respostas_sugeridas,
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
