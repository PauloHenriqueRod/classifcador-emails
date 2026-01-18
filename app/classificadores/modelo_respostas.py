import re
from typing import Dict, List, Tuple
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
from datetime import datetime
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class AnalisadorContexto:
    
    def __init__(self):
        self.problemas_conhecidos = {
            "acesso": {
                "palavras": ["acesso", "permissão", "liberação", "autorização", "credenciais"],
                "sistemas": ["confluence", "gitlab", "jira", "sharepoint", "erp", "sap"],
                "urgencia_base": 0.6
            },
            "indisponibilidade": {
                "palavras": ["fora do ar", "indisp", "parado", "travado", "congelado", "sem responder"],
                "sistemas": ["api", "servidor", "banco dados", "sistema", "aplicação"],
                "urgencia_base": 0.9
            },
            "erro_sistema": {
                "palavras": ["erro", "bug", "falha", "exceção", "erro:", "code:", "stacktrace"],
                "sistemas": ["sistema", "aplicação", "módulo", "integração"],
                "urgencia_base": 0.7
            },
            "performance": {
                "palavras": ["lento", "demora", "performance", "lag", "timeout", "travando"],
                "sistemas": ["banco", "servidor", "rede", "api"],
                "urgencia_base": 0.6
            },
            "dados": {
                "palavras": ["relatório", "dados", "informação", "export", "backup", "restore"],
                "sistemas": ["banco", "data warehouse", "bi"],
                "urgencia_base": 0.5
            },
            "segurança": {
                "palavras": ["segurança", "hack", "vazamento", "acesso indevido", "suspeito"],
                "sistemas": ["sistema", "aplicação", "rede"],
                "urgencia_base": 0.95
            }
        }
        
        
        self.padroes_problema = {
            r"(?:ticket|chamado|caso|protocolo)\s*#?(\d+)": "ticket_referencia",
            r"(?:erro|erro code)\s*:?\s*(\d+|[A-Z0-9]+)": "codigo_erro",
            r"(?:versão|v\.?)\s*([\d\.]+)": "versao_software",
            r"(?:ambiente|env)\s*:?\s*(\w+)": "ambiente",
            r"(?:navegador|browser)\s*:?\s*(\w+)": "navegador",
            r"(?:sistem operacional|so|windows|linux|mac)\s*:?\s*(\w+)": "so",
        }
    
    def analisar_tipo_problema(self, texto: str) -> Dict:
        texto_lower = texto.lower()
        
        tipos_encontrados = {}
        max_score = 0
        tipo_principal = None
        
        for tipo, config in self.problemas_conhecidos.items():
            score = 0
            palavras_encontradas = []
            
            for palavra in config["palavras"]:
                if palavra in texto_lower:
                    score += 1
                    palavras_encontradas.append(palavra)
            
            if score > 0:
                tipos_encontrados[tipo] = {
                    "score": score,
                    "palavras": palavras_encontradas,
                    "urgencia_base": config["urgencia_base"]
                }
                if score > max_score:
                    max_score = score
                    tipo_principal = tipo
        
        return {
            "tipos": tipos_encontrados,
            "tipo_principal": tipo_principal,
            "confianca": min(1.0, max_score / 3) if max_score > 0 else 0
        }
    
    def extrair_informacoes_tecnicas(self, texto: str) -> Dict:
        info_tecnica = {
            "ticket_numero": [],
            "codigo_erro": [],
            "versao": [],
            "ambiente": [],
            "navegador": [],
            "sistema_operacional": []
        }
        
        for padrao, categoria in self.padroes_problema.items():
            matches = re.findall(padrao, texto, re.IGNORECASE)
            if matches:
                if categoria == "ticket_referencia":
                    info_tecnica["ticket_numero"] = matches
                elif categoria == "codigo_erro":
                    info_tecnica["codigo_erro"] = matches
                elif categoria == "versao_software":
                    info_tecnica["versao"] = matches
                elif categoria == "ambiente":
                    info_tecnica["ambiente"] = matches
                elif categoria == "navegador":
                    info_tecnica["navegador"] = matches
                elif categoria == "so":
                    info_tecnica["sistema_operacional"] = matches
        
        return info_tecnica
    
    def detectar_tons(self, texto: str) -> Dict:
        texto_lower = texto.lower()
        
        tons = {
            "formal": len(re.findall(r'\b(prezado|estimado|prezadíssim|cumprimento)\b', texto_lower)),
            "informal": len(re.findall(r'\b(oi|olá|galera|pessoal|fala)\b', texto_lower)),
            "frustrado": len(re.findall(r'\b(frustrado|insatisfeito|desapontado|decepcionado)\b', texto_lower)),
            "cortês": len(re.findall(r'\b(por favor|obrigado|agradeço|poderia|teria|gostaria)\b', texto_lower)),
            "imperativo": len(re.findall(r'(?:você deve|precisa|necessário|é preciso|exijo|quero)\b', texto_lower)),
        }
        
        tom_principal = max(tons, key=tons.get)
        score_ton = tons[tom_principal] / max(sum(tons.values()), 1)
        
        return {
            "tons": tons,
            "tom_principal": tom_principal,
            "confianca": score_ton
        }
    
    def analisar_contexto_temporal(self, texto: str) -> Dict:
        texto_lower = texto.lower()
        
        contexto_temporal = {
            "urgencia_temporal": 0,
            "referencias_tempo": []
        }
        
        
        urgentes = re.findall(r'\b(hoje|agora|imediatamente|urgente|pressa|breve|ontem|há \d+ dias?)\b', texto_lower)
        if urgentes:
            contexto_temporal["referencias_tempo"] = urgentes
            contexto_temporal["urgencia_temporal"] = min(1.0, len(urgentes) * 0.3)
        
        
        if any(palavra in texto_lower for palavra in ["atrasado", "vencido", "expirou", "passou"]):
            contexto_temporal["urgencia_temporal"] += 0.3
        
        return contexto_temporal


class GeradorRespostas:
    
    def __init__(self):
        try:
            self.nlp = spacy.load("pt_core_news_sm")
        except:
            print("Aviso: Modelo spacy não carregado.")
            self.nlp = None
        
        self.analisador = AnalisadorContexto()
        
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._nn = NearestNeighbors(n_neighbors=1, metric="cosine")
        self._labels_tfidf: List[str] = []
        self._treinar_sugeridor_tipos()
        
        
        self.templates_por_tipo = {
            "acesso": {
                "alta": "Prezado(a),\n\nSua solicitação de {sistema} foi recebida e está sendo processada com MÁXIMA PRIORIDADE.\nNossa equipe de segurança e TI foi acionada para validar os requisitos necessários.\nVocê receberá a confirmação de acesso em até 1-2 dias úteis.\n\nReferência: {ticket}\n\nAtenciosamente,",
                "média": "Olá,\n\nRecebemos sua requisição de {sistema}. Estamos validando as permissões necessárias com a equipe responsável.\nVocê receberá nosso retorno em até 2-3 dias úteis.\n\nReferência: {ticket}\n\nCordialmente,",
                "baixa": "Prezado(a),\n\nSua solicitação foi recebida e registrada em nosso sistema.\nProcessaremos conforme a ordem de prioridades e você será contatado em breve.\n\nReferência: {ticket}\n\nAtenciosamente,"
            },
            "indisponibilidade": {
                "alta": "Prezadíssimo(a), CRÍTICO: Identificamos que {sistema} está {status}.\n\nNossa equipe técnica ACABA DE SER ACIONADA para investigação imediata.\nEste é um incidente crítico e estamos trabalhando para restauração urgente.\n\nAtualizaremos você a cada 30 minutos.\n\nReferência: {ticket}\n\nMelhores cumprimentos,",
                "média": "Prezado(a),\n\nIdentificamos que {sistema} não está respondendo adequadamente.\nNossa equipe técnica está investigando o problema e trabalhando na restauração.\n\nEstaremos em contato em breve com atualizações.\n\nReferência: {ticket}\n\nAtenciosamente,",
                "baixa": "Olá,\n\nRecebemos o relato de indisponibilidade em {sistema}.\nEstamos investigando e retornaremos com informações em breve.\n\nReferência: {ticket}\n\nCordialmente,"
            },
            "erro_sistema": {
                "alta": "Prezado(a),\n\nIdentificamos o erro {codigo_erro} em {sistema}.\n\nNossa equipe de desenvolvimento foi acionada. Este é um problema crítico e estamos trabalhando na solução urgente.\nEsperamos resolver em {prazo}.\n\nReferência: {ticket}\nAmbiente: {ambiente}\n\nAtenciosamente,",
                "média": "Olá,\n\nRecebemos o relato do erro em {sistema}. Nossa equipe técnica está analisando a causa raiz.\nTrabalhamos para resolver o mais breve possível.\n\nReferência: {ticket}\n\nCordialmente,",
                "baixa": "Prezado(a),\n\nObrigado por reportar o erro. Estamos investigando e retornaremos com um diagnóstico em breve.\n\nReferência: {ticket}\n\nAtenciosamente,"
            },
            "performance": {
                "alta": "Prezado(a),\n\nIdentificamos problemas de performance em {sistema}.\nNossa equipe de infraestrutura está investigando possíveis gargalos.\nPriorizaremos a solução e retornaremos em breve.\n\nReferência: {ticket}\n\nAtenciosamente,",
                "média": "Olá,\n\nRecebemos seu relato sobre a lentidão em {sistema}.\nEstamos analisando a performance e possíveis causas.\n\nReferência: {ticket}\n\nCordialmente,",
                "baixa": "Prezado(a),\n\nObrigado pelo feedback sobre performance.\nIremos investigar e otimizar quando possível.\n\nReferência: {ticket}\n\nAtenciosamente,"
            },
            "dados": {
                "alta": "Prezado(a),\n\nRecebemos sua solicitação de dados com urgência.\nNossa equipe de analytics está preparando o relatório/export solicitado.\nEntrega prevista: {prazo}.\n\nReferência: {ticket}\n\nAtenciosamente,",
                "média": "Olá,\n\nSua solicitação de dados foi recebida.\nEstamos compilando as informações necessárias e enviaremos em breve.\n\nReferência: {ticket}\n\nCordialmente,",
                "baixa": "Prezado(a),\n\nRecebemos sua solicitação de dados.\nEntraremos em contato com as informações solicitadas.\n\nReferência: {ticket}\n\nAtenciosamente,"
            },
            "segurança": {
                "alta": "CRÍTICO - SEGURANÇA DA INFORMAÇÃO\n\nPrezadíssimo(a),\n\nIdentificamos uma possível ameaça à segurança conforme relatado.\nNossa equipe de segurança foi IMEDIATAMENTE ACIONADA para investigação e contenção.\n\nTrabalhamos com máxima urgência para remediar qualquer vulnerabilidade.\nEntre em contato conosco por telefone para detalhes sensíveis.\n\nReferência: {ticket}\n\nMelhores cumprimentos,",
                "média": "Prezado(a),\n\nObrigado por relatar a questão de segurança.\nNossa equipe de segurança está investigando com prioridade.\n\nReferência: {ticket}\n\nAtenciosamente,",
                "baixa": "Prezado(a),\n\nRecebemos sua comunicação sobre segurança.\nInvestigaremos conforme o protocolo de segurança da informação.\n\nReferência: {ticket}\n\nCordialmente,"
            }
        }
    
    def _treinar_sugeridor_tipos(self) -> None:
        corpus = []
        labels = []
        for tipo, cfg in self.analisador.problemas_conhecidos.items():
            texto_base = " ".join(cfg["palavras"] + cfg.get("sistemas", []))
            corpus.append(texto_base)
            labels.append(tipo)
        if not corpus:
            return
        X = self._tfidf.fit_transform(corpus)
        self._nn.fit(X)
        self._labels_tfidf = labels

    def calculo_similaridade(self, texto: str) -> Tuple[str, float]:
        if not texto.strip():
            return "", 0.0
        try:
            vec = self._tfidf.transform([texto])
            dist, idx = self._nn.kneighbors(vec, n_neighbors=1)
            sim = 1 - float(dist[0][0])
            label = self._labels_tfidf[int(idx[0][0])] if self._labels_tfidf else ""
            return label, max(0.0, sim)
        except Exception:
            return "", 0.0
    
    def avaliar_severidade_contextual(self, texto: str, classificacao: str, urgencia_detectada: str) -> Tuple[float, str]:
        """
        Avalia a severidade contextualmente, combinando múltiplos sinais
        """
        analise_problema = self.analisador.analisar_tipo_problema(texto)
        contexto_temporal = self.analisador.analisar_contexto_temporal(texto)
        
        
        score = 0
        
        
        if classificacao.lower() == "produtivo":
            score += 0.4
        
        
        if urgencia_detectada == "alta":
            score += 0.4
        elif urgencia_detectada == "média":
            score += 0.2
        
        
        if analise_problema["tipo_principal"]:
            score += analise_problema["confianca"] * 0.3
        
       
        score += min(contexto_temporal["urgencia_temporal"] * 0.3, 0.3)
        
        score = min(score, 1.0)
        
        
        if score >= 0.7:
            severidade = "crítica"
        elif score >= 0.5:
            severidade = "alta"
        elif score >= 0.3:
            severidade = "média"
        else:
            severidade = "baixa"
        
        return score, severidade
    
    def gerar_resposta_avancada(self, texto_email: str, classificacao: str) -> Dict:
        analise_problema = self.analisador.analisar_tipo_problema(texto_email)
        info_tecnica = self.analisador.extrair_informacoes_tecnicas(texto_email)
        tons = self.analisador.detectar_tons(texto_email)
        contexto_temporal = self.analisador.analisar_contexto_temporal(texto_email)
        tipo_ml, score_ml = self.calculo_similaridade(texto_email)
        
        nivel_urgencia = self._detectar_urgencia_basica(texto_email)
        
        score_severidade, severidade = self.avaliar_severidade_contextual(
            texto_email, classificacao, nivel_urgencia
        )
        
        nivel_resposta = "alta" if severidade in ["crítica", "alta"] else "média" if severidade == "média" else "baixa"
        
        tipo_principal = analise_problema["tipo_principal"] or "acesso"
        if (not analise_problema["tipo_principal"] or analise_problema["confianca"] < 0.3) and score_ml >= 0.2:
            tipo_principal = tipo_ml or tipo_principal
        resposta_base = self._obter_template(tipo_principal, nivel_resposta)
        
  
        resposta = self._resposta_personalizda(
            resposta_base,
            info_tecnica,
            tipo_principal,
            severidade
        )
        
        return {
            "resposta": resposta,
            "analise_problema": analise_problema,
            "info_tecnica": info_tecnica,
            "tons": tons,
            "severidade": severidade,
            "score_severidade": score_severidade,
            "recomendacoes": self._gerar_recomendacoes(
                analise_problema, tons, contexto_temporal, classificacao
            ),
            "follow_up": self._sugerir_follow_up(tipo_principal, severidade, info_tecnica)
        }
    
    def _detectar_urgencia_basica(self, texto: str) -> str:
        texto_lower = texto.lower()
        urgencia_alta = sum(1 for p in ["urgente", "crítico", "emergência", "prioridade", "rápido"] if p in texto_lower)
        urgencia_media = sum(1 for p in ["necessário", "importante", "precisamos"] if p in texto_lower)
        
        if urgencia_alta > 0:
            return "alta"
        elif urgencia_media > 0:
            return "média"
        else:
            return "baixa"
    
    def _obter_template(self, tipo: str, nivel: str) -> str:
        if tipo in self.templates_por_tipo:
            return self.templates_por_tipo[tipo].get(nivel, self.templates_por_tipo[tipo]["baixa"])
        return "Prezado(a),\n\nRecebemos sua mensagem e estamos processando sua solicitação.\n\nAtenciosamente,"
    
    def _resposta_personalizda(self, resposta: str, info_tecnica: Dict, tipo: str, severidade: str) -> str:
        if info_tecnica["ticket_numero"]:
            resposta = resposta.replace("{ticket}", f"Ticket #{info_tecnica['ticket_numero'][0]}")
        else:
            resposta = resposta.replace("\nReferência: {ticket}", "")
        
        resposta = resposta.replace("{sistema}", tipo.replace("_", " "))
        
        if info_tecnica["codigo_erro"]:
            resposta = resposta.replace("{codigo_erro}", info_tecnica["codigo_erro"][0])
        else:
            resposta = resposta.replace("{codigo_erro}", "identificado")
        
        if info_tecnica["ambiente"]:
            resposta = resposta.replace("{ambiente}", info_tecnica["ambiente"][0])
        else:
            resposta = resposta.replace("Ambiente: {ambiente}\n", "")
        
        if severidade == "crítica":
            prazo = "1-2 horas"
        elif severidade == "alta":
            prazo = "4-8 horas"
        else:
            prazo = "1-2 dias úteis"
        resposta = resposta.replace("{prazo}", prazo)
        
        resposta = resposta.replace("{status}", "indisponível")
        
        return resposta
    
    def _gerar_recomendacoes(self, analise_problema: Dict, tons: Dict, contexto_temporal: Dict, classificacao: str) -> List[str]:
        recomendacoes = []
        
        if analise_problema["tipo_principal"] == "segurança":
            recomendacoes.append("✓ Escalar para equipe de segurança da informação IMEDIATAMENTE")
        
        if analise_problema["tipo_principal"] == "indisponibilidade":
            recomendacoes.append("✓ Verificar status dos sistemas críticos")
            recomendacoes.append("✓ Notificar todas as áreas afetadas")
        
        if tons["tons"]["frustrado"] > 2:
            recomendacoes.append("✓ Considerar contato telefônico para melhor relacionamento")
        
        if tons["tons"]["imperativo"] > 1:
            recomendacoes.append("✓ Priorizar este atendimento")
        
        if contexto_temporal["urgencia_temporal"] > 0.6:
            recomendacoes.append("✓ Solicitar aprovação de SLA reduzido")
        
        if classificacao.lower() == "produtivo":
            recomendacoes.append("✓ Manter acompanhamento próximo durante resolução")
        
        return recomendacoes
    
    def _sugerir_follow_up(self, tipo: str, severidade: str, info_tecnica: Dict) -> Dict:
        if severidade == "crítica":
            intervalo = "30 minutos"
        elif severidade == "alta":
            intervalo = "2 horas"
        else:
            intervalo = "24 horas"
        
        return {
            "intervalo": intervalo,
            "tipo": f"Atualização sobre {tipo}",
            "ticket": info_tecnica["ticket_numero"][0] if info_tecnica["ticket_numero"] else "N/A"
        }
    
    def gerar_multiplas_opcoes_avancadas(self, texto_email: str, classificacao: str, num_opcoes: int = 3) -> List[Dict]:
        resposta_avancada = self.gerar_resposta_avancada(texto_email, classificacao)
        
        opcoes = [
            {
                "titulo": f"Resposta Contextualizada - {resposta_avancada['severidade'].upper()}",
                "texto": resposta_avancada["resposta"],
                "confianca": resposta_avancada["score_severidade"],
                "recomendacoes": resposta_avancada["recomendacoes"],
                "follow_up": resposta_avancada["follow_up"],
                "analise_problema": resposta_avancada["analise_problema"],
                "info_tecnica": resposta_avancada["info_tecnica"]
            }
        ]
        
        return opcoes[:num_opcoes]