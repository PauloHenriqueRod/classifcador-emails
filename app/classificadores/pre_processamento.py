import re
import unicodedata
from typing import List
import pandas as pd

import spacy
from spacy.lang.pt.stop_words import STOP_WORDS


class PreProcessadorEmail:
    def __init__(self):
        # Carregar modelo spacy para português
        self.nlp = spacy.load("pt_core_news_sm")
        
        # Stopwords customizadas (comuns em corpos de email)
        self.stopwords_email = {
            "atenciosamente",
            "cumprimentos",
            "melhor",
            "oi",
            "olá",
            "caro",
            "manhã",
            "tarde",
            "noite",
            "encaminhado",
            "mensagem",
            "email",
            "enviado",
            "de",
            "assunto",
            "segue",
            "anexo",
            "prezado",
        }

        self.stopwords = STOP_WORDS.union(self.stopwords_email)

    
    def preprocessar(self, texto: str) -> str:
        # Remover emails
        texto = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", texto)
        # Remover URLs
        texto = re.sub(r"http\S+|www\S+|https\S+", " ", texto, flags=re.MULTILINE)
        # Remover números
        texto = re.sub(r"\d+", " ", texto)
        # Remover caracteres especiais e normalizar
        texto = texto.lower()
        texto = unicodedata.normalize("NFKD", texto)
        texto = texto.encode("ascii", "ignore").decode("utf-8")
        # Remover pontuação e outros caracteres especiais
        texto = re.sub(r"[^\w\s]", " ", texto)

        # Processar com Spacy
        doc = self.nlp(texto)

        tokens = [
            token.lemma_
            for token in doc
            if token.lemma_ not in self.stopwords
            and not token.is_space
            and len(token.lemma_) > 2
        ]

        return " ".join(tokens) 

    def preprocessar_lote(self, textos: List[str]) -> List[str]:
        return [self.preprocessar(texto) for texto in textos]


if __name__ == "__main__":
    try:
        arquivo_entrada = "emails.csv"
        df_entrada = pd.read_csv(arquivo_entrada, encoding='utf-8')
        print(f"Arquivo carregado com {len(df_entrada)} linhas")
        print(f"Colunas disponíveis: {df_entrada.columns.tolist()}")
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        exit()

    