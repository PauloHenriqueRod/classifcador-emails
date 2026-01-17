import os
import pandas as pd

from pre_processamento import PreProcessadorEmail

# Configuração dos arquivos
ARQUIVO_ENTRADA = "emails_produtivoo_improdutivos.csv"
ARQUIVO_SAIDA = "emails_processados.csv"


def preprocessar_emails_csv(arquivo_entrada: str, arquivo_saida: str) -> None:
    """
    Realiza o pré-processamento dos emails do CSV e salva em novo arquivo.
    
    Args:
        arquivo_entrada: Caminho do CSV de entrada com colunas 'texto' e 'label'
        arquivo_saida: Caminho do CSV de saída com textos pré-processados
    """
    try:
        # Carregar o CSV
        print(f"Lendo dataset de: {arquivo_entrada}")
        df = pd.read_csv(arquivo_entrada, encoding='utf-8')
        
        print(f"Arquivo carregado com {len(df)} linhas")
        print(f"Colunas disponíveis: {df.columns.tolist()}")
        
        # Remover linhas com valores nulos
        df_limpo = df.dropna(subset=['texto', 'label'])
        print(f"Após remover valores nulos: {len(df_limpo)} linhas")
        
        # Inicializar o pré-processador
        preprocessador = PreProcessadorEmail()
        
        # Realizar pré-processamento
        print("Realizando pré-processamento...")
        registros = []
        
        for idx, linha in df_limpo.iterrows():
            texto = str(linha['texto']).strip()
            label = str(linha['label']).strip()
            
            # Pré-processar o texto
            texto_preprocessado = preprocessador.preprocessar(texto)
            
            # Adicionar apenas se houve resultado
            if texto_preprocessado:
                registros.append({
                    "texto": texto,
                    "texto_preprocessado": texto_preprocessado,
                    "label": label
                })
        
        # Criar DataFrame com o resultado
        df_saida = pd.DataFrame(registros)
        
        # Salvar o resultado
        df_saida.to_csv(arquivo_saida, index=False, encoding='utf-8')
        
        print(f"\nDataset processado com {len(df_saida)} emails")
        print(f"Arquivo salvo em: {arquivo_saida}")
        print(f"\nDistribuição de rótulos:")
        print(df_saida['label'].value_counts())
        
    except FileNotFoundError:
        print(f"Erro: Arquivo '{arquivo_entrada}' não encontrado")
    except KeyError as e:
        print(f"Erro: Coluna não encontrada {e}. Esperadas: 'texto' e 'label'")
    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")


if __name__ == "__main__":
    preprocessar_emails_csv(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)
