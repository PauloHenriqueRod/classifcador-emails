import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


df = pd.read_csv("emails_processados.csv")

df = df.dropna(subset=["texto_preprocessado", "label"])

df_balanceado = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 2000), random_state=42)
)

df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)

X = df_balanceado["texto_preprocessado"]
y = df_balanceado["label"]


X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # importante para classes balanceadas
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2
    )),
    ("classificador", LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    ))
])

pipeline.fit(X_treino, y_treino)

# Salvar o modelo treinado
joblib.dump(pipeline, "modelo_classificacao.pkl")
print("Modelo salvo em: modelo_classificacao.pkl")

y_pred = pipeline.predict(X_teste)

print("Acurácia:", accuracy_score(y_teste, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_teste, y_pred))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_teste, y_pred))



novos_emails = [
    "preciso de sua assistência com o prazo do projeto",
    "parabéns pelo excelente trabalho que você realizou",
]

predicoes = pipeline.predict(novos_emails)

for email, pred in zip(novos_emails, predicoes):
    print(f"\nEmail: {email}")
    print(f"Classe prevista: {pred}")
