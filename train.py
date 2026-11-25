"""Baixa os dados reais, treina dois classificadores e compara seus resultados."""
from pathlib import Path
import io
import json
import zipfile
import urllib.request

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
DATA_FILE = DATA_DIR / "SMSSpamCollection"
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

def load_data():
    """Obtém a base oficial da UCI caso ela ainda não exista localmente."""
    DATA_DIR.mkdir(exist_ok=True)
    if not DATA_FILE.exists():
        print("Baixando a base SMS Spam Collection da UCI...")
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            member = next(n for n in archive.namelist() if n.endswith("SMSSpamCollection"))
            DATA_FILE.write_bytes(archive.read(member))

    df = pd.read_csv(
        DATA_FILE, sep="\t", header=None, names=["classe", "mensagem"],
        encoding="latin-1"
    )
    df = df.dropna().drop_duplicates(subset="mensagem").reset_index(drop=True)
    df["classe"] = df["classe"].str.strip().str.lower()
    df = df[df["classe"].isin(["ham", "spam"])]
    return df

def main():
    df = load_data()
    print(f"Mensagens após limpeza: {len(df)}")
    print("\nDistribuição das classes:")
    print(df["classe"].value_counts())

    # Análise exploratória: proporção das classes e tamanho das mensagens.
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "graficos").mkdir(exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="classe")
    plt.title("Quantidade de mensagens por classe")
    plt.xlabel("Classe (ham = legítima; spam = indesejada)")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "graficos" / "distribuicao_classes.png", dpi=150)
    plt.close()

    df["tamanho_mensagem"] = df["mensagem"].str.len()
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="classe", y="tamanho_mensagem")
    plt.title("Tamanho das mensagens por classe")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade de caracteres")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "graficos" / "tamanho_mensagens.png", dpi=150)
    plt.close()

    X = df["mensagem"]
    y = df["classe"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # TF-IDF transforma o texto em números. Como está dentro do Pipeline,
    # o vocabulário é aprendido somente com os dados de treino.
    modelos = {
        "MultinomialNB": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode")),
            ("classificador", MultinomialNB())
        ]),
        "LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode")),
            ("classificador", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])
    }

    resultados = {}
    melhor_nome, melhor_f1, melhor_modelo = None, -1, None

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        previsoes = modelo.predict(X_test)
        metricas = {
            "acuracia": accuracy_score(y_test, previsoes),
            "precisao_spam": precision_score(y_test, previsoes, pos_label="spam", zero_division=0),
            "recall_spam": recall_score(y_test, previsoes, pos_label="spam", zero_division=0),
            "f1_spam": f1_score(y_test, previsoes, pos_label="spam", zero_division=0)
        }
        resultados[nome] = metricas
        print(f"\n{nome}")
        for chave, valor in metricas.items():
            print(f"{chave}: {valor:.4f}")
        print(classification_report(y_test, previsoes, zero_division=0))

        if metricas["f1_spam"] > melhor_f1:
            melhor_nome, melhor_f1, melhor_modelo = nome, metricas["f1_spam"], modelo

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(melhor_modelo, MODEL_DIR / "model.joblib")

    # Matriz de confusão do modelo selecionado.
    pred_melhor = melhor_modelo.predict(X_test)
    matriz = confusion_matrix(y_test, pred_melhor, labels=["ham", "spam"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legítima", "Spam"], yticklabels=["Legítima", "Spam"])
    plt.title(f"Matriz de confusão — {melhor_nome}")
    plt.xlabel("Previsão")
    plt.ylabel("Valor real")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "graficos" / "matriz_confusao.png", dpi=150)
    plt.close()

    saida = {
        "quantidade_mensagens": int(len(df)),
        "distribuicao_classes": {k: int(v) for k, v in df["classe"].value_counts().items()},
        "melhor_modelo": melhor_nome,
        "metricas": resultados,
        "relatorio_classificacao": classification_report(
            y_test, pred_melhor, output_dict=True, zero_division=0
        )
    }
    (REPORT_DIR / "metricas.json").write_text(
        json.dumps(saida, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nModelo escolhido: {melhor_nome}")
    print("Modelo salvo em models/model.joblib")
    print("Métricas e gráficos salvos na pasta reports/")

if __name__ == "__main__":
    main()
