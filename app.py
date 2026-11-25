"""Interface para testar mensagens com o modelo treinado."""
from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.joblib"

st.set_page_config(page_title="Detector de Spam", page_icon="🛡️")
st.title("🛡️ Detector Inteligente de Spam")
st.write(
    "Digite uma mensagem para que o modelo de Machine Learning estime "
    "se ela é legítima ou indesejada."
)
st.caption("Protótipo educacional — a previsão pode conter erros.")

@st.cache_resource
def carregar_modelo():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Modelo não encontrado. Primeiro execute: python train.py")
    st.stop()

modelo = carregar_modelo()
mensagem = st.text_area("Digite ou cole sua mensagem:", height=150,
                        placeholder="Ex.: Parabéns! Você ganhou um prêmio...")
if st.button("Fazer previsão", type="primary"):
    if not mensagem.strip():
        st.warning("Digite uma mensagem antes de continuar.")
    else:
        resultado = modelo.predict([mensagem])[0]
        probabilidades = modelo.predict_proba([mensagem])[0]
        classes = list(modelo.classes_)
        confianca = float(probabilidades[classes.index(resultado)])

        if resultado == "spam":
            st.error("Resultado: possível SPAM 🚨")
        else:
            st.success("Resultado: mensagem provavelmente legítima ✅")
        st.write(f"Modelo utilizado: **{modelo.named_steps['classificador'].__class__.__name__}**")
        st.write(f"Probabilidade estimada para a classe prevista: **{confianca:.1%}**")
        st.caption("A probabilidade é uma estimativa do modelo, não uma garantia.")
