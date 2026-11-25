# ML Challenge — Detector Inteligente de Spam

**Estudante:** [Digite seu nome completo]  
**Trilha:** D — Filtro antispam  
**Tipo de problema:** Classificação de texto

## 1. Problema

O projeto desenvolve um protótipo que recebe uma mensagem de texto e classifica seu conteúdo como `spam` (mensagem indesejada) ou `ham` (mensagem legítima). A aplicação pode ajudar usuários a identificar mensagens suspeitas, mas não substitui a análise humana.

## 2. Base de dados

Foi escolhida a **SMS Spam Collection**, do UCI Machine Learning Repository. A base reúne 5.574 mensagens SMS rotuladas. A variável-alvo é `classe`, com os valores `ham` e `spam`; a entrada do modelo é o texto da mensagem.

- Fonte: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Autores: Tiago Almeida e José Hidalgo (2011).
- Licença: CC BY 4.0, conforme a página da UCI.
- A base é baixada automaticamente por `train.py`. Consulte `data/README.md`.

## 3. Como executar

Recomendado: Python 3.10 ou superior.

```bash
python -m venv .venv
```

No Windows:
```bash
.venv\Scripts\activate
```

Instale as dependências:
```bash
pip install -r requirements.txt
```

Treine e avalie os modelos:
```bash
python train.py
```

Inicie a aplicação:
```bash
streamlit run app.py
```

Abra o endereço local mostrado no terminal pelo Streamlit.

## 4. Análise exploratória

| Modelo | Acurácia | Precisão (spam) | Recall (spam) | F1 (spam) |
|---|---:|---:|---:|---:|
| MultinomialNB | 95,26% | 100% | 62,60% | 76,99% |
| LogisticRegression | 97,39% | 88,24% | 91,60% | 89,89% |

**Modelo escolhido:** Regressão Logística.

A Regressão Logística apresentou o melhor equilíbrio
entre precisão e recall, alcançando F1-score de 89,89%
para a identificação de mensagens spam.

## 5. Pré-processamento

- Remoção de linhas sem conteúdo e duplicatas.
- Separação entre `X` (mensagem) e `y` (classe).
- Divisão em 80% para treino e 20% para teste, com `random_state=42` e estratificação.
- `TfidfVectorizer` transforma palavras em representações numéricas.
- A vetorização fica dentro de um `Pipeline`, sendo ajustada somente nos dados de treino para evitar vazamento de informação.

## 6. Modelos comparados

1. **Multinomial Naive Bayes:** algoritmo probabilístico frequentemente usado em classificação de textos.
2. **Regressão Logística:** algoritmo de classificação que estima a classe a partir dos atributos numéricos.

Os dois modelos são avaliados no mesmo conjunto de teste. São registradas acurácia, precisão, recall e F1 para a classe spam. O modelo final é selecionado pelo maior F1 da classe spam. Os resultados reais ficam em `reports/metricas.json`.

## 7. Resultado da comparação

Execute `python train.py` e copie os valores reais de `reports/metricas.json` para a tabela:

| Modelo | Acurácia | Precisão (spam) | Recall (spam) | F1 (spam) |
|---|---:|---:|---:|---:|
| MultinomialNB | preencher | preencher | preencher | preencher |
| LogisticRegression | preencher | preencher | preencher | preencher |

**Modelo escolhido:** o script seleciona automaticamente o que obtiver maior F1 para spam. A justificativa final deve considerar também os falsos positivos, pois uma mensagem legítima marcada como spam pode causar transtornos.

## 8. Aplicação

A interface Streamlit permite digitar uma mensagem, acionar o botão “Fazer previsão” e visualizar a classe estimada e a probabilidade associada à classe prevista. O aplicativo carrega o modelo salvo em `models/model.joblib`; não treina novamente a cada clique.

## 9. Limitações e melhorias

- A base contém mensagens em inglês e é antiga; o desempenho pode ser menor em mensagens atuais em português.
- O modelo pode errar e não compreende contexto, ironia ou intenção como uma pessoa.
- Como melhoria, pode-se treinar com uma base atualizada e em português e testar mensagens de diferentes canais.

## 10. Uso de ferramentas de IA

Ferramentas de IA foram utilizadas como apoio para organização, explicação e revisão do código. O estudante deve executar, compreender, testar e validar os resultados antes da entrega.

## 11. Citação da base

Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.
