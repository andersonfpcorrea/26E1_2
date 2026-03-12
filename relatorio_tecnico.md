# Relatório Técnico: Classificação de Doenças Cardiovasculares

**Autor:** Anderson Corrêa
**Disciplina:** Fundamentos de Machine Learning com Scikit-Learn

---

## Introdução

Este relatório apresenta uma solução completa de machine learning para predição de doenças cardiovasculares (DCV) utilizando dados de saúde de pacientes. Avaliamos cinco configurações de modelos de complexidade crescente, desde um Perceptron linear simples até um ensemble Random Forest, alcançando aproximadamente 73% de acurácia nos dados de teste.

A análise demonstra por que modelos de ML específicos e interpretáveis superam abordagens genéricas para este domínio, e fornece insights para decisões clínicas.

---

## 1. Contexto do Problema

### 1.1 Domínio de Aplicação

Doenças cardiovasculares são uma das principais causas de morte no mundo, responsáveis por aproximadamente 17,9 milhões de mortes anuais (OMS, 2021). A detecção precoce permite intervenções no estilo de vida e tratamentos preventivos que podem reduzir significativamente o risco de mortalidade.

### 1.2 Por que Machine Learning?

A triagem tradicional baseada em regras fixas (ex: "se pressão arterial > 140/90, classificar como alto risco") não consegue capturar:

- **Interações não-lineares** entre fatores de risco
- **Padrões sutis** em combinações de features
- **Variação individual** nos perfis de risco

Machine learning pode aprender essas relações complexas a partir dos dados, fornecendo estratificação de risco mais refinada.

### 1.3 Desafios do Domínio

1. **Qualidade dos dados:** Medições de pressão arterial contêm valores impossíveis (negativos, extremamente altos)
2. **Interpretação das features:** Clínicos exigem predições explicáveis
3. **Custos de erros:** Falsos negativos (doença não detectada) são mais custosos que falsos positivos
4. **Balanceamento de classes:** O dataset escolhido é bem balanceado (~50/50)

---

## 2. Justificativas para Seleção de Modelos

### 2.1 Perceptron (Baseline)

**Por quê:** Estabelece um limite inferior com máxima interpretabilidade. A equação linear ŷ = w₀ + w₁x₁ + ... + wₙxₙ mostra diretamente a contribuição de cada feature.

**Limitações observadas:** Desempenho fraco indica que a fronteira de decisão é não-linear.

### 2.2 Árvore de Decisão (Não-linear)

**Por quê:** Captura interações entre features através de particionamento recursivo. Regras como "se pressão_arterial > X E idade > Y" são clinicamente significativas.

**Parâmetros escolhidos:** Inicialmente padrão, depois ajustados via GridSearch.

### 2.3 Random Forest (Ensemble)

**Por quê:** Reduz variância através de bagging enquanto mantém interpretabilidade via scores de importância de features.

**Parâmetros ajustados via RandomizedSearchCV:** n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features.

---

## 3. Interpretação dos Parâmetros Aprendidos

### 3.1 Coeficientes do Perceptron

O Perceptron aprende pesos que definem um hiperplano no espaço de features:

| Feature                   | Coeficiente | Interpretação                                  |
| ------------------------- | ----------- | ---------------------------------------------- |
| alco (álcool)             | +4.22       | Consumo de álcool aumenta risco                |
| ap_hi (pressão sistólica) | +3.02       | Maior pressão aumenta risco                    |
| age_years (idade)         | +2.15       | Pacientes mais velhos têm maior risco          |
| height (altura)           | +1.96       | Altura influencia positivamente                |
| ap_lo (pressão diastólica)| +1.85       | Pressão diastólica elevada aumenta risco       |
| bmi (IMC)                 | +1.83       | Maior IMC aumenta risco                        |
| weight (peso)             | +0.76       | Maior peso aumenta risco                       |
| active (atividade física) | +0.49       | Sinal positivo (contra-intuitivo)              |
| gluc (glicose)            | -0.39       | Sinal negativo (contra-intuitivo)              |
| cholesterol (colesterol)  | -2.01       | Sinal negativo (contra-intuitivo)              |
| smoke (tabagismo)         | -3.85       | Sinal negativo (contra-intuitivo)              |

_Nota: Sinais contra-intuitivos em algumas features (cholesterol, smoke negativos) refletem as limitações do Perceptron linear neste problema não-linear._

### 3.2 Regras da Árvore de Decisão

Principais regras de decisão aprendidas:

1. **Divisão raiz:** ap_hi ≤ 129.5 mm Hg
   - Se sim → ramo de menor risco
   - Se não → ramo de maior risco

2. **Segundo nível (pressão normal):** age_years ≤ 54.6
   - Pacientes mais jovens com pressão normal → menor risco
   - Pacientes mais velhos com pressão normal → risco moderado (cholesterol passa a ser decisivo)

3. **Segundo nível (pressão elevada):** ap_hi ≤ 138.5
   - Pressão moderadamente elevada (130-138) → risco intermediário
   - Pressão alta (>138) → alto risco

### 3.3 Importância de Features no Random Forest

Importância de features baseada na diminuição média de impureza:

1. **ap_hi (pressão sistólica):** 43,9%
2. **ap_lo (pressão diastólica):** 19,0%
3. **age_years (idade):** 13,9%
4. **cholesterol (colesterol):** 9,1%
5. **bmi (IMC):** 5,3%

Estes resultados estão alinhados com o conhecimento clínico sobre fatores de risco para DCV.

---

## 4. Discussão Crítica dos Resultados

### 4.1 Comparação de Desempenho dos Modelos

| Métrica  | Perceptron | AD (padrão) | AD (ajustada) | RF (padrão) | RF (otimizado) |
| -------- | ---------- | ----------- | ------------- | ----------- | -------------- |
| Acurácia | 63,4%      | 62,8%       | 72,8%         | 70,9%       | 73,3%          |
| Precisão | 61,7%      | 62,4%       | 73,9%         | 71,1%       | 75,6%          |
| Recall   | 68,6%      | 62,8%       | 69,4%         | 69,5%       | 67,8%          |
| F1-Score | 64,9%      | 62,6%       | 71,6%         | 70,3%       | 71,5%          |

### 4.2 Principais Observações

1. **Perceptron apresenta underfitting:** A taxa de erro de ~37% com modelo linear confirma que existem relações não-lineares.

2. **Árvore de Decisão sem regularização apresenta overfitting severo:** Acurácia de treino de 99,98% vs 62,8% no teste (gap de 37 pontos percentuais). Após regularização via GridSearchCV (max_depth=7, min_samples_leaf=2), a acurácia de teste sobe para 72,8%.

3. **Árvore de Decisão otimizada e Random Forest otimizado têm desempenho muito próximo:** F1-Score de 71,6% vs 71,5%, respectivamente. O Random Forest otimizado tem maior precisão (75,6%) enquanto a Árvore otimizada tem maior recall (69,4%).

4. **Recall vs Precisão:** Em triagem médica, não detectar uma doença (falso negativo) é pior que um falso alarme. O Perceptron obtém o maior recall (68,6%), mas com baixa precisão. Os modelos otimizados alcançam ~68-69% de recall com precisão significativamente melhor.

### 4.3 Estabilidade da Validação Cruzada

Resultados da validação cruzada em 5 folds (F1-Score):

**Árvore de Decisão Otimizada (GridSearchCV):**

| Fold               | F1-Score |
| ------------------ | -------- |
| 1                  | 70,8%    |
| 2                  | 72,0%    |
| 3                  | 70,6%    |
| 4                  | 71,4%    |
| 5                  | 71,9%    |
| **Média ± Desvio** | **71,3% ± 0,6%** |

**Random Forest (padrão, 100 estimadores):**

| Fold               | F1-Score |
| ------------------ | -------- |
| 1                  | 70,4%    |
| 2                  | 70,6%    |
| 3                  | 70,8%    |
| 4                  | 70,4%    |
| 5                  | 70,8%    |
| **Média ± Desvio** | **70,6% ± 0,2%** |

O baixo desvio padrão em ambos os modelos indica desempenho robusto entre as diferentes divisões dos dados. O Random Forest apresenta variação ainda menor (0,2% vs 0,6%).

---

## 5. Viabilidade no Mundo Real

### 5.1 Pontos Fortes

1. **Interpretável:** Importância das features alinhada com conhecimento clínico
2. **Inferência rápida:** Predições em milissegundos
3. **Sem dependências externas:** Usa apenas scikit-learn padrão
4. **Tolerante a diferentes escalas:** Modelos baseados em árvore não requerem normalização das features

### 5.2 Limitações

1. **~73% de acurácia é insuficiente para diagnóstico isolado:** Deve ser usado como ferramenta de triagem, não como diagnóstico definitivo
2. **Limitações do dataset:** Fatores de estilo de vida auto-reportados podem ser imprecisos
3. **Viés populacional:** Modelo treinado em demografia específica; pode não generalizar para todas as populações
4. **Sem dados temporais:** Predições instantâneas não capturam progressão da doença

### 5.3 Considerações de Implantação

- **Integração:** Endpoint de API retornando score de risco + principais fatores contribuintes
- **Monitoramento:** Acompanhar distribuição de predições ao longo do tempo para detecção de drift
- **Supervisão humana:** Todas as predições de alto risco revisadas por clínico

---

## 6. Melhorias Possíveis

### 6.1 Melhorias nos Dados

- Incluir histórico médico (condições prévias, medicamentos)
- Adicionar resultados laboratoriais (tipos específicos de colesterol, tendências de glicose)
- Incorporar dados de séries temporais para acompanhamento da progressão

### 6.2 Melhorias nos Modelos

- **Gradient Boosting (XGBoost, LightGBM):** Frequentemente supera Random Forest
- **Probabilidades calibradas:** Para pontuação de risco ao invés de classificação binária
- **Ensemble de modelos diversos:** Combinar pontos fortes de diferentes arquiteturas

### 6.3 Melhorias Operacionais

- **Otimização de threshold:** Ajustar limiar de decisão baseado em análise de custo clínico
- **Análise de subgrupos:** Verificar se o modelo funciona bem entre grupos de idade/gênero
- **Teste A/B:** Comparar recomendações do modelo com resultados de cuidado padrão

---

## 7. Conclusão

Este projeto demonstra o pipeline completo de machine learning para um problema de classificação do mundo real. Principais aprendizados:

1. **Baselines simples são importantes:** O Perceptron estabeleceu que modelos lineares são insuficientes, motivando abordagens não-lineares.

2. **Regularização é essencial:** Árvores de Decisão sem restrições apresentam overfitting; ajuste de hiperparâmetros melhora generalização.

3. **Ensembles e regularização fornecem desempenho robusto:** Tanto a Árvore de Decisão otimizada quanto o Random Forest alcançam ~73% de acurácia e ~71,5% de F1-Score.

4. **Conhecimento de domínio guia ML:** Entender que falsos negativos são custosos informou nossa seleção de métricas.

5. **ML complementa, não substitui, expertise:** O modelo é uma ferramenta de triagem, não um substituto para diagnóstico.
