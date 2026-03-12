# Projeto de Classificação de Doenças Cardiovasculares

**Aluno:** Anderson Corrêa

**Disciplina:** Fundamentos de Machine Learning com Scikit-Learn [26E1_2]

## Descrição

Este projeto implementa um pipeline completo de machine learning para **predição de doenças cardiovasculares** utilizando a biblioteca scikit-learn. O objetivo é classificar pacientes como tendo ou não doença cardiovascular com base em métricas de saúde e fatores de estilo de vida.

## Dataset

**Fonte:** [Kaggle - Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

- **Observações:** 70.000 pacientes
- **Features:** 11 atributos de saúde e estilo de vida
- **Variável-alvo:** Classificação binária (doença presente/ausente)
- **Balanceamento de classes:** ~50/50 (bem balanceado)

### Features

| Feature     | Descrição                            |
| ----------- | ------------------------------------ |
| age         | Idade em dias (convertida para anos) |
| gender      | 1=Feminino, 2=Masculino              |
| height      | Altura em cm                         |
| weight      | Peso em kg                           |
| ap_hi       | Pressão arterial sistólica           |
| ap_lo       | Pressão arterial diastólica          |
| cholesterol | 1=Normal, 2=Acima, 3=Muito acima     |
| gluc        | Nível de glicose (mesma escala)      |
| smoke       | Tabagismo (0/1)                      |
| alco        | Consumo de álcool (0/1)              |
| active      | Atividade física (0/1)               |

## Modelos Implementados

| Modelo                           | Propósito           | Principais Achados                                        |
| -------------------------------- | ------------------- | --------------------------------------------------------- |
| **Perceptron**                   | Baseline linear     | Estabelece limite inferior, coeficientes interpretáveis   |
| **Árvore de Decisão**            | Regras não-lineares | Captura interações entre features, propenso a overfitting |
| **Árvore de Decisão (ajustada)** | Regularizada        | Melhor generalização via otimização de hiperparâmetros    |
| **Random Forest**                | Ensemble            | Desempenho similar à AD ajustada, maior estabilidade      |

## Estrutura do Projeto

```
26E1_2/
├── projeto_26E1_2.ipynb      # Notebook principal de análise
├── cardio_train.csv          # Dataset (70k registros)
├── data_dictionary.md        # Descrição das features
├── relatorio_tecnico.md      # Relatório técnico
├── requirements.txt          # Dependências Python
├── README.md                 # Este arquivo
└── .gitignore
```

## Instruções de Configuração

### 1. Criar Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Executar Jupyter Notebook

```bash
jupyter notebook projeto_26E1_2.ipynb
```

### 4. Executar Todas as Células

No Jupyter: `Kernel` → `Restart & Run All`

## Conclusões

1. **Modelos lineares são insuficientes** para este problema - o baseline Perceptron mostra claro underfitting
2. **Árvores de Decisão capturam não-linearidades** mas requerem regularização para prevenir overfitting
3. **Modelos otimizados (AD ajustada e Random Forest) alcançam ~73% de acurácia** com boa generalização
4. **Análise de importância de features** revela pressão arterial e idade como principais preditores
5. **O modelo é implantável** mas requer consideração cuidadosa dos custos de falsos negativos no contexto médico
