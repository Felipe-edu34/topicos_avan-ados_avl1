# 🚀 Classificação de Obesidade: Pipeline ML & MLOps com PyTorch

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)

Este repositório contém o projeto final da **Primeira Avaliação** da disciplina de *Tópicos Avançados em Inteligência Artificial A*, ministrada pelo **Prof. Dr. Thommas K. S. Flores** no Centro de Ensino Superior do Seridó (CERES/UFRN).

O objetivo deste trabalho é demonstrar a construção de um pipeline ponta a ponta de aprendizado de máquina, justificando tecnicamente as escolhas de pré-processamento, engenharia de atributos, modelagem em Deep Learning (MLP) e a implementação de práticas de MLOps.

---

## 📋 Sumário
- [1. Análise Crítica e Seleção de Variáveis](#1-análise-crítica-e-seleção-de-variáveis)
- [2. Modelagem e Justificativa de Hiperparâmetros](#2-modelagem-e-justificativa-de-hiperparâmetros)
- [3. MLOps e Rastreamento de Experimentos](#3-mlops-e-rastreamento-de-experimentos)
- [4. Estrutura do Repositório](#4-estrutura-do-repositório)
- [5. Como Executar](#5-como-executar)
- [6. Autor](#6-autor)

---

## 🧠 1. Análise Crítica e Seleção de Variáveis

O pré-processamento foi focado em garantir a qualidade dos dados e a eficiência do modelo, evitando ruídos e redundâncias.
* **Remoção de Ruído Estrutural:** A coluna `ID` foi removida imediatamente para evitar que a rede neural tentasse "decorar" as amostras (risco de *overfitting*).
* **Transformação de Dados:** Variáveis categóricas (`Gender` e `Label`) foram convertidas para tensores numéricos inteiros utilizando `LabelEncoder`, requisito obrigatório para o processamento no PyTorch.

### 🔬 Rigor na Seleção de Variáveis (Feature Selection)
Para evitar o problema da multicolinearidade e reduzir a dimensionalidade, foram aplicados múltiplos métodos de análise:
1. **Análise de Correlação de Pearson:** O mapa de calor revelou uma altíssima redundância entre Peso, Altura e IMC (BMI). Como o IMC é um índice derivado do peso e altura, introduzir as três variáveis na rede não traria ganhos de informação, apenas aumento de custo computacional.
2. **Significância Estatística (ANOVA F-Score):** O teste F provou matematicamente que o IMC possui a relação mais forte de separação entre as classes alvo de obesidade.
3. **Importância de Variáveis (Random Forest):** O algoritmo de árvores confirmou que o IMC concentra quase a totalidade da importância preditiva.
**Decisão:** As variáveis `Weight` e `Height` foram descartadas. O espaço de entrada foi reduzido para apenas 3 variáveis (`Age`, `Gender` e `BMI`), focando no *signal-to-noise ratio*.

---

## ⚙️ 2. Modelagem e Justificativa de Hiperparâmetros

O modelo base escolhido foi uma Rede Neural Multilayer Perceptron (MLP) construída no **PyTorch**. Os dados de entrada foram padronizados (`StandardScaler` - Z-score) para garantir que a diferença de escalas (ex: Idade vs Gênero) não afetasse a convergência dos gradientes.

**Justificativa da Arquitetura:**
* **Camadas Ocultas (16 -> 8):** Como o espaço de features foi reduzido para 3, uma rede profunda e complexa geraria *overfitting* rápido. Optou-se por uma estrutura em funil, suficiente para capturar não-linearidades e suavizar a extração até as 4 classes de saída.
* **Função de Ativação:** `ReLU` (Rectified Linear Unit) foi utilizada nas camadas ocultas por sua eficiência contra o desaparecimento do gradiente.
* **Otimizador e Learning Rate:** Foi empregado o otimizador `Adam` com taxa de aprendizagem de `0.001`. O Adam combina os benefícios do RMSProp e AdaGrad, adaptando a taxa dinamicamente, enquanto o valor de 0.001 garantiu uma descida estável no gradiente de erro sem pular os mínimos locais.
* **Batch Size:** Configurado para `16`. Batches menores introduzem ruído estocástico saudável, ajudando a rede a generalizar melhor padrões fora do treino.

---

## 📈 3. MLOps e Rastreamento de Experimentos

Para garantir a reprodutibilidade e o monitoramento em tempo real (conforme as diretrizes da disciplina), a ferramenta **Weights & Biases (W&B)** foi integrada ao pipeline.

* **Tracking:** Registro automático da perda de treinamento (`train_loss`) e exatidão de validação (`val_accuracy`) a cada época.
* **Versioning:** Os hiperparâmetros (épocas, batch, neurônios) foram versionados no dicionário de configurações (`wandb.config`).
* **Model Registry:** O artefato final, contendo os pesos treinados (`mlp_model.pth`), foi salvo localmente e sincronizado com a nuvem.

🔗 **[CLIQUE AQUI PARA ACESSAR O DASHBOARD PÚBLICO DO W&B COM OS GRÁFICOS DO TREINAMENTO](COLE_AQUI_O_LINK_DO_SEU_PROJETO_NO_W&B)**

---

## 📂 4. Estrutura do Repositório

```text
├── data/
│   └── Obesity Classification.csv    # Dataset original (Fonte: Kaggle)
├── README.md                         # Este documento de análise e diretrizes
├── requirements.txt                  # Arquivo para instalação das dependências
├── projeto_obesidade.ipynb           # Notebook principal contendo todo o pipeline executável
└── mlp_model.pth                     # Artefato gerado com os pesos finais do modelo

```

## 🚀 5. Como Executar

1. clone o repositorio
git clone [https://github.com/SEU_USUARIO/projeto-obesidade-ia.git](https://github.com/Felipe-edu34/topicos_avan-ados_avl1.git)
cd projeto-obesidade-ia

2. Crie o ambiente virtual e instale as dependências:
```text
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
``` 

3. Configure o MLOps:
faça login na wandb

```text wandb login ```
(Cole a sua API Key quando o terminal solicitar. Por questões de segurança no Linux, os caracteres não aparecerão no ecrã enquanto cola, basta premir Enter).

