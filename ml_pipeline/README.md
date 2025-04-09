# Treinamento de Modelo para Detecção de Pneumonia

> **NOTA IMPORTANTE**: Este modelo encontra-se em fase experimental e apresenta limitações significativas na detecção precisa de pneumonia. Identificamos falhas que estão sendo analisadas e corrigidas. Atualmente, este modelo serve apenas como base para estabelecer o fluxo do projeto. Novas versões aprimoradas, implementações com agentes inteligentes e modelos com maior acurácia serão disponibilizados em breve.

## Visão Geral

O modelo é uma rede neural convolucional (CNN) treinada para classificar imagens de raio-X do tórax em duas categorias:
- `NORMAL`: Pulmões saudáveis
- `PNEUMONIA`: Pulmões com sinais de pneumonia

## Dataset

O treinamento utiliza o conjunto de dados "Chest X-Ray Images (Pneumonia)" que contém:
- Imagens de raio-X do tórax divididas em pastas de treino, teste e validação
- Duas classes: PNEUMONIA e NORMAL
- As imagens são processadas em escala de cinza e redimensionadas para 150x150 pixels

## Estrutura do Notebook

O notebook `training.ipynb` contém as seguintes etapas:

1. **Importação de bibliotecas**: TensorFlow/Keras, OpenCV, NumPy, Matplotlib, etc.
2. **Carregamento e pré-processamento de dados**: Leitura das imagens, redimensionamento e normalização
3. **Visualização de dados**: Análise da distribuição das classes e visualização de amostras
4. **Aumento de dados (Data Augmentation)**: Aplicação de transformações para melhorar a generalização do modelo
5. **Arquitetura do modelo**: Definição de uma CNN com várias camadas convolucionais e técnicas de regularização
6. **Treinamento do modelo**: Ajuste do modelo usando callbacks para redução da taxa de aprendizado
7. **Avaliação do modelo**: Métricas de desempenho e visualização de resultados
8. **Salvamento do modelo**: Exportação para o formato .h5
9. **Teste de inferência**: Teste do modelo em uma imagem individual

## Arquitetura do Modelo

O modelo utiliza a seguinte arquitetura:
- 5 blocos convolucionais com BatchNormalization e MaxPooling
- Dropout para regularização (taxas de 0.1 a 0.2)
- Camada densa final com ativação sigmoid para classificação binária
- Otimizador: RMSprop
- Função de perda: Binary Crossentropy

## Requisitos para Execução

- Python 3.6+
- TensorFlow 2.x
- Keras
- OpenCV (cv2)
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

## Como Executar

1. Baixe o conjunto de dados "Chest X-Ray Images (Pneumonia)" 
2. Ajuste os caminhos no notebook para apontar para a localização correta dos dados
3. Execute o notebook célula por célula para treinar o modelo
4. O modelo treinado será salvo como `pneumonia_detection_model.h5`

## Resultados

Após o treinamento, o modelo é avaliado quanto a:
- Acurácia no conjunto de teste
- Matriz de confusão
- Relatório de classificação (precisão, recall, F1-score)
- Visualização de predições corretas e incorretas

## Uso do Modelo Treinado

O modelo treinado (`pneumonia_detection_model.h5`) pode ser carregado e utilizado para fazer predições em novas imagens de raio-X, como demonstrado na última parte do notebook.

## Próximos Passos

- Correção das falhas identificadas na detecção de pneumonia
- Implementação de agentes inteligentes para pré-processamento de imagens
- Treinamento com datasets mais abrangentes e balanceados
- Ajuste fino da arquitetura do modelo
- Experimentação com diferentes técnicas de aumento de dados
- Validação cruzada para melhor estimativa de desempenho
- Implementação de técnicas de explicabilidade (como mapas de calor)
- Lançamento de um modelo de produção com maior confiabilidade

## Referências

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)