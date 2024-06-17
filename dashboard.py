import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.title("Análise de Anúncios Sociais")
st.write("Iniciando o carregamento dos dados...")

# Lendo o arquivo CSV em um DataFrame do pandas
df = pd.read_csv('social_ads.csv')

# Remover registros duplicados
df.drop_duplicates(inplace=True)
st.write("Dados carregados e duplicatas removidas.")

# Exibir estatísticas descritivas
st.subheader("Estatísticas descritivas")
st.write(df.describe())

# Exibir as primeiras linhas do DataFrame
st.subheader("Primeiras linhas do DataFrame")
st.write(df.head())

# Normalizar as variáveis numéricas 'Age' e 'EstimatedSalary'
scaler = StandardScaler()
df[['Age', 'EstimatedSalary']] = scaler.fit_transform(df[['Age', 'EstimatedSalary']])
st.write("Dados normalizados.")

# Criar faixas etárias e salariais
df['Faixa_Etaria'] = pd.cut(df['Age'], bins=[-3, -1, 0, 1, 2, 3], labels=['18-24', '25-34', '35-44', '45-54', '55-64'])
df['Faixa_Salarial'] = pd.cut(df['EstimatedSalary'], bins=[-3, -1, 0, 1, 2, 3], labels=['15k-40k', '40k-70k', '70k-100k', '100k-130k', '130k-160k'])
st.write("Faixas etárias e salariais criadas.")

# Distribuição de Idade
st.header("Distribuição de Idade")
fig, ax = plt.subplots()
sns.histplot(df['Age'], kde=True, ax=ax)
plt.title('Distribuição de Idade')
plt.xlabel('Idade')
plt.ylabel('Frequência')
st.pyplot(fig)

# Distribuição de Salário Estimado
st.header("Distribuição de Salário Estimado")
fig, ax = plt.subplots()
sns.histplot(df['EstimatedSalary'], kde=True, ax=ax)
plt.title('Distribuição de Salário Estimado')
plt.xlabel('Salário Estimado')
plt.ylabel('Frequência')
st.pyplot(fig)

# Distribuição de Compra Realizada
st.header("Distribuição de Compra Realizada")
fig, ax = plt.subplots()
sns.countplot(x='Purchased', data=df, ax=ax)
plt.title('Distribuição de Compra Realizada')
plt.xlabel('Compra Realizada (0 ou 1)')
plt.ylabel('Contagem')
st.pyplot(fig)

# Matriz de Correlação
st.header("Matriz de Correlação")
fig, ax = plt.subplots()
# Selecionar apenas colunas numéricas para a matriz de correlação
corr = df[['Age', 'EstimatedSalary', 'Purchased']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
plt.title('Matriz de Correlação')
st.pyplot(fig)

# Dividir o DataFrame em variáveis independentes (X) e a variável alvo (y)
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Dividir os dados em conjuntos de treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de árvore de decisão
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste e avaliar o desempenho
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(X_test)
st.subheader("Matriz de Confusão")
st.write(confusion_matrix(y_test, y_pred))
st.subheader("Relatório de Classificação")
st.write(classification_report(y_test, y_pred))

# Proporção de Compras Realizadas
compras = df['Purchased'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(compras, labels=['Não Comprou', 'Comprou'], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'orange'])
plt.title('Proporção de Compras Realizadas')
st.pyplot(plt)

# Proporção de Faixas Etárias entre os Compradores
compradores_por_faixa_etaria = df[df['Purchased'] == 1]['Faixa_Etaria'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(compradores_por_faixa_etaria, labels=compradores_por_faixa_etaria.index, autopct='%1.1f%%', startangle=90)
plt.title('Proporção de Faixas Etárias entre os Compradores')
st.pyplot(plt)

# Proporção de Faixas Salariais entre os Compradores
compradores_por_faixa_salarial = df[df['Purchased'] == 1]['Faixa_Salarial'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(compradores_por_faixa_salarial, labels=compradores_por_faixa_salarial.index, autopct='%1.1f%%', startangle=90)
plt.title('Proporção de Faixas Salariais entre os Compradores')
st.pyplot(plt)

st.write("Análise concluída.")
