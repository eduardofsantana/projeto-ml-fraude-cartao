import streamlit as st
import pandas as pd
import numpy as np
import joblib

# configuracao basica da pagina
st.set_page_config(page_title="Detector de Fraudes - Cartao de Credito", layout="wide")

st.title("SISTEMA DE DETECÇÃO DE FRAUDES EM CARTÕES DE CRÉDITO")
st.write("""
Aplicação desenvolvida para comparar o modelo baseline dos artigos estudados (**Regressao Logistica**) 
com o modelo proposto na etapa de melhoria (**XGBoost**), permitindo ajustar o limiar de decisão (threshold).
""")

# funcao com cache para nao recarregar os modelos toda hora que mexer na tela
@st.cache_resource
def carregar_arquivos():
    scaler = joblib.load('scaler.joblib')
    modelo_lr = joblib.load('modelo_baseline_lr.joblib')
    modelo_xgb = joblib.load('modelo_campeao.joblib')
    amostras = pd.read_csv('amostras_mvp.csv')
    resultados = pd.read_csv('resultados_experimentos.csv')
    return scaler, modelo_lr, modelo_xgb, amostras, resultados

try:
    scaler, modelo_lr, modelo_xgb, df_amostras, df_resultados = carregar_arquivos()
except Exception as erro:
    st.error(f"Erro ao carregar os arquivos do modelo. Verifique se todos os arquivos estao no repositorio. Detalhe: {erro}")
    st.stop()

# menu lateral para escolher o modelo e o threshold
st.sidebar.header("Configurações do modelo")

escolha_modelo = st.sidebar.selectbox(
    "Escolha o modelo para testar:",
    ["XGBoost (Melhoria proposta)", "Regressão Logística (Baseline Artigo 1)"]
)

# slider do threshold (inspirado no ajuste do Artigo 2)
limiar = st.sidebar.slider(
    "Limiar de Decisão (Threshold):",
    min_value=0.05, max_value=0.95, value=0.50, step=0.05
)

modelo_atual = modelo_xgb if "XGBoost" in escolha_modelo else modelo_lr

# dividindo o app em 3 abas simples
aba1, aba2, aba3 = st.tabs([
    "Teste Individual", 
    "Teste em Lote (CSV)", 
    "Tabela de Resultados (Artigo)"
])

# ABA 1: TESTE DE UMA TRANSACAO POR VEZ
with aba1:
    st.subheader("Simular análise de uma transacão do conjunto de teste")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        filtro = st.radio("Filtrar transações de exemplo por tipo:", ["Fraudes Reais (Class = 1)", "Normais Reais (Class = 0)"])
        classe_escolhida = 1 if "Fraudes" in filtro else 0
        
        df_opcoes = df_amostras[df_amostras['Class_Real'] == classe_escolhida]
        id_linha = st.selectbox("Selecione o índice da transação:", df_opcoes.index)
        
        # pegando os dados da linha escolhida
        dados_transacao = df_opcoes.loc[[id_linha]].drop(columns=['Class_Real'])
        rotulo_real = df_opcoes.loc[id_linha, 'Class_Real']
        
        st.info(f"**Valor (Amount):** $ {dados_transacao['Amount'].values[0]:.2f} \n\n"
                f"**Tempo (Time):** {dados_transacao['Time'].values[0]:.0f} s \n\n"
                f"**Classe Real:** {'Fraude (1)' if rotulo_real == 1 else 'Normal (0)'}")

    with col2:
        # aplicando o mesmo scaler usado no treinamento
        dados_norm = pd.DataFrame(scaler.transform(dados_transacao), columns=dados_transacao.columns)
        
        # calculando a probabilidade de ser fraude
        prob = float(modelo_atual.predict_proba(dados_norm)[:, 1][0])
        predicao = 1 if prob >= limiar else 0
        
        st.markdown("### Resultado da Predição")
        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilidade calculada", f"{prob*100:.2f}%")
        m2.metric("Threshold escolhido", f"{limiar*100:.0f}%")
        m3.metric("Classificação do modelo", "FRAUDE" if predicao == 1 else "LEGÍTIMA")
        
        # mostrando se o modelo acertou ou errou
        if predicao == 1 and rotulo_real == 1:
            st.success("Acerto (Verdadeiro Positivo): O modelo identificou a fraude corretamente!")
        elif predicao == 0 and rotulo_real == 0:
            st.success("Acerto (Verdadeiro Negativo): Transação normal aprovada corretamente.")
        elif predicao == 1 and rotulo_real == 0:
            st.warning("Erro (Falso Positivo): A transação era normal, mas o modelo bloqueou como fraude.")
        else:
            st.error("Erro (Falso Negativo): Era uma fraude real, mas o modelo deixou passar!")

        with st.expander("Visualizar colunas V1 a V28 da transacao"):
            st.dataframe(dados_transacao)

# ABA 2: TESTE EM LOTE
with aba2:
    st.subheader("Avaliar várias transações de uma vez")
    st.write("Você pode subir um arquivo CSV ou testar com as 50 amostras salvas do conjunto de teste.")
    
    arquivo_csv = st.file_uploader("Fazer upload de arquivo CSV (opcional)", type=["csv"])
    df_lote = pd.read_csv(arquivo_csv) if arquivo_csv is not None else df_amostras.copy()
    
    # removendo a coluna de resposta se ela existir no arquivo
    col_alvo = 'Class_Real' if 'Class_Real' in df_lote.columns else ('Class' if 'Class' in df_lote.columns else None)
    X_lote = df_lote.drop(columns=[col_alvo]) if col_alvo else df_lote.copy()
    
    # normalizando e prevendo
    X_lote_norm = pd.DataFrame(scaler.transform(X_lote), columns=X_lote.columns)
    probs_lote = modelo_atual.predict_proba(X_lote_norm)[:, 1]
    preds_lote = (probs_lote >= limiar).astype(int)
    
    df_saida = df_lote.copy()
    df_saida['Probabilidade_Fraude'] = np.round(probs_lote, 4)
    df_saida['Predicao_Modelo'] = np.where(preds_lote == 1, 'Fraude (1)', 'Normal (0)')
    
    c1, c2 = st.columns(2)
    c1.metric("Total de transações no lote", len(df_saida))
    c2.metric("Transações marcadas como fraude", int(preds_lote.sum()))
    
    colunas_mostrar = ['Time', 'Amount', 'Probabilidade_Fraude', 'Predicao_Modelo']
    if col_alvo:
        colunas_mostrar.append(col_alvo)
    st.dataframe(df_saida[colunas_mostrar], use_container_width=True)

# ABA 3: TABELA DO COLAB
with aba3:
    st.subheader("Métricas obtidas nos experimentos (Google Colab)")
    st.write("Comparação entre a replicação dos artigos e os modelos propostos:")
    st.dataframe(df_resultados, use_container_width=True)
