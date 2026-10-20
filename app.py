import streamlit as st
import pandas as pd
import numpy as np
import joblib

# configuracao basica da pagina
st.set_page_config(page_title="Detector de Fraudes - Cartão de Crédito", layout="wide")

st.title("🛡️ Sistema de Detecção de Fraudes em Cartões de Crédito")
st.markdown("""
Aplicativo desenvolvido para avaliar o desempenho de algoritmos de Inteligência Artificial na triagem de transações,
comparando as configurações de referência com o modelo **XGBoost (Proposto)** sob protocolo de validação independente.
""")

# funcao com cache para nao recarregar os modelos e dados toda hora
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
    st.error(f"Erro ao carregar os artefatos do modelo. Certifique-se de que os arquivos .joblib e .csv estão no repositório. Detalhe: {erro}")
    st.stop()

# menu lateral de controle do modelo e do limiar
st.sidebar.header("⚙️ Controle de Inferência")

escolha_modelo = st.sidebar.selectbox(
    "Motor de Classificação:",
    ["XGBoost (Proposta Ensemble)", "Regressão Logística (Baseline)"]
)

# slider de threshold variando do mais rigoroso ao mais permissivo
limiar = st.sidebar.slider(
    "Limiar de Decisão (Threshold):",
    min_value=0.01, max_value=0.99, value=0.50, step=0.01,
    help="Abaixar o limiar aumenta a captura de fraudes, mas pode gerar falsos positivos."
)

# EXPLICAÇÃO DETALHADA
with st.sidebar.expander("📖 Entenda como este sistema funciona", expanded=False):
    st.markdown("""
    **O que é este MVP (Produto Mínimo Viável)?**
    Este é um protótipo funcional de um sistema de segurança financeira. Ele usa Inteligência Artificial para ler os dados de uma transação de cartão de crédito no momento em que ela acontece e tentar adivinhar se é o dono do cartão comprando ou um golpista.
    
    **Como ele decide? (O Limiar / Threshold)**
    A IA não responde apenas "sim" ou "não". Ela dá uma *probabilidade* (de 0% a 100%) da compra ser fraude. 
    A barrinha acima (**Limiar de Decisão**) é a "régua" de corte que você define para a máquina:
    * Se você colocar o limiar em **80%**, o sistema só vai bloquear a transação se tiver muita certeza. Isso evita bloquear o cartão de um cliente real, mas pode deixar um golpista inteligente passar.
    * Se você colocar o limiar em **20%**, o sistema fica super rigoroso. Ele vai bloquear quase todos os golpistas, mas também vai acabar bloqueando as compras normais de clientes inocentes (o que chamamos de *Falso Positivo*).
    
    **Como testar?**
    1. Vá na aba "Simulador Individual".
    2. Escolha se quer testar uma "Fraude Real" ou uma "Operação Legítima".
    3. Alterne o modelo e mude o Limiar na barrinha aqui do lado para ver a IA mudando de ideia e errando ou acertando ao vivo!
    """)

modelo_atual = modelo_xgb if "XGBoost" in escolha_modelo else modelo_lr

# dividindo a interface
aba1, aba2, aba3 = st.tabs([
    "🔍 Simulador Individual", 
    "📂 Auditoria em Lote (CSV)", 
    "📊 Métricas do Experimento"
])

# ABA 1: TESTE DE UMA TRANSACAO POR VEZ
with aba1:
    st.subheader("Inspeção Individual de Transações da Base de Teste")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        filtro = st.radio("Filtrar amostras de validação por classe:", ["Fraudes Confirmadas (Class = 1)", "Operações Legítimas (Class = 0)"])
        classe_escolhida = 1 if "Fraudes" in filtro else 0
        
        df_opcoes = df_amostras[df_amostras['Class_Real'] == classe_escolhida]
        id_linha = st.selectbox("Selecione o ID da transação (Index):", df_opcoes.index)
        
        # extraindo os dados da transacao escolhida
        dados_transacao = df_opcoes.loc[[id_linha]].drop(columns=['Class_Real'])
        rotulo_real = df_opcoes.loc[id_linha, 'Class_Real']
        
        st.info(f"**Valor (Amount):** $ {dados_transacao['Amount'].values[0]:.2f} \n\n"
                f"**Tempo (Time):** {dados_transacao['Time'].values[0]:.0f} s \n\n"
                f"**Rótulo Verdadeiro:** {'🚨 FRAUDE (1)' if rotulo_real == 1 else '✅ LEGÍTIMA (0)'}")

    with col2:
        # aplicando padronizacao restrita do treino
        dados_norm = pd.DataFrame(scaler.transform(dados_transacao), columns=dados_transacao.columns)
        
        # calculando probabilidade ou funcao de decisao
        if hasattr(modelo_atual, "predict_proba"):
            prob = float(modelo_atual.predict_proba(dados_norm)[:, 1][0])
        else:
            prob = float(modelo_atual.decision_function(dados_norm)[0])
            
        predicao = 1 if prob >= limiar else 0
        
        st.markdown("### Resposta do Classificador")
        m1, m2, m3 = st.columns(3)
        m1.metric("Escore / Probabilidade", f"{prob*100:.2f}%" if prob <= 1.0 else f"{prob:.2f}")
        m2.metric("Limiar Aplicado", f"{limiar*100:.0f}%" if limiar <= 1.0 else f"{limiar:.2f}")
        m3.metric("Ação Sugerida", "BLOQUEAR 🚨" if predicao == 1 else "APROVAR ✅")
        
        # exibindo interpretacao da matriz de confusao
        if predicao == 1 and rotulo_real == 1:
            st.success("🎯 **Verdadeiro Positivo (VP):** O modelo identificou a fraude com sucesso.")
        elif predicao == 0 and rotulo_real == 0:
            st.success("🎯 **Verdadeiro Negativo (VN):** Operação aprovada corretamente.")
        elif predicao == 1 and rotulo_real == 0:
            st.warning("⚠️ **Falso Positivo (FP):** Alarme falso! Operação legítima bloqueada indevidamente.")
        else:
            st.error("❌ **Falso Negativo (FN):** Falha na segurança! A fraude não foi detectada pelo modelo.")

        with st.expander("Ver componentes anonimizadas (V1 a V28)"):
            st.dataframe(dados_transacao)

# ABA 2: TESTE EM LOTE
with aba2:
    st.subheader("Processamento Paralelo de Lotes")
    st.write("Faça upload de um CSV com novas transações ou utilize o lote de teste embarcado na aplicação.")
    
    arquivo_csv = st.file_uploader("Upload de lote CSV (opcional)", type=["csv"])
    df_lote = pd.read_csv(arquivo_csv) if arquivo_csv is not None else df_amostras.copy()
    
    col_alvo = 'Class_Real' if 'Class_Real' in df_lote.columns else ('Class' if 'Class' in df_lote.columns else None)
    X_lote = df_lote.drop(columns=[col_alvo]) if col_alvo else df_lote.copy()
    
    # normalizando o lote inteiro de uma vez
    X_lote_norm = pd.DataFrame(scaler.transform(X_lote), columns=X_lote.columns)
    
    if hasattr(modelo_atual, "predict_proba"):
        probs_lote = modelo_atual.predict_proba(X_lote_norm)[:, 1]
    else:
        probs_lote = modelo_atual.decision_function(X_lote_norm)
        
    preds_lote = (probs_lote >= limiar).astype(int)
    
    df_saida = df_lote.copy()
    df_saida['Escore_Modelo'] = np.round(probs_lote, 4)
    df_saida['Status_Avaliacao'] = np.where(preds_lote == 1, '🚨 Bloqueado', '✅ Aprovado')
    
    c1, c2 = st.columns(2)
    c1.metric("Volume Analisado (Registros)", len(df_saida))
    c2.metric("Alertas de Fraude Disparados", int(preds_lote.sum()))
    
    colunas_mostrar = ['Time', 'Amount', 'Escore_Modelo', 'Status_Avaliacao']
    if col_alvo:
        colunas_mostrar.append(col_alvo)
    st.dataframe(df_saida[colunas_mostrar], use_container_width=True)

# ABA 3: TABELA DO ARTIGO
with aba3:
    st.subheader("Métricas da Avaliação Independente (Protocolo B e A)")
    st.write("Tabela exportada dos experimentos, refletindo a precisão exata obtida na partição de teste.")
    st.dataframe(df_resultados, use_container_width=True)
