import streamlit as st
from adapters.importador import carregar_clientes_do_excel
from core.sistema import Sistema
from components.utils import (
    carregar_e_preprocessar_dados, 
    calcular_analise_icp,
    get_variaveis_default,
    formatar_valor
)
from services.ai_insights import gerar_insights_ia, gerar_acao_sugerida_para_insight
import locale
import pandas as pd
from typing import Dict, List

# Configurar locale para português do Brasil
try:
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
except locale.Error:
    locale.setlocale(locale.LC_ALL, '')  # Usa o padrão do sistema

def formatar_numero_br(valor, casas_decimais=2):
    """Formata número para o padrão brasileiro."""
    if isinstance(valor, (int, float)):
        return f"{valor:,.{casas_decimais}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return valor

@st.cache_data(ttl=3600)
def _formatar_perfil_capitao(perfil: dict) -> dict:
    """Formata os valores do perfil Capitão América de forma otimizada."""
    perfil_formatado = {}
    
    for campo, valor in perfil.items():
        if isinstance(valor, dict):
            if 'media' in valor:
                # Formatação para métricas numéricas
                if campo in ['ticket_medio', 'ltv']:
                    # Valores monetários
                    perfil_formatado[campo] = {
                        'media': f"R$ {formatar_numero_br(valor['media'])}",
                        'mediana': f"R$ {formatar_numero_br(valor['mediana'])}",
                        'min': f"R$ {formatar_numero_br(valor['min'])}",
                        'max': f"R$ {formatar_numero_br(valor['max'])}"
                    }
                else:
                    # Valores numéricos não monetários
                    perfil_formatado[campo] = {
                        'media': formatar_numero_br(valor['media'], 1),
                        'mediana': formatar_numero_br(valor['mediana'], 1),
                        'min': formatar_numero_br(valor['min'], 1),
                        'max': formatar_numero_br(valor['max'], 1)
                    }
            elif 'moda' in valor:
                # Formatação para métricas categóricas
                distribuicao_formatada = {
                    k: f"{formatar_numero_br(v, 1)}%" for k, v in valor['distribuicao'].items()
                }
                perfil_formatado[campo] = {
                    'moda': valor['moda'],
                    'distribuicao': distribuicao_formatada
                }
        else:
            perfil_formatado[campo] = valor
            
    return perfil_formatado

def _exibir_metricas_financeiras(perfil: dict):
    """Exibe as métricas financeiras em colunas."""
    col1, col2 = st.columns(2)
    with col1:
        ticket_medio = perfil.get('ticket_medio', {})
        if isinstance(ticket_medio, dict) and 'media' in ticket_medio:
            st.metric("Ticket Médio", ticket_medio['media'])
        else:
            st.metric("Ticket Médio", "N/A")
    with col2:
        ltv = perfil.get('ltv', {})
        if isinstance(ltv, dict) and 'media' in ltv:
            st.metric("LTV", ltv['media'])
        else:
            st.metric("LTV", "N/A")

@st.cache_data(ttl=3600)
def _processar_correlacoes(correlacoes: dict) -> tuple:
    """Processa correlações para exibição otimizada."""
    insights = []
    
    # Processando correlações por categoria
    for categoria, dados in correlacoes['categorias'].items():
        # Análise de Ticket Médio
        ticket = dados['ticket_medio']
        insights.append({
            'tipo': 'ticket_medio',
            'variavel': categoria,
            'insight': f"**{categoria.title()}**: {ticket['melhor_categoria']} tem ticket médio {ticket['diferenca_percentual']:.1f}% maior que {ticket['pior_categoria']}"
        })
        
        # Análise de LTV
        ltv = dados['ltv']
        insights.append({
            'tipo': 'ltv',
            'variavel': categoria,
            'insight': f"**{categoria.title()}**: {ltv['melhor_categoria']} tem LTV {ltv['diferenca_percentual']:.1f}% maior que {ltv['pior_categoria']}"
        })
    
    return insights

def exibir_dashboard():
    st.markdown("<h1 style='color: #FF8C00;'>SaleSniper - Análise de ICP (Ideal Customer Profile)</h1>", unsafe_allow_html=True)

    # Se já temos dados calculados, usar eles
    if st.session_state.get("icp_data") is not None:
        sistema = st.session_state["icp_data"]["sistema"]
        df = sistema.df
        capitao = st.session_state["icp_data"]["capitao"]
        correlacoes = st.session_state["icp_data"]["correlacoes"]
        
        # Botão para recalcular
        if st.button("Carregar novo arquivo"):
            st.session_state["icp_data"] = None
            st.rerun()
    else:
        # Carregamento inicial dos dados
        arquivo_clientes = st.file_uploader(
            "Envie o arquivo Excel de clientes (.xlsx)", type="xlsx", key="clientes"
        )

        if not arquivo_clientes:
            st.warning("Envie um arquivo Excel para continuar.")
            st.stop()

        with st.spinner("Carregando e processando dados..."):
            # Usando função cacheada para carregar e preprocessar dados
            df_original = carregar_clientes_do_excel(arquivo_clientes)
            df = carregar_e_preprocessar_dados(df_original)
            
            sistema = Sistema()
            sistema.carregar_dados(df)
            
            # Obtendo variáveis padrão
            vars_cat, vars_num = get_variaveis_default()
            
            # Usando função cacheada para análise ICP
            capitao, correlacoes = calcular_analise_icp(
                df,
                vars_cat,
                vars_num
            )
            
            # Salvar no estado da sessão
            st.session_state["icp_data"] = {
                "sistema": sistema,
                "capitao": capitao,
                "correlacoes": correlacoes,
                "df": df
            }

    # Exibição dos resultados
    st.markdown("### 🦸‍♂️ Perfil de Cliente Ideal")
    st.markdown("Perfil do cliente que gera o maior ticket médio")
    
    # Convertendo o DataFrame para dicionário
    perfil = capitao.iloc[0].to_dict()
    
    # Métricas financeiras em destaque
    col1, col2, col3 = st.columns(3)
    with col1:
        # Aplicar cor laranja aos valores das métricas principais
        st.markdown(f"<p style='color: #F0F0F0; font-size: 1.2em;'>Ticket Médio</p>", unsafe_allow_html=True)
        # Aplicar formatação brasileira antes de exibir em laranja
        ticket_medio_formatado = formatar_valor(perfil['ticket_medio'])
        st.markdown(f"<h3 style='color: #FF8C00;'>{ticket_medio_formatado}</h3>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p style='color: #F0F0F0; font-size: 1.2em;'>LTV</p>", unsafe_allow_html=True)
        # Aplicar formatação brasileira antes de exibir em laranja
        ltv_formatado = formatar_valor(perfil['ltv'])
        st.markdown(f"<h3 style='color: #FF8C00;'>{ltv_formatado}</h3>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<p style='color: #F0F0F0; font-size: 1.2em;'>LT (meses)</p>", unsafe_allow_html=True)
        # Para LT (meses), a formatação :.0f já é suficiente para exibir como inteiro
        st.markdown(f"<h3 style='color: #FF8C00;'>{perfil['meses_ativo']:.0f}</h3>", unsafe_allow_html=True)

    # Características do perfil ideal - manter como markdown com texto claro
    st.markdown("### 📋 Perfil Ideal")
    for campo in ['porte', 'dores', 'localizacao', 'segmento']:
        if campo in perfil:
            st.markdown(f"<span style='color: #F0F0F0;'>- **{campo.replace('_', ' ').title()}:** {perfil[campo]}</span>", unsafe_allow_html=True)

    # Processando correlações (usando cache)
    insights_rapidos = _processar_correlacoes(correlacoes)

    # Agrupar insights_rapidos por categoria
    insights_agrupados: Dict[str, List[Dict[str, str]]] = {}
    for insight_item in insights_rapidos:
        categoria = insight_item['variavel'].lower() # Usar a variável como chave da categoria
        if categoria not in insights_agrupados:
            insights_agrupados[categoria] = []

        # Extrair o texto limpo do insight (sem o prefixo "[Categoria]: ")
        insight_texto_limpo = insight_item['insight'].split(': ', 1)[-1] # Divide em no máximo 2 partes e pega a última
        
        # Gerar ação sugerida usando o texto limpo do insight
        acao_sugerida = gerar_acao_sugerida_para_insight(insight_texto_limpo)

        insights_agrupados[categoria].append({
            "insight": insight_item['insight'], # Manter o insight original para exibição se necessário, ou mudar para limpo
            "acao": acao_sugerida
        })

    # Definir as categorias e criar colunas para exibição
    categorias_para_exibir = ['dores', 'localizacao', 'porte', 'segmento']
    
    # Mover o título para antes das colunas
    st.markdown("### 🚀 Insights e Ações Sugeridas")

    cols = st.columns(len(categorias_para_exibir))

    # Exibir insights e ações em colunas por categoria
    for i, categoria in enumerate(categorias_para_exibir):
        with cols[i]:
            # Aplicar cor laranja ao título da categoria
            st.markdown(f"<h4 style='color: #FF8C00;'>{categoria.replace('_', ' ').title()}</h4>", unsafe_allow_html=True)
            if categoria in insights_agrupados:
                for item in insights_agrupados[categoria]:
                    # Remover o prefixo "**[Categoria]**: " do insight antes de exibir
                    insight_texto_limpo = item['insight'].split('**: ', 1)[-1] # Divide em no máximo 2 partes e pega a última
                    # Aplicar cor laranja ao texto do Insight
                    st.markdown(f"<span style='color: #FF8C00; font-weight: bold;'>Insight:</span> <span style='color: #F0F0F0;'>{insight_texto_limpo}</span>", unsafe_allow_html=True)
                    # Aplicar cor laranja ao texto da Ação Sugerida
                    st.markdown(f"<span style='color: #FF8C00; font-weight: bold;'>Ação Sugerida:</span> <span style='color: #F0F0F0;'>{item['acao']}</span>", unsafe_allow_html=True)
                    st.markdown("---") # Separador entre insights na mesma coluna
            else:
                st.info(f"Nenhum insight disponível para {categoria.replace('_', ' ').title()}.")
