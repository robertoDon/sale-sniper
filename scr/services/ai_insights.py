import subprocess
import json
from typing import Dict, List
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import requests
import time
from functools import lru_cache
import hashlib
import os
import re
import traceback
from huggingface_hub import InferenceClient

def _preparar_correlacoes(correlacoes: Dict[str, pd.DataFrame]) -> List[str]:
    """Prepara o texto das correlações de forma otimizada."""
    df = correlacoes["todas"].copy()
    df = df[~df["variavel"].isin(["ltv", "ticket_medio"])]
    
    # Encontrar correlações mais relevantes
    top_ltv = df.nlargest(3, "correlacao_com_ltv")
    top_ticket = df.nlargest(3, "correlacao_com_ticket")
    
    # Preparar textos em paralelo
    correlacoes_texto = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        # Adicionar correlações LTV
        for _, row in top_ltv.iterrows():
            futures.append(
                executor.submit(
                    lambda r: f"- {r['variavel']}: correlação de {r['correlacao_com_ltv']:.2f} com LTV",
                    row
                )
            )
        
        # Adicionar correlações Ticket Médio (apenas se não estiver já em LTV)
        for _, row in top_ticket.iterrows():
            if row['variavel'] not in top_ltv['variavel'].values:
                futures.append(
                    executor.submit(
                        lambda r: f"- {r['variavel']}: correlação de {r['correlacao_com_ticket']:.2f} com Ticket Médio",
                        row
                    )
                )
        
        # Coletar resultados
        for future in futures:
            correlacoes_texto.append(future.result())
    
    return correlacoes_texto

def _gerar_hash_prompt(prompt: str) -> str:
    """Gera um hash único para o prompt."""
    return hashlib.md5(prompt.encode()).hexdigest()



def _gerar_prompt(correlacoes_texto: List[str], top_insights: List[str]) -> str:
    """Gera o prompt para a IA de forma otimizada."""
    return f"""Analise os dados e escreva um relatório executivo conciso:

DADOS:
{chr(10).join(correlacoes_texto)}

INSIGHTS:
{chr(10).join(top_insights)}

Escreva um texto (20 linhas aproximadamente) em formato de parágrafos. 
O texto deve fluir naturalmente, sem títulos ou subtítulos explícitos. 
Estruture o texto da seguinte forma:

1. Primeiro parágrafo: Introduza as descobertas mais importantes (3-4 linhas)

2. Parágrafos seguintes: Desenvolva uma análise objetiva, explicando:
   - Significado prático das correlações
   - Impacto no negócio
   - Padrões de comportamento identificados
   - Oportunidades de crescimento

3. Parágrafo final: Conclua com recomendações estratégicas, incluindo:
   - Ações específicas
   - Métricas a monitorar
   - Timeline sugerida

Use linguagem profissional e direta.
Inclua exemplos concretos com dados específicos.
Mantenha o foco em insights acionáveis.
Seja específico ao mencionar segmentos, portes, dores ou localizações.
Sem títulos ou subtítulos."""

def _gerar_fallback(top_ltv: pd.Series, top_ticket: pd.Series) -> str:
    """Gera texto de fallback caso a IA falhe."""
    if not top_ltv.empty and not top_ticket.empty:
        # Extrair os valores específicos das variáveis
        var_ltv = top_ltv.iloc[0]
        var_ticket = top_ticket.iloc[0]
        
        # Determinar se as variáveis são qualitativas (segmento, porte, dor, localização)
        variaveis_qualitativas = {
            'segmento': 'do segmento',
            'porte': 'do porte',
            'dor': 'com a dor',
            'localizacao': 'da localização'
        }
        
        # Preparar as descrições das variáveis
        def preparar_descricao(var):
            variavel = var['variavel']
            if variavel in variaveis_qualitativas:
                if 'valor' in var and var['valor']:
                    # Se tiver múltiplos valores, separa por vírgula
                    valores = var['valor'].split(';') if ';' in str(var['valor']) else [var['valor']]
                    if len(valores) > 1:
                        valores_formatados = ', '.join([f'"{v.strip()}"' for v in valores])
                        return f"{variaveis_qualitativas[variavel]}s {valores_formatados}"
                    return f"{variaveis_qualitativas[variavel]} '{valores[0].strip()}'"
                return f"{variaveis_qualitativas[variavel]}"
            return variavel
        
        valor_ltv = preparar_descricao(var_ltv)
        valor_ticket = preparar_descricao(var_ticket)
        
        # Determinar se são os mesmos valores
        mesmo_valor = (var_ltv['variavel'] == var_ticket['variavel'] and 
                      'valor' in var_ltv and 'valor' in var_ticket and 
                      var_ltv['valor'] == var_ticket['valor'])
        
        texto_final = (
            f"Analisando nossa base de clientes, encontrei alguns padrões muito interessantes que podem nos ajudar "
            f"a entender melhor nosso negócio. O mais surpreendente foi descobrir que {valor_ltv} "
            f"tem um impacto enorme no LTV dos nossos clientes, com uma correlação de {var_ltv['correlacao_com_ltv']:.2f}. "
            f"Isso significa que quando focamos nessa característica, conseguimos aumentar significativamente o valor "
            f"que cada cliente gera para a empresa ao longo do tempo.\n\n"
            
            f"Por exemplo, quando olhamos para os clientes {valor_ltv}, vemos que eles "
            f"tendem a comprar mais vezes e gastar mais em cada compra. Isso é especialmente verdadeiro para "
            f"este perfil específico que se destaca nessa característica.\n\n"
            
            f"Outro ponto que chamou minha atenção foi a relação entre {valor_ticket} e o Ticket Médio. "
            f"A correlação de {var_ticket['correlacao_com_ticket']:.2f} nos mostra que essa variável tem um peso "
            f"importante no valor das compras dos clientes. Na prática, isso quer dizer que podemos usar essa informação "
            f"para ajustar nossa estratégia de preços e ofertas.\n\n"
            
            f"Quando analisamos mais a fundo, percebemos que clientes {valor_ticket} "
            f"têm um impacto ainda maior. Por exemplo, este perfil específico "
            f"tende a ter tickets médios significativamente mais altos.\n\n"
            
            f"Baseado nessas descobertas, acho que podemos fazer algumas mudanças interessantes. Primeiro, seria "
            f"legal criarmos ações específicas para potencializar o impacto de {valor_ltv} no LTV, "
            f"focando especialmente nos clientes que já se destacam com esta característica. Depois, podemos revisar "
            f"nossa abordagem de preços considerando {valor_ticket}, criando ofertas personalizadas "
            f"para os perfis que têm maior potencial. E por fim, que tal criarmos um programa de fidelização que "
            f"combine essas duas variáveis?\n\n"
            
            f"Essas mudanças, se implementadas juntas, podem trazer um aumento significativo tanto no ticket médio "
            f"quanto no LTV dos nossos clientes. Sugiro que a gente monitore os resultados nos próximos 90 dias "
            f"para ver como as coisas estão funcionando e fazer ajustes se necessário. Também seria interessante "
            f"acompanhar de perto o comportamento dos clientes {valor_ltv}"
        )
        
        if not mesmo_valor:
            texto_final += f" e {valor_ticket}"
        
        texto_final += ", para entender melhor como podemos replicar esse sucesso em outros perfis."
        
        return texto_final
    
    return "Não foi possível gerar insights com os dados disponíveis."

def gerar_insights_ia(correlacoes: Dict[str, pd.DataFrame]) -> str:
    print("[DEBUG] Iniciando geração de insights...")
    start_time = time.time()
    
    # Preparar dados de forma otimizada
    correlacoes_list = []
    
    # Processar correlações numéricas
    if 'numericas' in correlacoes:
        print("[DEBUG] Processando correlações numéricas...")
        matriz_corr = correlacoes['numericas']
        for var in matriz_corr.index:
            if var not in ['ltv', 'ticket_medio']:
                correlacoes_list.append({
                    'variavel': var,
                    'correlacao_com_ltv': matriz_corr.loc[var, 'ltv'],
                    'correlacao_com_ticket': matriz_corr.loc[var, 'ticket_medio']
                })
    
    # Processar correlações por categoria
    if 'categorias' in correlacoes:
        print("[DEBUG] Processando correlações por categoria...")
        for cat, dados in correlacoes['categorias'].items():
            # Adicionar o valor específico da categoria
            melhor_categoria_ltv = dados['ltv']['melhor_categoria']
            melhor_categoria_ticket = dados['ticket_medio']['melhor_categoria']
            
            correlacoes_list.append({
                'variavel': cat,
                'valor': melhor_categoria_ltv,
                'correlacao_com_ltv': dados['ltv']['diferenca_percentual'] / 100,
                'correlacao_com_ticket': dados['ticket_medio']['diferenca_percentual'] / 100
            })
    
    # Converter para DataFrame
    df = pd.DataFrame(correlacoes_list)
    print(f"[DEBUG] DataFrame criado com {len(df)} correlações")
    
    # Identificar top insights para o prompt
    top_ltv = df.nlargest(3, "correlacao_com_ltv")
    top_ticket = df.nlargest(3, "correlacao_com_ticket")
    
    try:
        print("[DEBUG] Preparando textos para o prompt...")
        # Preparar textos em paralelo
        correlacoes_texto = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            # Adicionar correlações LTV
            for _, row in top_ltv.iterrows():
                if 'valor' in row:
                    futures.append(
                        executor.submit(
                            lambda r: f"- {r['variavel']} '{r['valor']}': correlação de {r['correlacao_com_ltv']:.2f} com LTV",
                            row
                        )
                    )
                else:
                    futures.append(
                        executor.submit(
                            lambda r: f"- {r['variavel']}: correlação de {r['correlacao_com_ltv']:.2f} com LTV",
                            row
                        )
                    )
            
            # Adicionar correlações Ticket Médio (apenas se não estiver já em LTV)
            for _, row in top_ticket.iterrows():
                if row['variavel'] not in top_ltv['variavel'].values:
                    if 'valor' in row:
                        futures.append(
                            executor.submit(
                                lambda r: f"- {r['variavel']} '{r['valor']}': correlação de {r['correlacao_com_ticket']:.2f} com Ticket Médio",
                                r
                            )
                        )
                    else:
                        futures.append(
                            executor.submit(
                                lambda r: f"- {r['variavel']}: correlação de {r['correlacao_com_ticket']:.2f} com Ticket Médio",
                                r
                            )
                        )
            
            # Coletar resultados
            for future in futures:
                correlacoes_texto.append(future.result())
        
        print("[DEBUG] Gerando insights iniciais...")
        # Identificar top insights para o prompt
        top_insights = []
        for _, row in top_ltv.iterrows():
            if abs(row["correlacao_com_ltv"]) >= 0.2:
                if 'valor' in row:
                    top_insights.append(
                        f"- Forte relação entre {row['variavel']} '{row['valor']}' e LTV ({row['correlacao_com_ltv']:.2f})"
                    )
                else:
                    top_insights.append(
                        f"- Forte relação entre {row['variavel']} e LTV ({row['correlacao_com_ltv']:.2f})"
                    )
        
        for _, row in top_ticket.iterrows():
            if abs(row["correlacao_com_ticket"]) >= 0.2 and row["variavel"] not in [i.split()[3] for i in top_insights]:
                if 'valor' in row:
                    top_insights.append(
                        f"- Correlação significativa entre {row['variavel']} '{row['valor']}' e Ticket Médio ({row['correlacao_com_ticket']:.2f})"
                    )
                else:
                    top_insights.append(
                        f"- Correlação significativa entre {row['variavel']} e Ticket Médio ({row['correlacao_com_ticket']:.2f})"
                    )
        
        # Gerar prompt otimizado
        prompt = _gerar_prompt(correlacoes_texto, top_insights)
        print("[DEBUG] Prompt gerado, chamando Hugging Face via InferenceClient...")
        
        # Chamada para Hugging Face
        try:
            hf_token = os.environ.get("HF_TOKEN", None)
            client = InferenceClient(model="bigscience/bloomz-560m", token=hf_token)
            resposta = client.text_generation(prompt, max_new_tokens=200)
            print(f"[DEBUG] Resposta Hugging Face: {resposta.generated_text if hasattr(resposta, 'generated_text') else resposta}")
            if hasattr(resposta, 'generated_text'):
                return resposta.generated_text.strip()
            elif isinstance(resposta, str):
                return resposta.strip()
            else:
                print(f"[DEBUG] Resposta inesperada da Hugging Face: {resposta}")
        except Exception as e:
            print(f"[DEBUG] Erro ao gerar insights: {str(e)}")
            fallback = _gerar_fallback(top_ltv, top_ticket)
            print(f"[DEBUG] Fallback acionado: {fallback}")
            return fallback
    except Exception as e:
        print(f"[DEBUG] Erro ao gerar insights: {str(e)}")
        fallback = _gerar_fallback(top_ltv, top_ticket)
        print(f"[DEBUG] Fallback acionado: {fallback}")
        return fallback
    # Fallback final
    fallback = _gerar_fallback(top_ltv, top_ticket)
    print(f"[DEBUG] Fallback acionado: {fallback}")
    return fallback


def gerar_acao_sugerida_para_insight(insight_texto: str) -> str:
    """
    Gera uma ação sugerida para um insight específico usando a Hugging Face Inference API (modelo ptt5-base-portuguese-vocab).
    Se falhar, usa regra dinâmica baseada no insight.
    """
    print(f"[DEBUG] Gerando ação sugerida para o insight: {insight_texto}...")
    # Prompt para a Hugging Face
    hf_prompt = (
        f"Dado o insight: '{insight_texto}', gere uma ação sugerida criativa, específica e contextualizada para um gestor de vendas SaaS. Responda em português, apenas com a ação sugerida, sem explicações extras."
    )
    # 1. Tenta Hugging Face Inference API
    try:
        hf_token = os.environ.get("HF_TOKEN", None)
        client = InferenceClient(model="bigscience/bloomz-560m", token=hf_token)
        resposta = client.text_generation(hf_prompt, max_new_tokens=100)
        print(f"[DEBUG] Resposta Hugging Face: {resposta.generated_text if hasattr(resposta, 'generated_text') else resposta}")
        if hasattr(resposta, 'generated_text'):
            return resposta.generated_text.strip()
        elif isinstance(resposta, str):
            return resposta.strip()
        else:
            print(f"[DEBUG] Resposta inesperada da Hugging Face: {resposta}")
    except Exception as e:
        print(f"[DEBUG] Erro ao gerar ação sugerida: {str(e)}")
        print("[DEBUG] Acionando fallback dinâmico para ação sugerida.")
        # 2. Fallback dinâmico baseado no insight (personalizado)
        insight_lower = insight_texto.lower()
        match = re.search(r"([\wÀ-ÿ ]+) tem (ticket médio|ltv) ([\d\.,]+)% maior que ([\wÀ-ÿ ]+)", insight_texto, re.IGNORECASE)
        if match:
            grupo = match.group(1).strip()
            metrica = match.group(2).strip()
            percentual = match.group(3).strip()
            grupo_base = match.group(4).strip()
            if metrica.lower() == "ticket médio":
                return f"Foque campanhas de vendas no grupo {grupo} para aumentar ainda mais o ticket médio em relação a {grupo_base}."
            elif metrica.lower() == "ltv":
                return f"Invista em estratégias de retenção para clientes do grupo {grupo} visando elevar o LTV em relação a {grupo_base}."
        if "ticket médio" in insight_lower:
            return "Criar campanhas para otimizar o ticket médio do grupo destacado."
        if "ltv" in insight_lower:
            return "Desenvolver iniciativas para otimizar o LTV do grupo destacado."
        if any(reg in insight_lower for reg in ["região", "sudeste", "centro-oeste", "norte", "sul", "nordeste"]):
            return "Investir em campanhas direcionadas para a região destacada."
        if any(seg in insight_lower for seg in ["segmento", "saas", "retailtech", "saúde"]):
            return "Personalizar ofertas para o segmento identificado."
        if any(porte in insight_lower for porte in ["porte", "médio", "pequeno", "grande"]):
            return "Ajustar estratégias comerciais conforme o porte do cliente."
        if any(dor in insight_lower for dor in ["dor", "performance", "financeiro"]):
            return "Desenvolver soluções específicas para a dor identificada."
        print("[DEBUG] Fallback genérico acionado para ação sugerida.")
        return "Criar uma ação personalizada para este perfil visando aumentar LTV e ticket médio." 