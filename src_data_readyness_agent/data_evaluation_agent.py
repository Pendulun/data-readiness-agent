from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import MessagesState
import pandas as pd

from data_readyness_agent import common_data_structs, common_middleware, data_evaluation_tools


class State(MessagesState):
    dataset: pd.DataFrame
    dataset_profile: common_data_structs.DatasetProfile | None = None


def get_agent(openai_api_key: str,
              qt_maxima_iteracoes_agente: int) -> CompiledStateGraph:
    """
    Retorna a instância do agente de avaliação
    """
    model = ChatOpenAI(model='gpt-5-nano',
                       temperature=0.0,
                       api_key=openai_api_key)
    system_prompt = """
    Você é responsável por avaliar a prontidão de uma base de dados
    para um projeto de Data Science.

    Comece utilizando o DatasetProfile fornecido. Depois, planeje a investigação
    que você vai fazer. Uma vez planejado, siga o planejamento usando as ferramentas
    necessárias.

    Não solicite novamente informações já disponíveis no DatasetProfile.

    Use as ferramentas apenas quando:
    1. uma informação não estiver disponível no perfil;
    2. for necessário aprofundar uma possível inconsistência;
    3. for necessário validar uma hipótese levantada durante a análise.

    Evite repetir chamadas de ferramentas com os mesmos argumentos,
    a menos que exista uma justificativa clara para obter novos dados.

    Responda em português.

    Não solicite mais informações do usuário, apenas preencha a estrutura de dados da resposta
    """
    return create_agent(
        model=model,
        system_prompt=system_prompt,
        response_format=ToolStrategy(common_data_structs.EvalAgentResponse),
        state_schema=State,
        tools=[
            data_evaluation_tools.get_columns_names,
            data_evaluation_tools.check_duplicate_rows,
            data_evaluation_tools.check_duplicate_rows_all_cols,
            data_evaluation_tools.check_column_consistency,
            data_evaluation_tools.get_column_value_distribution,
            data_evaluation_tools.analyze_missingness_patterns,
            data_evaluation_tools.detect_outliers,
            data_evaluation_tools.get_n_rows,
            data_evaluation_tools.get_n_cols,
            data_evaluation_tools.get_null_counts,
            data_evaluation_tools.get_unique_counts,
            data_evaluation_tools.get_col_preview,
        ],
        middleware=[
            common_middleware.DebugMiddleware(),
            common_middleware.IterationLimitMiddleware(
                max_iterations=qt_maxima_iteracoes_agente)
        ])
