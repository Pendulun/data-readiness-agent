import pandas as pd
from langsmith import Client
import time
from typing_extensions import Optional

from src_data_readyness_agent import config


def get_last_evaluation_agent_trace(
        langsmith_api_key: str,
        langsmith_project: str) -> Optional[pd.DataFrame]:
    """
    Tenta pegar informações do último trace realizado desde que ele seja do agente de avaliação 
    """
    return _get_last_eval_with_tag(langsmith_api_key, langsmith_project,
                                   config.EVAL_AGENT_LANGSMITH_TAG)


def get_last_transformation_agent_trace(
        langsmith_api_key: str,
        langsmith_project: str) -> Optional[pd.DataFrame]:
    """
    Tenta pegar informações do último trace realizado desde que ele seja do agente de transformação 
    """
    return _get_last_eval_with_tag(langsmith_api_key, langsmith_project,
                                   config.TRANSFORM_AGENT_LANGSMITH_TAG)


def _get_last_eval_with_tag(langsmith_api_key: str, langsmith_project: str,
                            target_tag: str) -> Optional[pd.DataFrame]:
    """
        Tenta pegar informações do último trace realizado desde que ele seja do agente de avaliação 
        """
    client = Client(api_key=langsmith_api_key)

    havent_found_right_trace = True
    count = 0
    trace_info = None
    MAX_TRIES = 5
    SLEEP_TIME = 0.3
    while havent_found_right_trace and count < MAX_TRIES:
        print(f"[LOG] Tentativa {count+1}/{MAX_TRIES} de conseguir o trace")
        last_trace = next(
            client.list_runs(
                project_name=langsmith_project,
                is_root=True,
                limit=1,
            ))
        if target_tag in last_trace.tags:
            print("[LOG] Conseguiu!")
            metrics = {
                "input_tokens":
                last_trace.input_tokens,
                "cached_input_tokens": (last_trace.input_token_details
                                        or {}).get("cache_read", 0),
                "output_tokens":
                last_trace.output_tokens,
                "reasoning_output_tokens": (last_trace.output_token_details
                                            or {}).get("reasoning", 0),
                "total_tokens":
                last_trace.total_tokens,
                "input_cost":
                last_trace.input_cost,
                "output_cost":
                last_trace.output_cost,
                "total_cost":
                last_trace.total_cost,
                'langsmith_trace_id':
                last_trace.id
            }
            trace_info = pd.DataFrame({
                col: [val]
                for col, val in metrics.items()
            })
            havent_found_right_trace = False
        else:
            time.sleep(SLEEP_TIME)
            count += 1

    if havent_found_right_trace:
        print(
            f"[ERROR] Não foi possível conseguir o trace correto após {count} tentativas!"
        )

    return trace_info
