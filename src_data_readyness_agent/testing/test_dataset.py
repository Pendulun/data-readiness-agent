from dataclasses import dataclass
import pandas as pd
from typing_extensions import Dict, Tuple

from src_data_readyness_agent import agent
from src_data_readyness_agent.testing import expectations_structs, expectations_verifier, langsmith_trace_info, tools_info_extractor


@dataclass
class DatasetConfig():
    target_col: str
    prefered_language: str
    eval_agent_max_its: int
    eval_agent_max_supersteps: int
    transform_agent_max_its: int
    transform_agent_max_supersteps: int
    user_entry: str


@dataclass
class BenchmarkResults():
    eval_agent_results: pd.DataFrame
    transform_agent_results: pd.DataFrame
    transform_agent_tool_usage_results: pd.DataFrame


def get_model_configs() -> Tuple[Dict[str, str]]:
    """
    Retorna a ordem das configurações de modelos a serem utilizados nos testes de um dataset
    """
    return (
        {
            'eval_model': 'gpt-4.1-nano',
            'transform_model': 'gpt-4.1-nano'
        },
        {
            'eval_model': 'gpt-4o-mini',
            'transform_model': 'gpt-4.1-nano'
        },
        {
            'eval_model': 'gpt-4o-mini',
            'transform_model': 'gpt-4o-mini'
        },
        {
            'eval_model': 'gpt-5-mini',
            'transform_model': 'gpt-4o-mini'
        },
        {
            'eval_model': 'gpt-5-mini',
            'transform_model': 'gpt-5-mini'
        },
        {
            'eval_model': 'gpt-5-nano',
            'transform_model': 'gpt-5-mini'
        },
        {
            'eval_model': 'gpt-5-nano',
            'transform_model': 'gpt-5-nano'
        },
        {
            'eval_model': 'gpt-4.1-nano',
            'transform_model': 'gpt-5-nano'
        },
    )


def main(dataset: pd.DataFrame,
         openai_api_key: str,
         langsmith_api_key: str,
         langsmith_project: str,
         eval_expectations: expectations_structs.EvalAgentExpectations,
         transform_expectations: expectations_structs.
         TransformAgentExpectations,
         dataset_config: DatasetConfig,
         runs_per_config: int = 1) -> BenchmarkResults:
    """
    Avalia o sistema retornando os resultados
    """
    assert runs_per_config >= 1, f"runs_per_config não pode ser menor que 1! Recebeu {runs_per_config}"

    important_values_check = {
        'openai_api_key': openai_api_key,
        'langsmith_api_key': langsmith_api_key,
        'langsmith_project': langsmith_project
    }
    for name, val in important_values_check.items():
        assert val is not None and len(
            val.strip()) > 0, f'{name} não pode estar vazio!'

    configs = get_model_configs()

    all_eval_results = list()
    all_transform_results = list()
    all_transform_agent_tool_usage_results = list()

    run_id = 1
    for config_id, config in enumerate(configs):
        for inner_run_id in range(runs_per_config):
            print(
                f"[LOG] Começou config_id: {config_id+1}/{len(configs)} inner_run: {inner_run_id+1}/{runs_per_config}"
            )
            avaliacao, final_eval_df = _eval_evaluation_agent(
                dataset, openai_api_key, langsmith_api_key, langsmith_project,
                eval_expectations, dataset_config, run_id,
                config['eval_model'])

            all_eval_results.append(final_eval_df)

            final_transform_df, transform_tools_usage_info = _eval_transformation_agent(
                dataset, openai_api_key, langsmith_api_key, langsmith_project,
                transform_expectations, dataset_config, run_id,
                config['transform_model'], avaliacao)
            all_transform_results.append(final_transform_df)
            all_transform_agent_tool_usage_results.append(
                transform_tools_usage_info)
            run_id += 1

    eval_results = pd.concat(all_eval_results, ignore_index=True)
    transform_results = pd.concat(all_transform_results, ignore_index=True)
    transform_agent_tool_usage_results = pd.concat(
        all_transform_agent_tool_usage_results, ignore_index=True)

    return BenchmarkResults(
        eval_agent_results=eval_results,
        transform_agent_results=transform_results,
        transform_agent_tool_usage_results=transform_agent_tool_usage_results,
    )


def _eval_transformation_agent(
    dataset: pd.DataFrame,
    openai_api_key: str,
    langsmith_api_key: str,
    langsmith_project: str,
    transform_expectations: expectations_structs.TransformAgentExpectations,
    dataset_config: DatasetConfig,
    run_id: str,
    model: str,
    avaliacao: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Avalia o agente de transformação retornando seus resultados
    """
    transform_agent_inputs = agent.TransformAgentInputs(
        findings_str=avaliacao,
        dataset=dataset,
        openai_api_key=openai_api_key,
        langsmith_api_key=langsmith_api_key,
        langsmith_project=langsmith_project,
        qt_max_iteracoes_agente=dataset_config.transform_agent_max_its,
        qt_maxima_supersteps=dataset_config.eval_agent_max_supersteps,
        model=model)

    dados_transformados, tool_history = agent.get_base_transformada(
        transform_agent_inputs)

    transform_verifications_result_df = expectations_verifier.get_transform_agent_verification_results(
        dados_transformados, tool_history.current_column_names,
        transform_expectations)
    transform_trace_info = langsmith_trace_info.get_last_transformation_agent_trace(
        langsmith_api_key=langsmith_api_key,
        langsmith_project=langsmith_project,
    )
    tools_info = tools_info_extractor.main(tool_history)

    final_transform_df = pd.concat(
        [transform_verifications_result_df, transform_trace_info], axis=1)
    final_transform_df['model'] = model
    final_transform_df['run_id'] = run_id

    tools_info['run_id'] = run_id
    return final_transform_df, tools_info


def _eval_evaluation_agent(
    dataset: pd.DataFrame,
    openai_api_key: str,
    langsmith_api_key: str,
    langsmith_project: str,
    eval_expectations: expectations_structs.EvalAgentExpectations,
    dataset_config: DatasetConfig,
    run_id: str,
    model: str,
) -> Tuple[str, pd.DataFrame]:
    """
    Avalia o agente de avaliação retornando os resultados
    """

    eval_inputs = agent.EvalAgentInputs(
        dataset=dataset,
        openai_api_key=openai_api_key,
        langsmith_api_key=langsmith_api_key,
        langsmith_project=langsmith_project,
        qt_maxima_iteracoes_agente=dataset_config.eval_agent_max_its,
        target_col=dataset_config.target_col,
        user_entry=dataset_config.user_entry,
        qt_maxima_supersteps=dataset_config.eval_agent_max_supersteps,
        prefered_language=dataset_config.prefered_language,
        model=model)
    avaliacao = agent.get_avaliacao(eval_inputs)

    eval_verifications_result_df = expectations_verifier.get_eval_agent_verification_results(
        eval_expectations,
        avaliacao.findings,
    )

    trace_info = langsmith_trace_info.get_last_evaluation_agent_trace(
        langsmith_api_key=langsmith_api_key,
        langsmith_project=langsmith_project,
    )

    final_eval_df = pd.concat([eval_verifications_result_df, trace_info],
                              axis=1)
    final_eval_df['model'] = model
    final_eval_df['run_id'] = run_id
    return avaliacao.get_suggested_actions_str(), final_eval_df
