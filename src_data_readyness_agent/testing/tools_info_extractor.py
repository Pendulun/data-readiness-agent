import pandas as pd

from src_data_readyness_agent.data_transformation_agent.data_structs import ToolHistory
from src_data_readyness_agent.testing import config


def main(tool_history: ToolHistory) -> pd.DataFrame:
    tool_name_to_id_map = {
        row['name']: row.tool_id
        for idx, row in pd.read_csv(config.TOOLS_IDS_MAP_PATH).iterrows()
    }
    return get_tool_usage_df(tool_history.history_as_dict(),
                             tool_name_to_id_map)


def get_tool_usage_df(tool_usage: dict, tool_ids: dict) -> pd.DataFrame:
    tools_ids = list()
    tools_sucess = list()
    tools_args = list()
    tools_names = list()
    for tool, uses in tool_usage.items():
        for use in uses:
            tools_names.append(tool)
            tools_ids.append(tool_ids[tool])
            tools_sucess.append(use['sucess'])
            tools_args.append(str(use['args']))

    df = pd.DataFrame()
    df['tool_id'] = tools_ids
    df['tool_name'] = tools_names
    df['sucess'] = tools_sucess
    df['args'] = tools_args

    return df
