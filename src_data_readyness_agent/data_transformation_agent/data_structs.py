import copy
from pydantic import BaseModel, Field
from typing_extensions import Any, Dict, List


class ToolUsageInfo(BaseModel):
    args: Dict[str, Any] = Field(default_factory=dict,
                                 description="Os argumentos usados na tool")
    sucess: bool = Field(
        default=None,
        description="Indica se a tool foi executada com sucesso ou não")


class ToolHistoryUsage(BaseModel):
    tool_name: str = Field(description="O nome da tool")
    tool_calls: List[ToolUsageInfo] = Field(
        default_factory=list, description="Histórico de chamadas da tool")

    def add(self, usage: ToolUsageInfo):
        self.tool_calls.append(usage)

    def __len__(self):
        return len(self.tool_calls)

    @property
    def n_sucess_calls(self) -> int:
        return sum([call.sucess for call in self.tool_calls])


class SingleToolCall(BaseModel):
    tool_name: str = Field(description="O nome da tool")
    tool_args: ToolUsageInfo = Field(
        description="Informações da execução da chamada")


class ToolUsagePerCol(BaseModel):
    column: str = Field(description="Coluna")
    tool_calls: List[SingleToolCall] = Field(
        default_factory=list,
        description="Histórico de chamadas de tools envolvendo a coluna")

    def add(self, tool_call: SingleToolCall):
        self.tool_calls.append(tool_call)


class ToolHistory(BaseModel):
    history: Dict[str, ToolHistoryUsage] = Field(
        default_factory=dict, description="Histórico de uso de tools")
    sucess_history_per_col: Dict[str, ToolUsagePerCol] = Field(
        default_factory=dict,
        description="Histórico de uso de sucesso de tools por colunas")
    current_column_names: dict[str, str] = Field(default_factory=dict)

    def add_usage(self, tool_name: str, args_dict: Dict[str, Any],
                  sucess: bool):
        """
        Registra o uso de uma tool
        """
        args = copy.deepcopy(args_dict)
        usage_info = ToolUsageInfo(args=args, sucess=sucess)
        self.history.setdefault(
            tool_name, ToolHistoryUsage(tool_name=tool_name)).add(usage_info)

        # Como as colunas podem ser renomeadas, mantém guardado o novo valor de cada coluna original
        # Isso é útil ao fazer o benchmarking do sistema
        if tool_name == 'rename_column':
            original = args['column']

            # Se a coluna já foi renomeada antes, preserve a origem
            for orig, current in self.current_column_names.items():
                if current == original:
                    original = orig
                    break

            self.current_column_names[original] = args['new_column']

    def add_col_usage(self,
                      column: str,
                      tool_name: str,
                      args_dict: Dict[str, Any],
                      sucess=bool):
        """
        Registra o uso de uma tool sob uma coluna específica
        """
        args = ToolUsageInfo(args=copy.deepcopy(args_dict), sucess=sucess)

        usage_info = SingleToolCall(tool_name=tool_name, tool_args=args)
        self.sucess_history_per_col.setdefault(
            column, ToolUsagePerCol(column=column)).add(usage_info)

    def n_calls_per_tool(self) -> Dict[str, int]:
        result = dict()
        for tool, calls in self.history.items():
            result[tool] = len(calls)
        return result

    def n_all_tool_calls(self) -> int:
        return sum([len(calls) for calls in self.history.values()])

    def n_tools_called(self) -> int:
        return len(self.history.keys())

    def sucess_rate_per_tool(self) -> Dict[str, float]:
        result = dict()
        for tool, calls in self.history.items():
            n_calls = len(calls)
            n_sucess_calls = calls.n_sucess_calls
            # Como uma tool só está presente no histórico caso ela tenha sido
            # chamada, n_calls > 0
            result[tool] = n_sucess_calls / n_calls
        return result

    def history_as_dict(self):
        return {
            tool: [call.model_dump() for call in usages.tool_calls]
            for tool, usages in self.history.items()
        }

    def col_transformation_history_as_dict(self):
        return {
            col: [usage.model_dump() for usage in usages.tool_calls]
            for col, usages in self.sucess_history_per_col.items()
        }
