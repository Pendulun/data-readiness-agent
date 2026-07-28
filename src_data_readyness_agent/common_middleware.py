from langchain.agents.middleware import AgentMiddleware
from langchain.messages import AIMessage, SystemMessage

from src_data_readyness_agent import common_tools


class DebugMiddleware(AgentMiddleware):

    def after_model(self, state, runtime):
        last_message = state["messages"][-1]

        if isinstance(last_message, AIMessage):
            print("\n--- LLM RESPONSE ---")
            print("Content:", last_message.content)
            print("Tool calls:", last_message.tool_calls)

        return None


class IterationLimitMiddleware(AgentMiddleware):
    """
    Limita as iterações do agente
    """

    def __init__(self, max_iterations: int, dummy_tool: bool = False):
        self.max_iterations = max_iterations
        self.dummy_tool = dummy_tool

    def wrap_model_call(self, request, handler):
        iteration = self.count_model_calls(request.state["messages"])

        print(f"Iteração do agente: "
              f"{iteration}/{self.max_iterations}")

        # A última iteração é reservada para gerar a resposta final
        if iteration >= self.max_iterations - 1:
            print("Limite de iterações atingido.")

            request = request.override(
                # Isso requer que o dummy_tool seja indicado durante a instanciação do agente
                tools=[common_tools.dummy_tool] if self.dummy_tool else [],
                messages=[
                    *request.messages,
                    SystemMessage(
                        content=("O limite de investigação foi atingido. "
                                 "Não execute mais ferramentas. "
                                 "Gere agora a resposta final estruturada "
                                 "com base nas informações coletadas."))
                ])

        return handler(request)

    def count_model_calls(self, messages) -> int:
        # Conta quantas respostas da LLM já existem
        return sum(1 for message in messages
                   if message.__class__.__name__ == "AIMessage")
