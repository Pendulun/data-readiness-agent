from langchain.tools import tool


@tool
def dummy_tool():
    """
    Essa tool não faz nada, ela só existe para que o langchain não reclame
    """
    return "Nenhuma ação necessária. Pare de chamar tools e gere sua resposta final agora"
