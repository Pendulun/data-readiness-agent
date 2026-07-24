# data-readyness-agent

## Aprendizados

1. **Funções determinísticas provavelmente não precisam ser tools.** A não ser que ela seja cara, pode fazer sentido computar ela de uma vez no início antes de invocar o agente e informar o seu resultado em um State inicial.
2. **O agente pode querer chamar uma mesma tool repetidamente com os mesmos parâmetros de entrada.**
3. **É útil ter variações de uma mesma tool que possui entrada de tamanhos diferentes.** Por exemplo, a Tool `data_readyness_agent.agent.py:check_duplicate_rows` recebe um subconjunto de colunas e calcula algo. Para calcular para todas as colunas de uma vez, eu criei a variante `data_readyness_agent.agent.py:check_duplicate_rows_all_cols` de forma que o agente não precisa informar todas as colunas separadamente, diminuindo a quantidade de tokens geradas.
4. **Não assumir que o agente vai informar uma entrada válida para uma tool.** Por exemplo, a tool `data_readyness_agent.agent.py:detect_outliers` e várias outras checam se a coluna informada pelo agente realmente existe na base. Em especial, vários vezes a LLM quis informar uma coluna 'id' mesmo ela não existindo. Basicamente, trate o agente como um usuário qualquer de um sistema que pode inserir informações inválidas.
5. **O agente não consegue acessar diretamente o State inicial.** Mesmo informando um State inicial, a LLM não sabe nada além das mensagens e contextos passadas na hora da sua invocação. Portanto, é necessário que existam tools capazes de acessar os dados do State inicial para a LLM consultar.
6. **Limitar a quantidade de iterações do agente** O agente estava entrando em um loop de chamadas de tools mesmo que elas já tenham sido chamadas antes. Adicionar a propriedade de ciclos máximos de investigação é uma forma de forçar o agente a gerar a resposta e economizar tokens.
