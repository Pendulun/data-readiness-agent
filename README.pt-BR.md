# data-readiness-agent

🇺🇸 [Read in English](README.md)

O sistema desenvolvido analisa e transforma uma base estruturada de forma que ela fique mais adequada para uma tarefa de modelagem de machine learning clássico (regressão ou classificação). Para tal, um primeiro agente avalia a base e gera pontos de melhoria enquanto um segundo aplica transformações de acordo com a avaliação do primeiro. A aplicação dos agentes é feita de forma sequencial, com o primeiro gerando toda a análise e o segundo consumindo a sua saída para decidir como transformar a base. Ambos os agentes possuem acesso a conjuntos separados de `tools` que recuperam informações ou agem diretamente sobre a base.

O projeto foi desenvolvido usando `langchain` como framework para gerenciamento dos agentes e do `streamlit` para criar a interface e publicação. Internamente, é usado o modelo `gpt-5-nano`da OpenAI em ambos os agentes. Esse modelo é o modelo mais barato da OpenAI disponível atualmente (07/2026). Dessa forma, é necessário que o usuário informe uma chave própria de acesso à API da OpenAI. O custo de processar uma base depende do seu tamanho e da quantidade de problemas encontrados pelo sistema. Em testes realizados com uma base do Kaggle com 5500+ linhas, 8 colunas e uma quantidade pequena de problemas tratados, os custos ficaram em cerca de 2 a 4 centavos de dólar por processamento.

## Aprendizados

1. **Funções determinísticas provavelmente não precisam ser tools.** A não ser que ela seja cara, pode fazer sentido computar ela de uma vez no início antes de invocar o agente e informar o seu resultado em um State inicial.
2. **O agente pode querer chamar uma mesma tool repetidamente com os mesmos parâmetros de entrada.**
3. **É útil ter variações de uma mesma tool que possui entrada de tamanhos diferentes.** Por exemplo, a Tool `data_readyness_agent.agent.py:check_duplicate_rows` recebe um subconjunto de colunas e calcula algo. Para calcular para todas as colunas de uma vez, eu criei a variante `data_readyness_agent.agent.py:check_duplicate_rows_all_cols` de forma que o agente não precisa informar todas as colunas separadamente, diminuindo a quantidade de tokens geradas.
4. **Não assumir que o agente vai informar uma entrada válida para uma tool.** Por exemplo, a tool `data_readyness_agent.agent.py:detect_outliers` e várias outras checam se a coluna informada pelo agente realmente existe na base. Em especial, vários vezes a LLM quis informar uma coluna 'id' mesmo ela não existindo. Basicamente, trate o agente como um usuário qualquer de um sistema que pode inserir informações inválidas.
5. **O agente não consegue acessar diretamente o State inicial.** Mesmo informando um State inicial, a LLM não sabe nada além das mensagens e contextos passadas na hora da sua invocação. Portanto, é necessário que existam tools capazes de acessar os dados do State inicial para a LLM consultar.
6. **Limitar a quantidade de iterações do agente** O agente estava entrando em um loop de chamadas de tools mesmo que elas já tenham sido chamadas antes. Adicionar a propriedade de ciclos máximos de investigação é uma forma de indicar o agente a gerar a resposta e economizar tokens.
7. **Um agente que só valida a base e encontra problemas não é agêntico ou tão útil**. Dessa forma, tratar a validação como um agente e as correções como outro é algo mais complexo e útil.
8. **É possível impedir transformações paralelas do agente**. O agente de transformação de dados tentava aplicar múltiplas alterações nos dados originais em um mesmo ciclo e isso fazia levantar erros. Foi possível impedir isso com `model_kwargs={"parallel_tool_calls": False}` na instanciação do modelo. Isso faz com que o agente chame apenas uma tool por ciclo.
9. **É possível limitar com certeza a execução de um agente**. Existe o argumento `config={"recursion_limit": <qt_maxima_supersteps>}` que pode ser informado na invocação do agente para limitar a quantidade de supersteps a serem executados. Se o agente ultrapassa esse limite, ele levanta um erro e não gera a resposta. Em um sistema em que pessoas vão ser cobradas pelo uso (usando a chave da OpenAI), impedir que o agente entre em loop é uma forma de não causar prejuízos indesejados.

## Executando localmente

A partir da pasta raiz do projeto, crie um ambiente virtual com: `make install`. A seguir, rode o sistema com `make run`. É necessário ter o `uv` instalado. 

Também é possível instalar o projeto manualmente com o `pip` ao criar um ambiente virtual, instalar as dependências indicadas em `pyproject.toml` e rodar o sistema com `python -m streamlit run main.py`.
