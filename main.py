from dotenv import dotenv_values
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper

config = dotenv_values(".env")

search = GoogleSearchAPIWrapper(google_api_key=config.get("GOOGLE_API_KEY"), google_cse_id=config.get("GOOGLE_CSE_ID"))
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="uses Internet engine"
    )
]
llm = ChatOpenAI(temperature=0, openai_api_key=config.get("OPENAI_API_KEY"))
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if __name__ == '__main__':
    agent_chain = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory)

    question = ""
    while question != "exit":
        question = input("YOUR QUESTION: ")
        if question.lower() in ["exit", "quit", "q"]:
            print("GOODBYE!")
            break
        answer = agent_chain.run(input=question)
        print("\nYOUR ANSWER:", answer)
