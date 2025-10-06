import os
from typing import Dict

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables
load_dotenv()

# --- 1. SETUP: LLM and Tools ---

print("Setting up LLM...")
# Set up the Hugging Face model
try:
    hf_api_key = os.getenv("HUGGINGFACE_API_TOKEN")
    if not hf_api_key:
        raise ValueError("HUGGINGFACE_API_TOKEN not found in .env file")
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.7,
        huggingfacehub_api_token=hf_api_key
    )
    chat_model = ChatHuggingFace(llm=llm_endpoint)
except Exception as e:
    print(f"Error setting up Hugging Face model: {e}")
    exit()

# Set up the tools the agent can use
tools = [DuckDuckGoSearchRun(name="web_search")]


# --- 2. DEFINE THE CHAINS (ROUTER, AGENT, and CHAT) ---

# 2.1 The Agent Chain (for complex, tool-using queries)
agent_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(chat_model, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# Ensure agent output is a simple string
agent_chain = agent_executor | (lambda x: x["output"])

# 2.2 The Simple Chat Chain (for conversational queries)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and friendly AI assistant for RideOnCabio. Answer the user's questions clearly and concisely, in 30 words or less. If the user asks about their booked cab or vehicle, you must reply with: 'your cab is not confirmed yet pls wait until it confirms'"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])
chat_chain = chat_prompt | chat_model | StrOutputParser()

# 2.3 The Router (to decide which chain to use)
router_prompt_template = """
You are a helpful assistant for RideOnCabio, a small demo ride-sharing project.

Your job is to classify the user's input as either 'agent' or 'chat'.

Guidelines:
- If the user asks about their ride, cab status, driver, or booking (for example: "where is my ride", "is my cab confirmed", "who is my driver", "when will the cab arrive"),
  do NOT switch to agents. Instead, simply respond:
  "No rider has accepted your ride yet, please wait until it confirms."

- If the user talks about trip planning, fare estimation, distance, location routes, or any question that requires reasoning or external data,
  classify it as 'agent'.

- For greetings, small talk, or general chat, classify it as 'chat'.

Output format:
Return only one word: either 'agent' or 'chat'.

User Input:
{input}
"""

router_prompt = ChatPromptTemplate.from_template(router_prompt_template)
router = router_prompt | chat_model | StrOutputParser()


# --- 3. CREATE THE MAIN CHAIN WITH BRANCHING LOGIC ---

# RunnableBranch uses the output of the router to decide which chain to run.
branch = RunnableBranch(
    (lambda x: "agent" in x["topic"].lower(), agent_chain),
    chat_chain, # Default to the chat chain
)

# The full chain first invokes the router to get the topic, then passes the input to the chosen branch.
full_chain = {"topic": router, "input": lambda x: x["input"], "chat_history": lambda x: x["chat_history"]} | branch


# --- 4. ADD MEMORY AND RUN THE CHAT LOOP ---

store: Dict[str, ChatMessageHistory] = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap the entire branched chain in memory
final_chain = RunnableWithMessageHistory(
    full_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def get_bot_response(question: str, session_id: str) -> str:
    """
    Get the chatbot's response to a given question.
    """
    try:
        response = final_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        return response
    except Exception as e:
        return f"An error occurred: {e}"
