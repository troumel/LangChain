# %%
import os
from dotenv import load_dotenv
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chat_models.openai import ChatOpenAI

# %%
url = "https://365datascience.com/upcoming-courses"

# %%
loader = WebBaseLoader(url)
raw_documents = loader.load()
print(raw_documents)

# %%
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(raw_documents)
print(documents)

# %%
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

api_key = api_key.strip()

# %%
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# %%
vectorstore = FAISS.from_documents(documents, embeddings)

# %%
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
print(memory)

# %%
qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0),
    vectorstore.as_retriever(),
    memory=memory,
)

# %%
query = "What data science courses are available? Who are the instructors?"
result = qa({"question": query})

# %%
result["answer"]
