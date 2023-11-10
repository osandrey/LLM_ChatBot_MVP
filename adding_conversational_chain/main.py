import os
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from decouple import config


# Disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


vector_db = Chroma(
    persist_directory="../vector_db",
    collection_name="rich_dad_poor_dad",
    embedding_function=embedding_function,
)


# create chat model
llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), temperature=0)

# create memory
memory = ConversationBufferMemory(
    return_messages=True, memory_key="chat_history")

# create retriever chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=vector_db.as_retriever(
        search_kwargs={'fetch_k': 4, 'k': 3}, search_type='mmr'),
    chain_type="refine",
)

# question
question = "Who is the name of the book?"

# call QA chain
response = qa_chain({"question": question})


print(response.get("answer"))