from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

FILE_PATH = "../documents/Rich-Dad-Poor-Dad.pdf"
# FILE_PATH = "../documents/OReilly.TypeScript.Cookbook.pdf"

#Create Loader
loader = PyPDFLoader(FILE_PATH)

#Split Document
pages = loader.load_and_split()
# print(len(pages))


#Embedding functions
embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Create vector store
vectordb = Chroma.from_documents(
    documents=pages,
    embedding=embedding_function,
    persist_directory="../vector_db",
    collection_name="rich_dad_poor_dad"
)


#Make persistent
vectordb.persist()