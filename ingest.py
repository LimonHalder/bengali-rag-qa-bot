from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

#text_path = "data/input.txt"

passage_path = "resource/process_data/section1.txt"
mcq_path = "resource/process_data/section2.txt"


embed_model_id = "l3cube-pune/bengali-sentence-similarity-sbert"
chunk_size = 400
chunk_overlap = 200
persist_dir = "./chroma_db"
collection_name = "qa_collection"

# Load and chunk text
# with open(text_path, "r", encoding="utf-8") as f:
#     text = f.read()

passage_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "।", "!", "?", "\n", ",", ";", ":"],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

mcq_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "।", "!", "?", "\n", ",", ";", ":"],
    chunk_size=100,
    chunk_overlap=10
)


with open(passage_path, 'r', encoding='utf-8') as f:
    passage_text = f.read()

with open(mcq_path, 'r', encoding='utf-8') as f:
    mcq_text = f.read()

passage_chunks = passage_splitter.split_text(passage_text)
mcq_chunks = mcq_splitter.split_text(mcq_text)

docs = [
    Document(page_content=chunk, metadata={"source": "passage"})
    for chunk in passage_chunks
] + [
    Document(page_content=chunk, metadata={"source": "mcq"})
    for chunk in mcq_chunks
]


# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name=embed_model_id)

# Build and persist vector store
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory=persist_dir,
    collection_name=collection_name,
)
db.persist()
print(f"✅ Vector store created and persisted in: {persist_dir}")

