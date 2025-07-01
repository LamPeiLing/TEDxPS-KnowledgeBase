import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

metadata_path = Path("data/tedx_data.json")
transcript_path = Path("data/tedxps_transcripts.json")
vectorstore_path = "vectorstore/db_faiss"

def create_vector_db():
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcripts = json.load(f)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = []

    for video in metadata:
        transcript = transcripts.get(str(video["id"]))
        if not transcript:
            continue
        chunks = text_splitter.split_text(transcript)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={
                "id": video["id"],
                "link": video["youtube_link"],
                "title": video["title"],
                "speaker": video["name"],
                "occupation": video["occupation"],
                "bio": video["bio"],
                "introduction": video["introduction"],
                "translations": video["translations"]
            }))

    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(documents, embedding)
    db.save_local(vectorstore_path)

if __name__ == "__main__":
    create_vector_db()