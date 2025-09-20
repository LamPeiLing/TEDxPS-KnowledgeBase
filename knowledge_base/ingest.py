import json
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

TRANSCRIPT_PATH = 'data/tedxps_transcripts.json'
METADATA_PATH = 'data/tedx_data.json'
DB_FAISS_PATH = 'vectorstore/faiss_db'

def create_vector_db():
    # Load transcripts
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        transcripts = json.load(f)

    # Load metadata (speaker, titles, links, etc.)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata_entries = json.load(f)

    # Build a lookup dict by ID
    metadata_lookup = {str(entry["id"]): entry for entry in metadata_entries}

    documents = []
    for video_id, text in transcripts.items():

        meta = metadata_lookup.get(str(video_id), {})

        # Extract multilingual title and name (default en + zh if available)
        title_en = None
        title_zh = None
        name_en = None
        name_zh = None

        if "translations" in meta:
            for tr in meta["translations"]:
                if tr["locale"] == "en":
                    title_en = tr.get("title")
                    name_en = tr.get("name")
                elif tr["locale"] == "zh":
                    title_zh = tr.get("title")
                    name_zh = tr.get("name")

        # Create Document with both locales in metadata
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "id": video_id,
                    "youtube_link": meta.get("youtube_link"),
                    "title_en": title_en,
                    "title_zh": title_zh,
                    "name_en": name_en,
                    "name_zh": name_zh,
                    "occupation": meta.get("occupation"),
                    "bio": meta.get("bio"),
                    "event_title": meta.get("event", {}).get("title"),
                }
            )
        )

    # Split long transcripts into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "], # fallback splitting rules
        length_function=len
    )
    texts = text_splitter.split_documents(documents)

    # Multilingual embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        model_kwargs={'device': 'cpu'}
    )

    # Create FAISS vectorstore
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()