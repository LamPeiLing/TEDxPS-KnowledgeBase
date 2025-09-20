from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langdetect import detect

vectorstore_path = "knowledge_base/vectorstore/faiss_db"

template = """
You are an expert TEDx Petaling Street chatbot. First, answer the user's question using your knowledge base.
Then, provide a bullet list of TEDx videos as supporting references with their title, speaker name, and YouTube link. Say "I Don't Know" if you are not sure.
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""

def detect_language(query: str) -> str:
    try:
        lang = detect(query)
        if lang.startswith("zh"):
            return "zh"
        elif lang.startswith("en"):
            return "en"
        elif lang.startswith("ms"):  # Malay
            return "ms"
        else:
            return "en"  # fallback
    except:
        return "en"

def format_reference(doc, lang="en"):
    if lang == "zh":
        title = doc.metadata.get("title_zh") or doc.metadata.get("title_en")
        name = doc.metadata.get("name_zh") or doc.metadata.get("name_en")
    else:  # English fallback
        title = doc.metadata.get("title_en") or doc.metadata.get("title_zh")
        name = doc.metadata.get("name_en") or doc.metadata.get("name_zh")

    return f"- 《{title}》 by {name}\n  {doc.metadata.get('youtube_link')}"

def set_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 2}), return_source_documents=True, chain_type_kwargs={'prompt': prompt})
    return qa_chain

#Loading the model
def load_llm():
    llm = CTransformers(
        model = "TheBloke/openchat-3.5-0106-GGUF",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.7
    )
    # llm = ChatOpenAI(temperature=0.3)
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa


def chatbot_response(query: str):
    # Detect query language
    lang = detect_language(query)

    # Run retrieval QA
    qa_result = qa_bot()
    response = qa_result({'query': query})
    answer = response["result"]
    sources = response["source_documents"]

    # Format references in user’s language
    bullets = "\n".join([format_reference(doc, lang) for doc in sources])

    return f"{answer}\n\n **References:**\n{bullets}"

if __name__ == "__main__":
    q = input("Ask a TEDx question: ")
    print(chatbot_response(q))
