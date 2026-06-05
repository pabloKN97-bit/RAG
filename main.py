import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Imports modernos y estables para RAG
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings

app = FastAPI()

# ── Configuración de CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONFIGURACIÓN DEL RAG ──
DOCS_DIR = "datos"
MODEL_NAME = "qwen2.5:0.5b" 

if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)
    with open(os.path.join(DOCS_DIR, "ejemplo.txt"), "w", encoding="utf-8") as f:
        f.write("El código secreto de la empresa es 99X22. El horario de atención es de 9:00 a 18:00.")

print("Cargando y procesando documentos...")
loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model=MODEL_NAME)
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ── PROMPT ESTRICTO ──
system_prompt = (
    "Eres un asistente virtual estricto. Tu única tarea es responder preguntas basándote "
    "exclusivamente en el contexto proporcionado abajo.\n"
    "REGLAS CRÍTICAS:\n"
    "1. Si la respuesta no viene explícitamente en el contexto, di exactamente: 'No tengo esa información en mi base de datos.'\n"
    "2. No inventes, no supongas, y no uses tu conocimiento general externo.\n"
    "3. Mantén las respuestas claras y al grano.\n\n"
    "Contexto:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

llm = OllamaLLM(model=MODEL_NAME)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── ENDPOINTS DE LA API (Primero definimos las rutas) ──
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_ia(request: QueryRequest):
    try:
        # 1. Recuperar los documentos relevantes
        relevant_docs = retriever.invoke(request.question)
        context_text = format_docs(relevant_docs)
        
        # 2. Unir todo en el prompt y enviarlo al LLM
        chain = prompt | llm
        response = chain.invoke({"context": context_text, "input": request.question})
        
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── ARRANQUE DEL SERVIDOR ──
print("¡Backend del RAG listo y desplegado!")
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=5000)
