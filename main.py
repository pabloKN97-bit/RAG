import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

app = FastAPI()

# ── Configuración de CORS ──
# Esto permite que tu archivo HTML (frontend) se comunique con este backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, cambia esto por la URL de tu web
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONFIGURACIÓN DEL RAG ──
# 1. Crea una carpeta llamada "datos" al lado de este archivo y mete tus archivos .txt ahí.
DOCS_DIR = "datos"
MODEL_NAME = "llama3" # O el modelo que tengas descargado en Ollama (ej. mistral, qwen2.5)

if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)
    # Creamos un archivo de ejemplo por si acaso
    with open(os.path.join(DOCS_DIR, "ejemplo.txt"), "w", encoding="utf-8") as f:
        f.write("El código secreto de la empresa es 99X22. El horario de atención es de 9:00 a 18:00.")

print("Cargando y procesando documentos...")
# Leer todos los archivos .txt de la carpeta
loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
docs = loader.load()

# Fragmentar el texto en trozos óptimos para el modelo
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Configurar el generador de vectores (Embeddings) usando Ollama
embeddings = OllamaEmbeddings(model=MODEL_NAME)

# Crear la base de datos vectorial en memoria con nuestros documentos
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Recupera los 3 fragmentos más relevantes

# ── PROMPT ESTRICTO (La clave de tu petición) ──
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

# Inicializar el modelo de Ollama y las cadenas de ejecución (Chains)
llm = Ollama(model=MODEL_NAME)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("¡Backend del RAG listo y desplegado!")

# ── ENDPOINTS DE LA API ──
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_ia(request: QueryRequest):
    try:
        # Ejecutar la búsqueda en los documentos y enviarla al LLM
        response = rag_chain.invoke({"input": request.question})
        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
