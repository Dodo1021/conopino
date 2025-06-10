import os
import time
import unicodedata
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

# 🔐 Configuración de credenciales
# Asegúrate de usar solo guiones ASCII '-' en tu clave
os.environ["PINECONE_API_KEY"] = "pcsk_B55id_JjHn2TodDPmPRZ8r8tcwpjoge4t93AaGCyELzFjQpCdVukqUuw6gQh3t7K6EQsc"
os.environ["OPENAI_API_KEY"]  = (
    "sk-proj-UaYjSRoZpOps7RvAcJhU6NB0CNtgUYcOLA7TBsPFQYDPOmzS3M-AlMXpJhF0s_"
    "5MbBA8vG6EgmT3BlbkFJ8LYtycmFIcPpmVBK4Ol5UvCzRJaNU5K-G3Nlq3waRPRcthPKbCUQfcw5jnL43YM5-dd-Lw00IA"
)

# Opcional: detectar si quedan caracteres no ASCII en la API key
key = os.environ["OPENAI_API_KEY"]
non_ascii = [c for c in key if ord(c) > 127]
if non_ascii:
    print("⚠️ Caracteres no ASCII en OPENAI_API_KEY:", non_ascii)
else:
    print("✅ OPENAI_API_KEY limpia (solo ASCII)")

INDEX_NAME     = "catalogo"
PINECONE_REGION = "us-east-1"
excel_folder   = r"C:\Users\David Sanchez\OneDrive\Documentos\Catalogos\Sandvik"

# 1️⃣ Inicializa el cliente Pinecone (SDK v4)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# 2️⃣ Crea el índice si no existe
existing = [info["name"] for info in pc.list_indexes()]
if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME,
        # text-embedding-3-large devuelve vectores de 3072 dimensiones
        # para que Pinecone acepte los embeddings, el índice debe tener esa
        # misma dimensión
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )
# Espera a que el índice esté listo
while not pc.describe_index(INDEX_NAME).status["ready"]:
    time.sleep(1)

# 3️⃣ Configura el modelo de embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    # Limita los lotes a 100 textos para no exceder el máximo de tokens
    chunk_size=100,
)

# 4️⃣ Procesa tus archivos de Excel
all_docs = []
for fname in os.listdir(excel_folder):
    if fname.lower().endswith((".xls", ".xlsx")):
        path = os.path.join(excel_folder, fname)
        try:
            print(f"Procesando: {fname}...")
            sheets = pd.read_excel(path, sheet_name=None)
            for sheet_name, df in sheets.items():
                text = df.to_csv(index=False)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000, chunk_overlap=200
                )
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={"source": fname, "sheet": sheet_name},
                    )
                    all_docs.append(doc)
        except Exception as e:
            print(f"❌ Error en {fname}: {e}")
print(f"📄 Chunks generados: {len(all_docs)}")

# Función para limpiar texto y evitar problemas de codificación
def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c if c.isprintable() else " " for c in text)
    text = text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    return text

for doc in all_docs:
    doc.page_content = clean_text(doc.page_content)

# 5️⃣ Carga los embeddings a Pinecone via langchain-pinecone
print("🚀 Subiendo datos a Pinecone...")
vectorstore = PineconeVectorStore.from_documents(
    all_docs,
    embedding=embeddings,
    index_name=INDEX_NAME,
    # Usa lotes más pequeños para evitar solicitudes con demasiados tokens
    embeddings_chunk_size=100,
)
print("✅ ¡Carga finalizada! Índice listo para usar.")

# 6️⃣ Limpieza final
# Pinecone SDK v4 no expone un método `close` en el cliente síncrono,
# así que no hay que cerrar explícitamente la conexión.
print("🔒 Proceso terminado.")
