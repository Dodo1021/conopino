import os
import time
import unicodedata
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

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
pdf_folder     = r"C:\Users\David Sanchez\OneDrive\Documentos\Catalogos\Sandvik"

# 1️⃣ Inicializa el cliente Pinecone (SDK v4)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# 2️⃣ Crea el índice si no existe
existing = [info["name"] for info in pc.list_indexes()]
if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # compatible con text-embedding-3-large
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )
# Espera a que el índice esté listo
while not pc.describe_index(INDEX_NAME).status["ready"]:
    time.sleep(1)

# 3️⃣ Configura el modelo de embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

# 4️⃣ Procesa tus PDFs
all_docs = []
for fname in os.listdir(pdf_folder):
    if fname.lower().endswith(".pdf"):
        path = os.path.join(pdf_folder, fname)
        try:
            print(f"Procesando: {fname}...")
            loader = PyPDFLoader(path)
            docs = loader.load_and_split()
            all_docs.extend(docs)
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
    index_name=INDEX_NAME
)
print("✅ ¡Carga finalizada! Índice listo para usar.")

# 6️⃣ Limpieza final
# Pinecone SDK v4 no expone un método `close` en el cliente síncrono,
# así que no hay que cerrar explícitamente la conexión.
print("🔒 Proceso terminado.")
