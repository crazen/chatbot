import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import Document

# -----------------------------
# Configurações
# -----------------------------
HIST_FILE = "chat_history.json"

# Carregar variáveis de ambiente
load_dotenv()
try:
    nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
except:
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")

if nvidia_api_key:
    os.environ['NVIDIA_API_KEY'] = nvidia_api_key
else:
    st.error("⚠️ NVIDIA_API_KEY não encontrada. Configure nos Secrets do Streamlit.")
    st.stop()

# Inicialização do session_state
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "final_documents" not in st.session_state:
    st.session_state.final_documents = []

# -----------------------------
# Inicializar Memória
# -----------------------------
def load_memory():
    if "memory" not in st.session_state:
        memory = ConversationBufferMemory(return_messages=True)
        # Se houver histórico salvo, recarregar
        if os.path.exists(HIST_FILE):
            with open(HIST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for m in data:
                memory.chat_memory.add_user_message(m["user"])
                memory.chat_memory.add_ai_message(m["ai"])
        st.session_state.memory = memory

def save_memory():
    """Salva histórico no arquivo JSON."""
    history = []
    messages = st.session_state.memory.chat_memory.messages
    for i in range(0, len(messages), 2):  # pares user/ai
        if i + 1 < len(messages):
            history.append({"user": messages[i].content, "ai": messages[i+1].content})
    with open(HIST_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# -----------------------------
# Função para carregar documentos TXT e PDF
# -----------------------------
def carregar_docs(pasta="docs"):
    documentos = []
    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)

        if arquivo.lower().endswith(".pdf"):
            loader = PyPDFLoader(caminho)
            documentos.extend(loader.load())

        elif arquivo.lower().endswith(".txt"):
            loader = TextLoader(caminho, encoding="utf-8")
            documentos.extend(loader.load())

    return documentos

# -----------------------------
# Função para criar embeddings e FAISS
# -----------------------------
def vector_embedding():
    if not st.session_state.vectors:
        st.session_state.embeddings = NVIDIAEmbeddings()

        # Carregar PDFs e TXTs
        docs = carregar_docs("./docs")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        st.session_state.final_documents = text_splitter.split_documents(docs)

        if st.session_state.final_documents:
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )
            st.success("✅ Vector Store DB criado com sucesso!")
        else:
            st.warning("⚠ Nenhum documento válido encontrado na pasta docs.")

# -----------------------------
# Interface Streamlit (ChatGPT-like)
# -----------------------------
st.set_page_config(page_title="Nvidia NIM Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Nvidia NIM Demo")

llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
load_memory()

# Prompt para RAG
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only. If the user tries to ask something out of the context into your database, pdfs or texts, say you can't help with anything thats out of the climate research context.
<context>
{context}
<context>
Question: {input}
""")

# Mostrar histórico no estilo chat
for msg in st.session_state.memory.chat_memory.messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Botão para criar embeddings
if st.button("📂 Criar Embeddings dos Documentos"):
    vector_embedding()

# Entrada no estilo chat
if user_input := st.chat_input("Digite sua pergunta..."):

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("🤖 Pensando..."):
        # Se há base vetorial, usar RAG
        if st.session_state.vectors:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({'input': user_input})
            answer = response['answer']

            # Atualizar memória
            st.session_state.memory.chat_memory.add_user_message(user_input)
            st.session_state.memory.chat_memory.add_ai_message(answer)
            save_memory()

            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(f"⏱ Tempo de resposta: {time.process_time() - start:.2f} segundos")

            # Mostrar documentos similares
            with st.expander("🔎 Documentos semelhantes"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(doc.page_content)
                    st.write("---")
        else:
            # Se não há FAISS, apenas memória de conversa simples
            conversation = ConversationChain(llm=llm, memory=st.session_state.memory, verbose=False)
            answer = conversation.predict(input=user_input)
            save_memory()

            with st.chat_message("assistant"):
                st.markdown(answer)
