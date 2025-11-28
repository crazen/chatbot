# streamlit_app.py
import streamlit as st
import os, json, time
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from pathlib import Path

load_dotenv()

# Supabase config - colocar em secrets do Streamlit ou .env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Configure SUPABASE_URL e SUPABASE_KEY nas variáveis de ambiente.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Função utilitária para obter user_id da query string
def get_current_user():
    params = st.query_params
    user_id = params.get("user_id", None)
    token = params.get("access_token", None)
    return user_id, token


# Local storage paths por usuário
def user_storage_dir(user_id: str):
    base = Path("user_data")
    base.mkdir(exist_ok=True)
    user_dir = base / user_id
    user_dir.mkdir(exist_ok=True)
    return user_dir

# Carregar histórico do Supabase (se existir) senão local
def load_user_history(user_id: str):
    if not user_id:
        return []
    resp = supabase.table("chat_history").select("messages").eq("user_id", user_id).order("updated_at", desc=True).limit(1).execute()
    if resp and resp.data and len(resp.data) > 0:
        return resp.data[0]["messages"]
    # fallback: arquivo local
    p = user_storage_dir(user_id) / "chat_history.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []

def save_user_history(user_id: str, messages):
    # messages: array of {role, content}
    # Upsert: mantemos apenas uma linha por usuário (latest)
    if not user_id:
        return
    row = {"user_id": user_id, "messages": messages}
    # Try upsert: delete previous and insert to keep simple
    supabase.table("chat_history").delete().eq("user_id", user_id).execute()
    supabase.table("chat_history").insert(row).execute()
    # local fallback copy
    p = user_storage_dir(user_id) / "chat_history.json"
    p.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")

# FAISS index por usuário (opcional)
def faiss_path_for_user(user_id: str):
    return user_storage_dir(user_id) / "faiss_index.pkl"

# Carregar docs (igual ao seu)
def carregar_docs(pasta="docs"):
    documentos = []
    if not os.path.exists(pasta):
        return documentos
    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)
        if arquivo.lower().endswith(".pdf"):
            loader = PyPDFLoader(caminho)
            documentos.extend(loader.load())
        elif arquivo.lower().endswith(".txt"):
            loader = TextLoader(caminho, encoding="utf-8")
            documentos.extend(loader.load())
    return documentos

# --- STREAMLIT UI ---
st.set_page_config(page_title="Nvidia NIM Chatbot (Multi-User)", page_icon="🤖", layout="centered")
st.title("🤖 Nvidia NIM Demo — Multi-User com Supabase")

user_id, access_token = get_current_user()
if not user_id:
    st.warning("Você não está autenticado. Faça login na página de login e seja redirecionado com ?user_id=...")
    st.info("Abra a página de login (login.html), autentique e volte.")
    st.stop()

st.markdown(f"**Usuário atual:** `{user_id}`")

# Inicializar client LLM
llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")

# Carregar histórico para conversa
history_messages = load_user_history(user_id)  # lista de {role, content}
if "memory" not in st.session_state or st.session_state.get("user_for_memory") != user_id:
    mem = ConversationBufferMemory(return_messages=True)
    # popular memória com histórico do Supabase
    for m in history_messages:
        if m.get("role") == "user":
            mem.chat_memory.add_user_message(m.get("content"))
        elif m.get("role") == "assistant":
            mem.chat_memory.add_ai_message(m.get("content"))
    st.session_state.memory = mem
    st.session_state.user_for_memory = user_id

# embeddings / faiss por usuário
if "vectors" not in st.session_state or st.session_state.get("vectors_user") != user_id:
    emb = NVIDIAEmbeddings()
    st.session_state.embeddings = emb
    docs = carregar_docs("./docs")
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    final_documents = splitter.split_documents(docs)

    # build or load faiss per user
    faiss_file = faiss_path_for_user(user_id)
    if faiss_file.exists():
        try:
            st.session_state.vectors = FAISS.load_local(str(faiss_file), st.session_state.embeddings)
        except Exception as e:
            st.warning("Erro ao carregar FAISS local. Recriando. " + str(e))
            st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
            st.session_state.vectors.save_local(str(faiss_file))
    else:
        if final_documents:
            st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
            st.session_state.vectors.save_local(str(faiss_file))
        else:
            st.session_state.vectors = None
    st.session_state.vectors_user = user_id

# Prompt para RAG
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
<context>
{context}
<context>
Question: {input}
""")

# Mostrar histórico (como chat)
for msg in st.session_state.memory.chat_memory.messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Input do usuário
if user_input := st.chat_input("Digite sua pergunta..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("🤖 Pensando..."):
        if st.session_state.vectors:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({'input': user_input})
            answer = response['answer']

            # atualizar memória e persistir no Supabase
            st.session_state.memory.chat_memory.add_user_message(user_input)
            st.session_state.memory.chat_memory.add_ai_message(answer)

            # reconstruir messages array para salvar (role, content)
            messages = []
            for m in st.session_state.memory.chat_memory.messages:
                role = "user" if m.type == "human" else "assistant"
                messages.append({"role": role, "content": m.content})

            save_user_history(user_id, messages)

            st.session_state.memory = st.session_state.memory  # manter
            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(f"⏱ Tempo de resposta: {time.process_time() - start:.2f} segundos")

            with st.expander("🔎 Documentos semelhantes"):
                for i, doc in enumerate(response.get("context", [])):
                    st.markdown(doc.page_content)
                    st.write("---")
        else:
            # fallback: sem RAG
            conversation = ConversationChain(llm=llm, memory=st.session_state.memory, verbose=False)
            answer = conversation.predict(input=user_input)

            st.session_state.memory.chat_memory.add_user_message(user_input)
            st.session_state.memory.chat_memory.add_ai_message(answer)

            messages = []
            for m in st.session_state.memory.chat_memory.messages:
                role = "user" if m.type == "human" else "assistant"
                messages.append({"role": role, "content": m.content})

            save_user_history(user_id, messages)

            with st.chat_message("assistant"):
                st.markdown(answer)
