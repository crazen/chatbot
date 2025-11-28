# streamlit_app.py
import streamlit as st
import os, json, time, uuid
from dotenv import load_dotenv
from supabase import create_client, Client
from pathlib import Path
from typing import List, Optional

# LangChain / NVIDIA
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

load_dotenv()

# -------------------------
# Supabase config
# -------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Configure SUPABASE_URL e SUPABASE_KEY nas variáveis de ambiente.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------
# Utilitários de usuário e disco
# -------------------------
def get_current_user():
    params = st.query_params
    user_id = params.get("user_id", None)
    token = params.get("access_token", None)
    return user_id, token

def user_storage_dir(user_id: str) -> Path:
    base = Path("user_data")
    base.mkdir(exist_ok=True)
    user_dir = base / user_id
    user_dir.mkdir(exist_ok=True)
    (user_dir / "uploads").mkdir(exist_ok=True)
    return user_dir

def faiss_path_for_session(user_id: str, session_id: str) -> Path:
    return user_storage_dir(user_id) / f"faiss_{session_id}"

# -------------------------
# Funções para perfil / avatar
# -------------------------
def get_user_profile(user_id: str) -> Optional[dict]:
    try:
        resp = supabase.table("users_profile").select("*").eq("id", user_id).limit(1).execute()
        if resp and getattr(resp, "data", None):
            return resp.data[0]
    except Exception:
        pass
    return None

def upsert_user_profile(user_id: str, full_name: Optional[str] = None, avatar_url: Optional[str] = None):
    payload = {"id": user_id}
    if full_name is not None:
        payload["full_name"] = full_name
    if avatar_url is not None:
        payload["avatar_url"] = avatar_url
    try:
        supabase.table("users_profile").upsert(payload).execute()
    except Exception as e:
        st.warning(f"Falha ao upsert users_profile: {e}")

def _normalize_public_url(url_obj):
    if isinstance(url_obj, dict):
        return url_obj.get("publicUrl") or url_obj.get("public_url") or url_obj.get("publicURL") or None
    return url_obj

def upload_avatar_to_storage(user_id: str, file) -> Optional[str]:
    try:
        filename = f"{user_id}_{int(time.time())}_{file.name}"
        content = file.read()
        supabase.storage.from_("avatars").upload(filename, content, file_options={"content-type": file.type})
        url_obj = supabase.storage.from_("avatars").get_public_url(filename)
        url = _normalize_public_url(url_obj) or url_obj
        upsert_user_profile(user_id, avatar_url=url)
        return url
    except Exception as e:
        st.warning("Erro ao enviar avatar para Storage: " + str(e))
        return None

# -------------------------
# Funções de sessões
# -------------------------
def get_sessions(user_id: str) -> List[dict]:
    try:
        r = (
            supabase.table("chat_sessions")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .execute()
        )
        return r.data if r and getattr(r, "data", None) else []
    except Exception:
        return []

def create_session(user_id: str, name: str = "Nova Conversa") -> str:
    try:
        r = supabase.table("chat_sessions").insert({"user_id": user_id, "name": name}).execute()
        if r and getattr(r, "data", None):
            return r.data[0]["id"]
    except Exception:
        pass
    sid = str(uuid.uuid4())
    try:
        supabase.table("chat_sessions").insert({"id": sid, "user_id": user_id, "name": name}).execute()
    except Exception:
        pass
    return sid

def rename_session(session_id: str, new_name: str):
    try:
        supabase.table("chat_sessions").update({"name": new_name}).eq("id", session_id).execute()
    except Exception as e:
        st.warning(f"Erro ao renomear: {e}")

def delete_session(session_id: str):
    try:
        supabase.table("chat_sessions").delete().eq("id", session_id).execute()
    except Exception as e:
        st.warning(f"Erro ao deletar sessão: {e}")

def load_messages(session_id: str) -> List[dict]:
    try:
        r = (
            supabase.table("chat_messages")
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        return r.data if r and getattr(r, "data", None) else []
    except Exception:
        return []

def save_message(session_id: str, role: str, content: str):
    try:
        supabase.table("chat_messages").insert({"session_id": session_id, "role": role, "content": content}).execute()
    except Exception as e:
        st.warning(f"Erro ao salvar mensagem: {e}")
    try:
        sync_chat_history_with_latest_session(session_id)
    except Exception:
        pass

def sync_chat_history_with_latest_session(session_id: str):
    msgs = load_messages(session_id)
    if not msgs:
        return
    try:
        r = supabase.table("chat_sessions").select("user_id").eq("id", session_id).limit(1).execute()
        if not r or not getattr(r, "data", None):
            return
        user_id = r.data[0]["user_id"]
        out = [{"role": m["role"], "content": m["content"]} for m in msgs]
        supabase.table("chat_history").delete().eq("user_id", user_id).execute()
        supabase.table("chat_history").insert({"user_id": user_id, "messages": out}).execute()
    except Exception:
        pass

# -------------------------
# CORREÇÃO: Carregar documentos de múltiplas fontes
# -------------------------
def load_documents_from_folder(folder_path: Path) -> List:
    """Carrega documentos de uma pasta específica"""
    documentos = []
    if not folder_path.exists():
        return documentos
    
    for arquivo in os.listdir(folder_path):
        caminho = folder_path / arquivo
        if not caminho.is_file():
            continue
            
        if arquivo.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(str(caminho))
                docs = loader.load()
                # Adicionar metadata de fonte
                for doc in docs:
                    doc.metadata["source_folder"] = folder_path.name
                documentos.extend(docs)
            except Exception as e:
                st.warning(f"Erro ao carregar PDF {arquivo}: {e}")
        elif arquivo.lower().endswith(".txt"):
            try:
                loader = TextLoader(str(caminho), encoding="utf-8")
                docs = loader.load()
                # Adicionar metadata de fonte
                for doc in docs:
                    doc.metadata["source_folder"] = folder_path.name
                documentos.extend(docs)
            except Exception as e:
                st.warning(f"Erro ao carregar TXT {arquivo}: {e}")
    
    return documentos

def load_all_documents(user_id: str) -> List:
    """
    Carrega documentos de TODAS as fontes:
    1. Pasta 'docs' (documentos globais/compartilhados)
    2. Uploads do usuário em 'user_data/{user_id}/uploads'
    """
    all_docs = []
    
    # 1. Carregar da pasta 'docs' (documentos globais)
    docs_folder = Path("docs")
    if docs_folder.exists():
        global_docs = load_documents_from_folder(docs_folder)
        st.sidebar.info(f"📚 {len(global_docs)} docs da pasta 'docs'")
        all_docs.extend(global_docs)
    
    # 2. Carregar uploads do usuário
    user_uploads = user_storage_dir(user_id) / "uploads"
    if user_uploads.exists():
        user_docs = load_documents_from_folder(user_uploads)
        st.sidebar.info(f"📁 {len(user_docs)} docs de uploads")
        all_docs.extend(user_docs)
    
    return all_docs

def save_uploaded_files_to_user(user_id: str, files):
    folder = user_storage_dir(user_id) / "uploads"
    for f in files:
        dest = folder / f.name
        with open(dest, "wb") as g:
            g.write(f.read())
        try:
            key = f"{user_id}/{int(time.time())}_{f.name}"
            with open(dest, "rb") as fh:
                supabase.storage.from_("user_uploads").upload(key, fh, file_options={"content-type": f.type})
        except Exception:
            pass

# -------------------------
# FAISS
# -------------------------
def get_or_create_faiss(user_id: str, session_id: str, documents, embeddings, force_recreate=False):
    """Carrega ou cria o índice FAISS"""
    faiss_path = faiss_path_for_session(user_id, session_id)
    
    if force_recreate or not faiss_path.exists():
        if not documents:
            return None
        try:
            vectors = FAISS.from_documents(documents, embeddings)
            vectors.save_local(str(faiss_path))
            return vectors
        except Exception as e:
            st.error(f"Erro ao criar índice FAISS: {e}")
            return None
    
    try:
        vectors = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
        return vectors
    except Exception as e:
        st.warning(f"Erro ao carregar FAISS: {e}. Recriando...")
        if documents:
            try:
                vectors = FAISS.from_documents(documents, embeddings)
                vectors.save_local(str(faiss_path))
                return vectors
            except Exception as e2:
                st.error(f"Erro ao recriar índice: {e2}")
                return None
        return None

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="NIM Chat — Multi-User", page_icon="🤖", layout="wide")
st.title("🤖 NIM Chat — Multi-User (Uploads, Avatar, Multi-Chat)")

user_id, access_token = get_current_user()
if not user_id:
    st.warning("Você não está autenticado. Faça login e volte com ?user_id=... na URL.")
    st.stop()

# Perfil do usuário
profile = get_user_profile(user_id)
col1, col2 = st.columns([1, 4])
with col1:
    if profile and profile.get("avatar_url"):
        st.image(profile.get("avatar_url"), width=120)
    else:
        st.write("Sem avatar")
    avatar_file = st.file_uploader("Enviar avatar", type=["png", "jpg", "jpeg"], key="avatar_uploader")
    if avatar_file:
        url = upload_avatar_to_storage(user_id, avatar_file)
        if url:
            st.success("Avatar enviado.")
            profile = get_user_profile(user_id)

with col2:
    st.markdown(f"**Usuário:** `{user_id}`")
    if profile and profile.get("full_name"):
        st.markdown(f"**Nome:** {profile.get('full_name')}")

st.write("---")

# Sidebar
with st.sidebar:
    st.header("💬 Conversas")
    sessions = get_sessions(user_id)
    
    if "current_session" not in st.session_state:
        if sessions:
            st.session_state.current_session = sessions[0]["id"]
        else:
            st.session_state.current_session = create_session(user_id, "Primeira conversa")

    sess_list_display = [f"{s['name']} — {s['id']}" for s in sessions]
    choice = st.selectbox("Selecione conversa", options=sess_list_display if sess_list_display else [f"Nova — {st.session_state.current_session}"])
    
    if choice:
        sid = choice.split("—")[-1].strip()
        if sid != st.session_state.current_session:
            st.session_state.current_session = sid
            if "memory_session" in st.session_state:
                del st.session_state["memory_session"]
            st.rerun()

    st.write("")
    if st.button("➕ Nova conversa"):
        nid = create_session(user_id, "Nova Conversa")
        st.session_state.current_session = nid
        if "memory_session" in st.session_state:
            del st.session_state["memory_session"]
        if "last_uploaded_files" in st.session_state:
            del st.session_state["last_uploaded_files"]
        st.rerun()

    new_name = st.text_input("Renomear conversa", key="rename_input")
    if st.button("Renomear"):
        if new_name:
            rename_session(st.session_state.current_session, new_name)
            st.success(f"✅ Renomeado para: {new_name}")
            time.sleep(1)
            st.rerun()

    if st.button("🗑️ Deletar conversa atual"):
        delete_session(st.session_state.current_session)
        s = get_sessions(user_id)
        if s:
            st.session_state.current_session = s[0]["id"]
        else:
            st.session_state.current_session = create_session(user_id, "Nova conversa")
        if "memory_session" in st.session_state:
            del st.session_state["memory_session"]
        if "last_uploaded_files" in st.session_state:
            del st.session_state["last_uploaded_files"]
        st.rerun()

    st.write("---")
    st.header("📁 Uploads")
    uploaded = st.file_uploader("Enviar documentos (pdf, txt)", type=["pdf", "txt"], accept_multiple_files=True, key="uploader")
    if uploaded:
        # Verificar se são arquivos novos
        if "last_uploaded_files" not in st.session_state:
            st.session_state["last_uploaded_files"] = []
        
        current_files = [f.name for f in uploaded]
        
        # Só processar se forem arquivos diferentes
        if current_files != st.session_state["last_uploaded_files"]:
            save_uploaded_files_to_user(user_id, uploaded)
            st.session_state["last_uploaded_files"] = current_files
            st.session_state["force_recreate_faiss"] = True
            st.success(f"✅ {len(uploaded)} arquivo(s) enviado(s)!")
            st.info("🔄 Clique em 'Recriar índice RAG' para indexar os novos documentos")
        else:
            st.info(f"📁 {len(uploaded)} arquivo(s) já carregado(s)")

    if st.button("🔄 Recriar índice RAG"):
        st.session_state["force_recreate_faiss"] = True
        st.rerun()

    st.write("---")
    st.markdown("**Configurações de embeddings**")
    if "chunk_size" not in st.session_state:
        st.session_state["chunk_size"] = 400  
    if "chunk_overlap" not in st.session_state:
        st.session_state["chunk_overlap"] = 50 

    st.session_state["chunk_size"] = st.number_input("Chunk size", min_value=100, max_value=500, value=st.session_state["chunk_size"], step=50, help="Máximo 500 para evitar erro de tokens")
    st.session_state["chunk_overlap"] = st.number_input("Chunk overlap", min_value=0, max_value=100, value=st.session_state["chunk_overlap"], step=10)

# -------------------------
# Inicialização
# -------------------------
llm = ChatNVIDIA(model="NVIDIABuild-Autogen-94")
emb = NVIDIAEmbeddings()

session_id = st.session_state.current_session
messages = load_messages(session_id)

# Memória
if "memory" not in st.session_state or st.session_state.get("memory_session") != session_id:
    mem = ConversationBufferMemory(return_messages=True)
    for m in messages:
        try:
            if m["role"] == "user":
                mem.chat_memory.add_user_message(m["content"])
            else:
                mem.chat_memory.add_ai_message(m["content"])
        except Exception:
            pass
    st.session_state.memory = mem
    st.session_state.memory_session = session_id

# -------------------------
# CORREÇÃO: Carregar TODOS os documentos
# -------------------------
docs = load_all_documents(user_id)
chunk_size = int(st.session_state.get("chunk_size", 700))
chunk_overlap = int(st.session_state.get("chunk_overlap", 100))
splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
final_documents = splitter.split_documents(docs) if docs else []

if docs:
    st.sidebar.success(f"✅ Total: {len(docs)} documentos")
    st.sidebar.info(f"📄 {len(final_documents)} chunks processados")
    
    # Mostrar amostra dos documentos
    if st.sidebar.checkbox("Ver amostra de docs"):
        st.sidebar.write("**Primeiros 3 documentos:**")
        for i, doc in enumerate(docs[:3]):
            st.sidebar.write(f"Doc {i+1}: {len(doc.page_content)} chars")
            st.sidebar.write(f"Preview: {doc.page_content[:100]}...")
else:
    st.sidebar.warning("⚠️ Nenhum documento encontrado")

force_recreate = st.session_state.pop("force_recreate_faiss", False)
vectors = get_or_create_faiss(user_id, session_id, final_documents, emb, force_recreate=force_recreate)

if vectors:
    try:
        test_results = vectors.similarity_search("test", k=1)
        st.sidebar.success(f"✅ RAG ativo com {len(test_results)} docs")
    except Exception:
        st.sidebar.warning("⚠️ RAG pode estar vazio")

# PROMPT MELHORADO - mais conciso e amigável
prompt = ChatPromptTemplate.from_template("""
Você é um assistente útil e conciso. Use o contexto fornecido para responder a pergunta.

Regras:
- Responda de forma direta e natural
- Se a informação não estiver no contexto, diga apenas: "Não encontrei essa informação nos documentos."
- Seja amigável mas objetivo
- Evite repetições e explicações desnecessárias, porém caso a mesma pergunta for feita,  responda como se fosse a primeira que fosse pergunado.

Contexto:
{context}

Pergunta: {input}

Resposta:""")

# -------------------------
# Histórico
# -------------------------
for m in st.session_state.memory.chat_memory.messages:
    if m.type == "human":
        with st.chat_message("user"):
            st.markdown(m.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(m.content)

# -------------------------
# Input
# -------------------------
if user_input := st.chat_input("Digite sua pergunta..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    save_message(session_id, "user", user_input)
    st.session_state.memory.chat_memory.add_user_message(user_input)

    start = time.process_time()
    with st.spinner("🤖 Pensando..."):
        if vectors:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vectors.as_retriever(search_kwargs={"k": 4})
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            try:
                response = retrieval_chain.invoke({"input": user_input})
                answer = response.get("answer") or response.get("output_text") or response.get("result") or ""
                context_docs = response.get("context") or []
            except Exception as e:
                answer = f"Erro ao gerar resposta: {e}"
                context_docs = []
        else:
            # Sistema de prompt melhorado para conversação normal
            system_prompt = """Você é um assistente útil, amigável e conciso. 
Responda de forma natural e direta, sem ser repetitivo ou excessivamente formal. 
Se não souber algo, admita simplesmente sem longas explicações."""
            
            conversation = ConversationChain(
                llm=llm, 
                memory=st.session_state.memory, 
                verbose=False
            )
            try:
                answer = conversation.predict(input=user_input)
            except Exception as e:
                answer = f"Erro: {e}"
            context_docs = []

    save_message(session_id, "assistant", answer)
    st.session_state.memory.chat_memory.add_ai_message(answer)

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(f"⏱ {time.process_time() - start:.2f}s")
        
        if context_docs:
            with st.expander(f"🔎 Documentos usados ({len(context_docs)})"):
                for i, doc in enumerate(context_docs):
                    try:
                        content = getattr(doc, "page_content", None) or doc.get("page_content", None) or str(doc)
                        metadata = getattr(doc, "metadata", {}) or doc.get("metadata", {})
                        source = metadata.get("source", "Desconhecido")
                        source_folder = metadata.get("source_folder", "")
                        
                        st.markdown(f"**Doc {i+1}** ({source_folder}): `{Path(source).name}`")
                        st.text(content[:400] + "..." if len(content) > 400 else content)
                        st.write("---")
                    except Exception:
                        pass
        elif not vectors:
            st.info("💡 Adicione documentos para usar RAG")