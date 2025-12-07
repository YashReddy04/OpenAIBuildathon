import os
import tempfile
import streamlit as st
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from typing import List, Tuple, Optional
from dataclasses import dataclass
from uuid import uuid4
from datetime import datetime

# -------------------------
# IMPORTS (LLM / LangChain pieces)
# -------------------------
# REPLACED: Ollama imports -> OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
# from langchain.memory import ConversationBufferMemory  # commented as before
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate as CoreChatPromptTemplate
from langchain_core.documents import Document
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# -------------------------
# App config (Must be first)
# -------------------------
st.set_page_config(page_title="⚖ AI Legal Toolkit", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# OpenAI API key setup (NO hardcoding)
# -------------------------
# Tries Streamlit secrets first, then environment variable.
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY not found. Please set it in .streamlit/secrets.toml "
        'as OPENAI_API_KEY="your_key" or as an environment variable.'
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -------------------------
# AUTHENTICATION SETUP
# -------------------------
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("config.yaml not found. Please create the config file.")
    st.stop()

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

# -------------------------
# LOGIN / REGISTER LOGIC
# -------------------------
if 'authentication_status' not in st.session_state or st.session_state['authentication_status'] is None:
    # Create tabs for Login and Register
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        try:
            authenticator.login()
        except Exception as e:
            st.error(e)
    with tab2:
        try:
            result = authenticator.register_user(location='main')
            if result:
                email, username, name = result
                st.success('User registered successfully')
                # Save to config file
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(f"Registration Error: {e}")

# -------------------------
# If authenticated, continue
# -------------------------
if st.session_state.get("authentication_status"):
    # -------------------------
    # Styling (top-level CSS kept for compatibility)
    # -------------------------
    st.markdown(
        """
        <style>
        :root{
          --bg:#071019;
          --panel:#0c1220;
          --muted:#9aa4b2;
          --text:#e6eef7;
          --accent:#f6b042;
        }
        .stApp { background: linear-gradient(180deg,var(--bg), #041018); color:var(--text); }
        .header {
          display:flex; align-items:center; gap:16px; margin-bottom:18px;
          background: linear-gradient(90deg, rgba(12,18,28,0.65), rgba(8,12,18,0.6));
          padding:18px; border-left:4px solid var(--accent); border-radius:8px;
          box-shadow: 0 10px 30px rgba(2,6,23,0.6);
        }
        .brand { font-size:20px; font-weight:700; }
        .tagline { color:var(--muted); margin-top:4px; font-size:13px; }
        .panel {
          background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
          border-radius: 8px; padding: 16px; box-shadow: 0 8px 26px rgba(5,8,12,0.45);
          border: 1px solid rgba(255,255,255,0.03);
        }
        .panel h3 { margin:0; color:var(--text); font-size:16px; }
        .panel p { margin-top:6px; color:var(--muted); font-size:13px; }
        .muted { color:var(--muted); font-size:13px; }
        .accent { color:var(--accent); font-weight:700; }
        .stTextInput, .stFileUploader, .stTextArea { margin-bottom: 8px; }
        .stCodeBlock pre { background: rgba(255,255,255,0.02) !important; color:var(--muted) !important; border-radius:6px; padding:10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------
    # Load law CSVs (cached)
    # -------------------------
    @st.cache_data
    def load_law_csvs(constitution_path: str, bns_path: str) -> Tuple[str, str, List[Document]]:
        constitution_text = ""
        bns_text = ""
        docs: List[Document] = []
        # Load constitution CSV
        try:
            constitution_df = pd.read_csv(constitution_path, encoding="utf-8")
        except Exception:
            try:
                constitution_df = pd.read_csv(constitution_path, encoding="latin1")
            except FileNotFoundError:
                st.error(f"File not found: {constitution_path}")
                return "", "", []

        # Load BNS CSV
        try:
            bns_df = pd.read_csv(bns_path, encoding="utf-8")
        except Exception:
            try:
                bns_df = pd.read_csv(bns_path, encoding="latin1")
            except FileNotFoundError:
                st.error(f"File not found: {bns_path}")
                return "", "", []

        def df_to_text(df: pd.DataFrame, key_col_a=None, key_col_b=None, prefix=""):
            lines = []
            if key_col_a and key_col_b and key_col_a in df.columns and key_col_b in df.columns:
                for _, row in df.iterrows():
                    lines.append(f"{prefix} {row[key_col_a]}: {row[key_col_b]}")
            else:
                for _, row in df.iterrows():
                    lines.append(" | ".join([f"{col}: {row[col]}" for col in df.columns]))
            return "\n".join(lines)

        constitution_text = df_to_text(constitution_df, key_col_a="Article", key_col_b="Description", prefix="Article")
        bns_text = df_to_text(bns_df, key_col_a="Section", key_col_b="Description", prefix="Section")

        for _, row in constitution_df.iterrows():
            text = " | ".join([f"{col}: {row[col]}" for col in constitution_df.columns])
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(constitution_path)}))

        for _, row in bns_df.iterrows():
            text = " | ".join([f"{col}: {row[col]}" for col in bns_df.columns])
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(bns_path)}))

        return constitution_text, bns_text, docs

    CONSTITUTION_PATH = os.getenv("CONSTITUTION_CSV", "data/Constitution Of India.csv")
    BNS_PATH = os.getenv("BNS_CSV", "data/bns_sections.csv")


    constitution_text, bns_text, law_docs = load_law_csvs(CONSTITUTION_PATH, BNS_PATH)

    # -------------------------
    # Helpers: load FIR
    # -------------------------
    def load_fir_text(uploaded_file) -> str:
        if not uploaded_file:
            return ""
        if uploaded_file.name.lower().endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            text = " ".join([page.page_content for page in pages])
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        else:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
        return text.strip()

    # -------------------------
    # LLM & embeddings (OpenAI)
    # -------------------------
    @st.cache_resource
    def get_llm(model_name: str = "gpt-4.1", temperature: float = 0.7):
        # High temperature for more creative/lengthy generation
        return ChatOpenAI(model=model_name, temperature=temperature)

    @st.cache_resource
    def get_embeddings():
        # Use a modern OpenAI embedding model
        return OpenAIEmbeddings(model="text-embedding-3-large")

    # -------------------------
    # Simulation prompt & runner (REALISTIC & LENGTHY)
    # -------------------------
    SIM_PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""You are conducting a realistic Indian courtroom trial simulation based on the provided FIR, BNS sections, and Constitution of India.

    Below is the contextual data you must base the simulation on:

    {context}

    Conversation History:
    {history}

    User's query or issue:
    {question}

    ### Courtroom Simulation Rules:
    - The trial should proceed in dialogue format, alternating between:
      - 🧑‍⚖ Judge
      - 👨‍💼 Defense Lawyer
      - 👩‍💼 Opposition Lawyer
    - Each lawyer speaks in full sentences (1–3 paragraphs each turn), arguing with clarity and legal grounding.
    - Use realistic tone and references (e.g., “Under Section 420 of BNS…” or “As per Article 21 of the Constitution…”).
    - The Judge should intervene occasionally and finally deliver a verdict or observation at the end.
    - Maintain natural flow like an actual hearing — e.g., Defense presents, Opposition rebuts, Judge concludes.

    Your output format should resemble:
    ---
    👨‍💼 Defense Lawyer: [speaks]

    👩‍💼 Opposition Lawyer: [rebuts]

    🧑‍⚖ Judge: [comments or gives ruling]
    ---

    Now, begin the courtroom simulation for the query.""")

    def create_context_for_simulation(fir_text: str, constitution_text: str, bns_text: str) -> str:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        fir_chunks = splitter.split_text(f"FIR DOCUMENT:\n{fir_text}") if fir_text else []
        laws_context = (
            f"\nINDIAN CONSTITUTION EXTRACTS:\n{constitution_text[:8000]}"
            f"\n\nBNS SECTIONS:\n{bns_text[:8000]}"
        )
        return "\n".join(fir_chunks) + "\n" + laws_context

    def run_simulation(fir_text: str, user_question: str, history: ChatMessageHistory) -> str:
        # Default focus if empty to ensure lengthy generation
        if not user_question or not user_question.strip():
            user_question = "Extensive arguments on Bail Application and validity of arrest"

        # UPDATED: use GPT-4.1 instead of llama3
        agent = get_llm(model_name="gpt-4.1", temperature=0.7)
        context = create_context_for_simulation(fir_text, constitution_text, bns_text)
        formatted_prompt = SIM_PROMPT_TEMPLATE.format(
            context=context,
            history=history.messages if history else [],
            question=user_question
        )
        resp = agent.invoke(formatted_prompt)
        try:
            content = resp.content if hasattr(resp, "content") else (resp[0].text if isinstance(resp, (list, tuple)) else str(resp))
        except Exception:
            content = str(resp)
        return content

    # -------------------------
    # Vector DB builders
    # -------------------------
    @st.cache_resource
    def build_vectordb_for_pdf_and_law(uploaded_file):
        embeddings = get_embeddings()
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_docs = PyPDFLoader(pdf_path).load()
        all_docs = pdf_docs + law_docs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(all_docs)

        # Check if we have documents to embed
        if not chunks:
            st.error("No documents to process.")
            return None

        vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)
        return vectordb

    @st.cache_resource
    def build_vectordb_for_law_only():
        embeddings = get_embeddings()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(law_docs)

        if not chunks:
            return None

        vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)
        return vectordb

    # -------------------------
    # Argument generation (RAG)
    # -------------------------
    ARGUMENT_PROMPT = CoreChatPromptTemplate.from_messages([
        ("system", """You are an experienced senior legal researcher and argument writer.
        Your task is to analyze the given case and produce a comprehensive legal analysis and courtroom-style document.

        You must:
        - Refer to the provided context (from the uploaded case PDF, Constitution of India, and BNS laws).
        - Expand every point thoroughly, writing at least three detailed paragraphs for each heading.
        - Where there are subpoints or bullet points, expand each of them into three coherent paragraphs.
        - Use appropriate legal terminology, structured reasoning, and references to relevant sections, doctrines, or constitutional principles.
        - Write in a formal, persuasive tone, suitable for submission to court or legal counsel.

        Follow this detailed structure:

        1. Client-Side Arguments: - Present the key arguments supporting the client’s case, each with at least three paragraphs.  
           - Include supporting references from BNS sections, constitutional articles, and principles of justice.

        2.  Opponent’s Possible Arguments: - Anticipate at least three counterpoints that the opposing counsel may present.  
           - Expand each with three paragraphs exploring their rationale and potential strength.

        3. Counter-Arguments: - For each opponent argument, provide a detailed rebuttal with at least three paragraphs.  
           - Cite legal reasoning, statutory interpretation, and case precedents when appropriate.

        4. Suggestions & Strategy: - Provide at least three paragraphs suggesting strategic actions, courtroom approaches, or pre-trial considerations.
           - Include references to legal precedents, judicial behavior, and procedural recommendations.

        Be analytical, detailed, and structured. Avoid repetition.  
        The content must read like a professional legal memorandum prepared for court submission.

        Context:
        {context}
        """),
        ("user", "Analyze the uploaded case and generate detailed, expanded arguments, counterarguments, and strategy in full paragraphs under each section(mention the laws and sections and articles if required).")
    ])

    def generate_arguments_rag(vectordb) -> str:
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        # UPDATED: GPT-4.1 instead of llama3
        llm = get_llm(model_name="gpt-4.1", temperature=0.3)
        qa_chain = create_stuff_documents_chain(llm, ARGUMENT_PROMPT)
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        out = rag_chain.invoke({"input": "Generate arguments based on the case"})
        try:
            return out.get("answer") if isinstance(out, dict) else str(out)
        except Exception:
            return str(out)

    # -------------------------
    # General QA: improved prompt & UI
    # -------------------------
    GENERAL_Q_PROMPT = CoreChatPromptTemplate.from_messages([
        ("system", """You are a legal research assistant specialized in the Constitution of India and BNS-like statutes.
        When answering a user's question, follow this structured output EXACTLY:
        1) long (3 - 4 paras)
        2) Legal Basis (list relevant Articles / Sections with short parenthetical explanation)
        3) Explanation (2-3 short paragraphs with legal reasoning)
        4) Practical Implication (1 paragraph)
        5) Sources (list retrieved doc names or identifiers)

        Always ground responses ONLY in the supplied context. If the context doesn't directly state a rule, say "Based on the supplied materials..." and avoid inventing case law.
        Context:
        {context}
        """),
        ("user", "{input}")
    ])

    @st.cache_resource
    def get_general_qa_chain():
        # UPDATED: GPT-4.1 instead of llama3
        llm = get_llm(model_name="gpt-4.1", temperature=0.2)
        qa_chain = create_stuff_documents_chain(llm, GENERAL_Q_PROMPT)

        vdb = build_vectordb_for_law_only()
        if vdb:
            retriever = vdb.as_retriever(search_kwargs={"k": 5})
            rag_chain = create_retrieval_chain(retriever, qa_chain)
            return rag_chain
        else:
            return None

    # -----------------------------
    # UI SECTION (same as before, only text mentions of Llama adjusted)
    # -----------------------------
    @dataclass
    class CaseDoc:
        id: str
        name: str
        size: str
        uploadedAt: str
        type: str

    @dataclass
    class Case:
        id: str
        name: str
        date: str
        documents: List[CaseDoc]

    # -----------------------------
    # Session state initialisation for UI
    # -----------------------------
    def init_ui_state():
        # preserve existing session data where possible
        if "is_dark_mode" not in st.session_state:
            st.session_state.is_dark_mode = True

        if "is_sidebar_collapsed" not in st.session_state:
            st.session_state.is_sidebar_collapsed = False

        if "is_new_case_modal_open" not in st.session_state:
            st.session_state.is_new_case_modal_open = False

        if "show_settings" not in st.session_state:
            st.session_state.show_settings = False

        if "is_logged_in_ui" not in st.session_state:
            # keep linked to authenticator state
            st.session_state.is_logged_in_ui = True

        if "user_details_ui" not in st.session_state:
            st.session_state.user_details_ui = {
                "name": st.session_state.get("name", "John Doe"),
                "email": config.get("credentials", {}).get("users", [{}])[0].get("email", "john.doe@example.com") if config.get("credentials") else "john.doe@example.com",
                "dateOfBirth": "1990-01-01",
                "phoneNumber": "+1 (555) 123-4567",
            }
        if "cases" not in st.session_state:
            st.session_state.cases = []


        if "selected_case_id" not in st.session_state:
            st.session_state.selected_case_id = "case-1"

        # keep chat history for simulation if missing
        if "history" not in st.session_state:
            st.session_state.history = ChatMessageHistory()

        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []

    # -----------------------------
    # Styling (light / dark skin) injection (sample CSS)
    # -----------------------------
    def inject_global_styles():
        if st.session_state.is_dark_mode:
            bg = "#020617"
            main_bg = "#020617"
            text = "#e5e7eb"
            card_bg = "#020617"
            sidebar_bg = "#020617"
            accent = "#6366f1"
            border = "rgba(148, 163, 184, 0.4)"
        else:
            bg = "#f3f4f6"
            main_bg = "#f9fafb"
            text = "#111827"
            card_bg = "#ffffff"
            sidebar_bg = "#ffffff"
            accent = "#1d4ed8"
            border = "rgba(148, 163, 184, 0.5)"

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-color: {bg};
            }}
            .block-container {{
                padding-top: 0.5rem;
                padding-bottom: 0.5rem;
                background-color: {main_bg};
            }}
            section[data-testid="stSidebar"] > div {{
                background-color: {sidebar_bg};
                border-right: 1px solid {border};
            }}
            h1, h2, h3, h4, h5, h6, p, span, label {{
                color: {text} !important;
            }}
            .case-card {{
                background: {card_bg};
                padding: 0.75rem 0.9rem;
                border-radius: 0.75rem;
                margin-bottom: 0.5rem;
                border: 1px solid {border};
            }}
            .case-card.selected {{
                border: 1px solid {accent};
                box-shadow: 0 0 0 1px rgba(99, 102, 241, 0.4);
            }}
            .case-title {{
                font-weight: 600;
                font-size: 0.9rem;
            }}
            .case-date {{
                font-size: 0.75rem;
                opacity: 0.7;
            }}
            .doc-card {{
                background: {card_bg};
                border-radius: 0.9rem;
                padding: 0.9rem 1rem;
                border: 1px solid {border};
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.45);
            }}
            .doc-title {{
                font-weight: 600;
                font-size: 0.9rem;
                margin-bottom: 0.25rem;
            }}
            .doc-meta {{
                font-size: 0.75rem;
                opacity: 0.8;
                margin-bottom: 0.3rem;
            }}
            .pill-tab > button {{
                border-radius: 999px !important;
            }}
            .newcase-card {{
                background: {card_bg};
                border-radius: 1rem;
                padding: 1rem 1.2rem;
                border: 1px solid {border};
                margin-bottom: 1rem;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------
    # UI Helpers: get selected case etc.
    # -----------------------------
    def get_selected_case() -> Optional[Case]:
        for case in st.session_state.cases:
            if case.id == st.session_state.selected_case_id:
                return case
        return None

    def handle_new_case(case_name: str, files):
        docs: List[CaseDoc] = []
        files = files or []
        for index, file in enumerate(files):
            size_bytes = getattr(file, "size", None)
            if size_bytes is None:
                try:
                    file_bytes = file.read()
                    size_bytes = len(file_bytes)
                except Exception:
                    size_bytes = 0
            size_mb = size_bytes / 1024 / 1024 if size_bytes else 0.0
            docs.append(
                CaseDoc(
                    id=f"doc-{uuid4()}-{index}",
                    name=getattr(file, "name", f"file-{index}"),
                    size=f"{size_mb:.1f} MB",
                    uploadedAt="Just now",
                    type=(getattr(file, "name", "") or "").split(".")[-1] if getattr(file, "name", "") else "file",
                )
            )

        new_case = Case(
            id=f"case-{uuid4()}",
            name=case_name,
            date="Just now",
            documents=docs,
        )

        st.session_state.cases = [new_case] + st.session_state.cases
        st.session_state.selected_case_id = new_case.id
        st.session_state.is_new_case_modal_open = False
        try:
            st.toast("New case created")
        except Exception:
            pass
        st.rerun()

    def handle_delete_document(document_id: str):
        updated_cases: List[Case] = []
        for case in st.session_state.cases:
            if case.id == st.session_state.selected_case_id:
                filtered_docs = [d for d in case.documents if d.id != document_id]
                updated_cases.append(Case(id=case.id, name=case.name, date=case.date, documents=filtered_docs))
            else:
                updated_cases.append(case)
        st.session_state.cases = updated_cases
        st.rerun()

    # -----------------------------
    # Sidebar (sample)
    # -----------------------------
    def render_sidebar():
        with st.sidebar:
            st.markdown("### ⚖️ Legal Case Workspace")

            col1, col2 = st.columns(2)
            with col1:
                dark_val = st.toggle("Dark mode", value=st.session_state.is_dark_mode)
                st.session_state.is_dark_mode = dark_val

            with col2:
                if st.button("Collapse" if not st.session_state.is_sidebar_collapsed else "Expand", use_container_width=True):
                    st.session_state.is_sidebar_collapsed = not st.session_state.is_sidebar_collapsed
                    st.rerun()

            st.markdown("---")

            # User / auth (uses authenticator / session_state)
            if st.session_state.is_logged_in_ui:
                user = st.session_state.user_details_ui
                st.markdown(f"*Signed in as*  \n{user['name']}  \n<small>{user['email']}</small>", unsafe_allow_html=True)
                if st.button("Settings", use_container_width=True):
                    st.session_state.show_settings = True
                    st.rerun()

                if st.button("Logout", use_container_width=True):
                    st.session_state.is_logged_in_ui = False
                    st.success("Logged out (UI)")
                    st.rerun()
            else:
                st.info("You are not logged in.")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Login", use_container_width=True):
                        st.session_state.is_logged_in_ui = True
                        st.success("Logged in (demo)")
                        st.rerun()
                with c2:
                    if st.button("Sign up", use_container_width=True):
                        try:
                            st.toast("Signup clicked (demo only)")
                        except Exception:
                            st.info("Signup clicked (demo only)")

            st.markdown("---")

            # New Case
            if st.button("➕ New Case", use_container_width=True):
                st.session_state.is_new_case_modal_open = True

            st.markdown("### Cases")

            if st.session_state.is_sidebar_collapsed:
                # Minimal view: just numbered buttons
                for idx, case in enumerate(st.session_state.cases, start=1):
                    if st.button(f"{idx}. {case.name[:16]}…", key=f"collapsed_case_{case.id}", use_container_width=True):
                        st.session_state.selected_case_id = case.id
                        st.rerun()
            else:
                # Full case cards
                for case in st.session_state.cases:
                    selected = case.id == st.session_state.selected_case_id
                    css_class = "case-card selected" if selected else "case-card"
                    st.markdown(f"""<div class="{css_class}"><div class="case-title">{case.name}</div><div class="case-date">{case.date}</div></div>""", unsafe_allow_html=True)
                    if st.button("Open", key=f"open_{case.id}", use_container_width=True):
                        st.session_state.selected_case_id = case.id
                        st.rerun()

    # -----------------------------
    # "Modal-like" New Case form (top card)
    # -----------------------------
    def render_new_case_card():
        if not st.session_state.is_new_case_modal_open:
            return

        st.markdown("### 🆕 Create New Case", help="Name your case and attach initial documents.")
        with st.container():
            st.markdown('<div class="newcase-card">', unsafe_allow_html=True)
            with st.form("new_case_form", clear_on_submit=False):
                case_name = st.text_input("Case Name", placeholder="Enter case name...")
                uploaded_files = st.file_uploader(
                    "Add Documents",
                    type=None,
                    accept_multiple_files=True,
                    help="Click to upload or drag files"
                )

                col1, col2 = st.columns(2)
                with col1:
                    submitted = st.form_submit_button("Create Case")
                with col2:
                    cancel = st.form_submit_button("Cancel")

                if cancel:
                    st.session_state.is_new_case_modal_open = False
                    st.rerun()

                if submitted:
                    if not case_name or not case_name.strip():
                        st.error("Case name is required.")
                    else:
                        handle_new_case(case_name.strip(), uploaded_files or [])

            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

    # -----------------------------
    # Documents tab content
    # -----------------------------
    def render_documents_tab(selected_case: Case):
        st.markdown("### Case Documents")
        st.caption(f"{len(selected_case.documents)} documents uploaded")

        docs = selected_case.documents
        if not docs:
            st.info("No documents uploaded for this case yet.")
            return

        cols_per_row = 3
        for i in range(0, len(docs), cols_per_row):
            row_docs = docs[i : i + cols_per_row]
            cols = st.columns(len(row_docs))
            for col, doc in zip(cols, row_docs):
                with col:
                    st.markdown('<div class="doc-card">', unsafe_allow_html=True)
                    st.markdown(f"""<div class="doc-title">{doc.name}</div><div class="doc-meta">{doc.size} &nbsp;·&nbsp; Uploaded {doc.uploadedAt}</div>""", unsafe_allow_html=True)

                    b1, b2, b3 = st.columns(3)
                    with b1:
                        st.button("Preview", key=f"preview_{doc.id}", use_container_width=True)
                    with b2:
                        st.button("⬇Download", key=f"download_{doc.id}", use_container_width=True)
                    with b3:
                        if st.button("delete", key=f"delete_{doc.id}", use_container_width=True):
                            handle_delete_document(doc.id)
                            st.stop()

                    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Main panel (tabs) integrating existing flows
    # -----------------------------
    def render_main_panel():
        selected_case = get_selected_case()
        case_name = selected_case.name if selected_case else "Select a case"

        # Header
        top_cols = st.columns([3, 1])
        with top_cols[0]:
            st.markdown(f"## {case_name}")
            if selected_case:
                st.caption(f"Opened on: {selected_case.date}")
            else:
                st.caption("Choose a case from the sidebar to get started.")

        with top_cols[1]:
            if st.button("📁 Collapse Sidebar" if not st.session_state.is_sidebar_collapsed else "📂 Expand Sidebar", use_container_width=True):
                st.session_state.is_sidebar_collapsed = not st.session_state.is_sidebar_collapsed

        # New case card (appears under the header when + New Case clicked)
        render_new_case_card()

        if not selected_case:
            return

        # Tabs row – Argument, Q&A, Courtroom, Documents
        tabs = st.tabs(["Argument Generator", "Q&A System", "Courtroom Simulation", "Documents"])

        # ---- Argument Generator tab (integrated with your RAG)
        with tabs[0]:
            st.markdown("### Argument Generator")
            st.write("Upload a case PDF to generate an extended legal memorandum for this case (RAG).")
            uploaded_case = st.file_uploader("Upload case document (PDF)", type=["pdf"], key="ui_arg_upload")
            gen_args_btn = st.button("Generate Arguments", key="ui_generate_args")
            if gen_args_btn:
                if not uploaded_case:
                    st.warning("Please upload a PDF.")
                else:
                    with st.spinner("Building index and generating arguments with GPT-4.1..."):
                        try:
                            vectordb = build_vectordb_for_pdf_and_law(uploaded_case)
                            if vectordb:
                                ans = generate_arguments_rag(vectordb)
                                st.success("Arguments generated.")
                                st.markdown(ans)
                            else:
                                st.error("Failed to build vector database.")
                        except Exception as e:
                            st.error(f"Error: {e}")

        # ---- Q&A System tab
        with tabs[1]:
            st.markdown("### Q&A System")
            st.write("Ask focused questions about this case or related laws.")
            user_q = st.text_input("Ask a legal question (e.g., 'What are advantages of the Indian Constitution?')", key="ui_general_q")
            ask_btn = st.button("Ask", key="ui_ask_btn")
            if ask_btn:
                if not user_q or not user_q.strip():
                    st.warning("Please type a question.")
                else:
                    with st.spinner("Searching law corpus and composing structured answer..."):
                        try:
                            rag_chain = get_general_qa_chain()
                            if rag_chain:
                                response = rag_chain.invoke({"input": user_q})
                                if isinstance(response, dict) and response.get("answer"):
                                    ans = response.get("answer")
                                else:
                                    ans = str(response)
                                st.session_state.qa_history.append({"q": user_q, "a": ans})
                                st.markdown("### Answer")
                                st.markdown(ans)
                                with st.expander("Show law excerpts used (best-effort)"):
                                    st.markdown("Displayed excerpts are the top documents from the law corpus used for retrieval.")
                                    try:
                                        vdb_law = build_vectordb_for_law_only()
                                        if vdb_law:
                                            retriever = vdb_law.as_retriever(search_kwargs={"k": 5})
                                            docs = retriever.invoke(user_q)
                                            for i, d in enumerate(docs[:5], 1):
                                                src = d.metadata.get("source", "law_corpus")
                                                snippet = (d.page_content[:600] + "...") if len(d.page_content) > 600 else d.page_content
                                                st.markdown(f"{i}. Source:** {src}")
                                                st.code(snippet)
                                    except Exception as e:
                                        st.info(f"Source excerpts not available: {e}")
                            else:
                                st.error("Could not initialize QA chain (data missing?).")
                        except Exception as e:
                            st.error(f"Error answering QA: {e}")

            # show QA history (compact)
            if st.session_state.qa_history:
                st.markdown("---")
                st.markdown("#### Recent QA")
                for item in reversed(st.session_state.qa_history[-6:]):
                    st.markdown(f"Q: {item['q']}")
                    st.markdown(f"A: {item['a'][:800]}{'...' if len(item['a'])>800 else ''}")
                    st.write("")

        # ---- Courtroom Simulation tab
        with tabs[2]:
            st.markdown("### Courtroom Simulation")
            st.write("Upload an FIR to simulate a courtroom dialogue (Judge / Defense / Opposition).")
            uploaded_fir = st.file_uploader("Upload FIR (PDF or TXT)", type=["pdf", "txt"], key="ui_fir_upload")
            case_question = st.text_input("Simulation Focus (Optional)", placeholder="Leave blank for automatic Bail/Charge hearing simulation", key="ui_sim_focus")
            simulate_btn = st.button("Simulate Court Trial", key="ui_simulate_btn")
            if simulate_btn:
                if not uploaded_fir:
                    st.warning("Please upload an FIR first.")
                else:
                    with st.spinner("Generating realistic courtroom dialogue (this may take a moment)..."):
                        fir_text = load_fir_text(uploaded_fir)
                        try:
                            sim_text = run_simulation(fir_text, case_question, st.session_state.history)
                            st.success("Simulation complete.")
                            st.markdown("### Courtroom Proceedings")
                            st.write(sim_text)
                            try:
                                st.session_state.history.add_user_message("Start Simulation")
                                st.session_state.history.add_ai_message(sim_text)
                            except Exception:
                                pass
                        except Exception as e:
                            st.error(f"Simulation error: {e}")

            with st.expander("Law excerpts (Constitution / BNS)"):
                st.markdown("Constitution (excerpt)")
                st.code(constitution_text[:1800])
                st.markdown("BNS (excerpt)")
                st.code(bns_text[:1800])

        # ---- Documents tab
        with tabs[3]:
            render_documents_tab(selected_case)

    # -----------------------------
    # Settings page (kept from sample UI)
    # -----------------------------
    def render_settings_page():
        st.markdown("##Account Settings")
        user = st.session_state.user_details_ui
        back_col, _ = st.columns([1, 3])
        with back_col:
            if st.button("⬅️ Back to cases"):
                st.session_state.show_settings = False
                st.rerun()

        st.markdown("---")

        tab1, tab2 = st.tabs(["Profile", "Security"])

        with tab1:
            st.markdown("### Profile Details")
            with st.form("profile_form"):
                name = st.text_input("Full name", value=user["name"])
                email = st.text_input("Email", value=user["email"])
                dob = st.date_input("Date of birth", value=datetime.fromisoformat(user["dateOfBirth"]))
                phone = st.text_input("Phone number", value=user["phoneNumber"])

                submitted = st.form_submit_button("Save changes")
                if submitted:
                    st.session_state.user_details_ui = {
                        "name": name,
                        "email": email,
                        "dateOfBirth": dob.isoformat(),
                        "phoneNumber": phone,
                    }
                    st.success("Profile updated")

        with tab2:
            st.markdown("### Change password")
            with st.form("password_form"):
                old_pwd = st.text_input("Old password", type="password")
                new_pwd = st.text_input("New password", type="password")
                confirm = st.text_input("Confirm new password", type="password")

                submitted_pwd = st.form_submit_button("Update password")
                if submitted_pwd:
                    if not old_pwd or not new_pwd:
                        st.error("All fields are required.")
                    elif new_pwd != confirm:
                        st.error("New passwords do not match.")
                    else:
                        st.success("Password changed (demo only).")

            st.markdown("---")
            st.markdown("### Danger zone")
            if st.button(" Delete account (demo)", type="secondary"):
                st.warning("Account deleted (demo). You are now logged out.")
                st.session_state.is_logged_in_ui = False
                st.session_state.show_settings = False
                st.rerun()

    # -----------------------------
    # Main entry for UI rendering
    # -----------------------------
    init_ui_state()
    inject_global_styles()

    if st.session_state.show_settings and st.session_state.is_logged_in_ui:
        render_settings_page()
    else:
        render_sidebar()
        render_main_panel()

    # -------------------------
    # Footer/disclaimer (kept)
    # -------------------------
    st.markdown("---")
    st.caption("⚠ Disclaimer: This tool provides simulations and research assistance. It is not a substitute for professional legal advice.")
elif st.session_state.get("authentication_status") is False:
    st.error('Username/password is incorrect')
