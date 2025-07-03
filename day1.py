# Fix SQLite version issue for ChromaDB on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
import streamlit as st
import chromadb
from transformers import pipeline
from pathlib import Path
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice


from datetime import datetime

# --- Custom CSS for a holistic, healthy, friendly look ---
def add_custom_css():
    st.markdown("""
    <style>
    html, body, .stApp {
        min-height: 100vh;
        background: linear-gradient(120deg, #ffe0f0 0%, #fce4ec 40%, #f8bbd0 70%, #fff1fa 100%) !important;
        font-family: 'Segoe UI', 'Arial Rounded MT Bold', Arial, sans-serif;
        position: relative;
    }
    /* Girly emoji and sparkly overlay using SVG as background image */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: 0;
        opacity: 0.13;
        pointer-events: none;
        background: url('data:image/svg+xml;utf8,<svg width="480" height="480" xmlns="http://www.w3.org/2000/svg"><text x="30" y="80" font-size="48">📚</text><text x="320" y="120" font-size="44">💖</text><text x="120" y="400" font-size="40">🦄</text><text x="400" y="300" font-size="44">✨</text><text x="200" y="200" font-size="38">🦋</text><text x="100" y="200" font-size="38">🌸</text><text x="350" y="400" font-size="38">💡</text></svg>');
        background-repeat: repeat;
        background-size: 480px 480px;
    }
    .main-header {
        font-size: 2.8rem;
        color: #d81b60;
        text-align: center;
        margin-bottom: 2.2rem;
        padding: 1.2rem;
        background: linear-gradient(90deg, #fce4ec, #fff1fa);
        border-radius: 18px;
        box-shadow: 0 2px 12px rgba(216, 27, 96, 0.08);
        letter-spacing: 1px;
    }
    .success-box {
        padding: 1rem;
        background-color: #f8bbd0;
        border: 1px solid #fce4ec;
        border-radius: 8px;
        margin: 1rem 0;
        color: #ad1457;
        font-size: 1.1rem;
    }
    .info-box {
        padding: 1rem;
        background-color: #fce4ec;
        border: 1px solid #f8bbd0;
        border-radius: 8px;
        margin: 1rem 0;
        color: #d81b60;
        font-size: 1.1rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #f8bbd0, #fff1fa);
        color: #d81b60;
        border: none;
        font-size: 1.1rem;
        box-shadow: 0 1px 4px rgba(216, 27, 96, 0.08);
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #fff1fa, #f8bbd0);
        color: #ad1457;
    }
    .metric-card {
        background: #fff;
        padding: 1.2rem;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(216, 27, 96, 0.07);
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        color: #d81b60;
        background: #fce4ec;
        border-radius: 8px 8px 0 0;
        margin-right: 0.5rem;
        padding: 0.7rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: #fff1fa;
        color: #ad1457;
    }
    .st-expanderHeader {
        font-size: 1.05rem;
        color: #d81b60;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Pastel background CSS
pastel_bg_css = """
<style>
body, .stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #e0f7fa 50%, #fce4ec 100%);
}
.stApp {
    background-color: #f8fafc !important;
}
</style>
"""


# Convert uploaded file to markdown text
def convert_to_markdown(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        pdf_opts = PdfPipelineOptions(do_ocr=False)
        pdf_opts.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_opts,
                    backend=DoclingParseV2DocumentBackend
                )
            }
        )
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext in [".doc", ".docx"]:
        converter = DocumentConverter()
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")

    raise ValueError(f"Unsupported extension: {ext}")


# Reset ChromaDB collection
def reset_collection(client, collection_name: str):
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    return client.create_collection(name=collection_name)


# Add text chunks to ChromaDB
def add_text_to_chromadb(text: str, filename: str, collection_name: str = "documents"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)

    if not hasattr(add_text_to_chromadb, 'client'):
        add_text_to_chromadb.client = chromadb.Client()
        add_text_to_chromadb.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        add_text_to_chromadb.collections = {}

    if collection_name not in add_text_to_chromadb.collections:
        try:
            collection = add_text_to_chromadb.client.get_collection(name=collection_name)
        except:
            collection = add_text_to_chromadb.client.create_collection(name=collection_name)
        add_text_to_chromadb.collections[collection_name] = collection

    collection = add_text_to_chromadb.collections[collection_name]

    for i, chunk in enumerate(chunks):
        embedding = add_text_to_chromadb.embedding_model.encode(chunk).tolist()

        metadata = {
            "filename": filename,
            "chunk_index": i,
            "chunk_size": len(chunk)
        }

        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[metadata],
            ids=[f"{filename}_chunk_{i}"]
        )

    return collection



# Q&A function with source tracking
def get_answer_with_source(collection, question):
    results = collection.query(query_texts=[question], n_results=3)
    docs = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0] if "ids" in results else ["unknown"] * len(docs)

    if not docs or min(distances) > 1.5:
        return "I don't have information about that topic in my documents.", "No source"

    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with 'I don't know.' Do not add information from outside the context.

Answer:"""

    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(prompt, max_length=150)
    answer = response[0]['generated_text'].strip()
    # Extract source from best matching document
    best_source = ids[0].split('_chunk_')[0] if ids else "unknown"
    return answer, best_source

# Search history feature
def add_to_search_history(question, answer, source):
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    st.session_state.search_history.insert(0, {
        'question': question,
        'answer': answer,
        'source': source,
        'timestamp': str(datetime.now().strftime("%H:%M:%S"))
    })
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[:10]

def show_search_history():
    st.subheader("🕒 Recent Searches")
    if 'search_history' not in st.session_state or not st.session_state.search_history:
        st.info("No searches yet.")
        return
    for i, search in enumerate(st.session_state.search_history):
        with st.expander(f"Q: {search['question'][:50]}... ({search['timestamp']})"):
            st.write("**Question:**", search['question'])
            st.write("**Answer:**", search['answer'])
            st.write("**Source:**", search['source'])

# Document manager with delete and preview
def show_document_manager():
    st.subheader("📋 Manage Documents")
    if not st.session_state.get('converted_docs', []):
        st.info("No documents uploaded yet.")
        return
    for i, doc in enumerate(st.session_state.converted_docs):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"📄 {doc['filename']}")
            st.write(f"   Words: {len(doc['content'].split())}")
        with col2:
            if st.button("Preview", key=f"preview_{i}"):
                st.session_state[f'show_preview_{i}'] = True
        with col3:
            if st.button("Delete", key=f"delete_{i}"):
                st.session_state.converted_docs.pop(i)
                # Rebuild database
                st.session_state.collection = reset_collection(st.session_state.client, "documents")
                for d in st.session_state.converted_docs:
                    add_text_to_chromadb(d['content'], d['filename'], collection_name="documents")
                st.rerun()
        if st.session_state.get(f'show_preview_{i}', False):
            with st.expander(f"Preview: {doc['filename']}", expanded=True):
                st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                if st.button("Hide Preview", key=f"hide_{i}"):
                    st.session_state[f'show_preview_{i}'] = False
                    st.rerun()

# Document statistics
def show_document_stats():
    st.subheader("📊 Document Statistics")
    if not st.session_state.get('converted_docs', []):
        st.info("No documents to analyze.")
        return
    total_docs = len(st.session_state.converted_docs)
    total_words = sum(len(doc['content'].split()) for doc in st.session_state.converted_docs)
    avg_words = total_words // total_docs if total_docs > 0 else 0
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", total_docs)
    with col2:
        st.metric("Total Words", f"{total_words:,}")
    with col3:
        st.metric("Average Words/Doc", f"{avg_words:,}")
    file_types = {}
    for doc in st.session_state.converted_docs:
        ext = Path(doc['filename']).suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
    st.write("**File Types:**")
    for ext, count in file_types.items():
        st.write(f"• {ext}: {count} files")

# Helper: convert uploaded files to markdown and store in session
def convert_uploaded_files(uploaded_files):
    converted_docs = []
    for file in uploaded_files:
        suffix = Path(file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name
        text = convert_to_markdown(temp_file_path)
        converted_docs.append({
            'filename': file.name,
            'content': text
        })
    return converted_docs

# Helper: add docs to database
def add_docs_to_database(collection, docs):
    count = 0
    for doc in docs:
        add_text_to_chromadb(doc['content'], doc['filename'], collection_name="documents")
        count += 1
    return count

# --- Enhanced, holistic, user-friendly UI with tabs ---
def create_tabbed_interface():
    tab1, tab2, tab3, tab4 = st.tabs(["🌸 Upload", "💖 Questions", "📋 Manage", "📊 Stats"])
    with tab1:
        st.header("🌸 Upload & Convert Notes")
        uploaded_files = st.file_uploader(
            "🌸 Upload your IMB notes, readings, or files",
            type=["pdf", "doc", "docx", "txt"],
            accept_multiple_files=True,
            help="Supported: PDF, Word, and text files"
        )
        if st.button("💾 Add to Knowledge Base", type="primary"):
            if uploaded_files:
                with st.spinner("Organizing your notes with love..."):
                    converted_docs = convert_uploaded_files(uploaded_files)
                if 'converted_docs' not in st.session_state:
                    st.session_state.converted_docs = []
                if 'client' not in st.session_state:
                    st.session_state.client = chromadb.Client()
                if 'collection' not in st.session_state:
                    st.session_state.collection = reset_collection(st.session_state.client, "documents")
                num_added = add_docs_to_database(st.session_state.collection, converted_docs)
                st.session_state.converted_docs.extend(converted_docs)
                st.success(f"🌸 Added {num_added} notes to your IMB Knowledge Base!")
            else:
                st.info("Please select files to upload first.")
    with tab2:
        st.header("💖 Ask Anything About Your Notes")
        if st.session_state.get('converted_docs', []):
            question, search_button, clear_button = enhanced_question_interface()
            if search_button and question:
                with st.spinner("Thinking and searching for you..."):
                    answer, source = get_answer_with_source(st.session_state.collection, question)
                st.markdown("### ✨ Your Personalized Answer")
                st.write(answer)
                st.info(f"📄 Source: {source}")
                add_to_search_history(question, answer, source)
            if clear_button:
                st.session_state.search_history = []
                st.success("Search history cleared!")
            if st.session_state.search_history:
                show_search_history()
        else:
            st.info("🌸 Upload some notes first to start your Q&A journey!")
    with tab3:
        show_document_manager()
    with tab4:
        show_document_stats()
    st.markdown("""
    <div style='text-align:center; margin-top:2.5rem; color:#d81b60; font-size:1.1rem;'>
        💖 <b>Blanka, you are building your future one note at a time!</b> 💖
    </div>
    """, unsafe_allow_html=True)


# MAIN APP
def enhanced_question_interface():
    st.subheader("💬 Ask Your Question")
    with st.expander("💡 Example questions you can ask"):
        st.write("Ask about anything you have learned during your IMB studies. 🥰")
    question = st.text_input(
        "Type your question here:",
        placeholder="e.g., What are the main findings in my research notes?"
    )
    col1, col2 = st.columns([1, 1])
    # The following lines must NOT be indented further than this line
    with col1:
        search_button = st.button("🔍 Search Notes", type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear History")
    return question, search_button, clear_button

def main():
    add_custom_css()
    st.markdown('<h1 class="main-header">🌸 Blanka\'s Personal IMB Knowledge Base 🌸</h1>', unsafe_allow_html=True)
    st.markdown("Upload your notes, organize your academic year, and ask anything! Your personal assistant is here for you 💖")
    if 'converted_docs' not in st.session_state:
        st.session_state.converted_docs = []
    if 'client' not in st.session_state:
        st.session_state.client = chromadb.Client()
    if 'collection' not in st.session_state:
        st.session_state.collection = reset_collection(st.session_state.client, "documents")
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    create_tabbed_interface()

if __name__ == "__main__":
    main()
