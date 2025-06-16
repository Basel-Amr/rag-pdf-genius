import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv
import PyPDF2
import uuid

# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class SimpleModelSelector:
    """Simple class to handle model selection"""

    def __init__(self):
        # Available LLM models
        self.llm_models = {"openai": "GPT-4", "ollama": "Llama3"}

        # Available embedding models with their dimensions
        self.embedding_models = {
            "openai": {
                "name": "OpenAI Embeddings",
                "dimensions": 1536,
                "model_name": "text-embedding-3-small",
            },
            "chroma": {"name": "Chroma Default", "dimensions": 384, "model_name": None},
            "nomic": {
                "name": "Nomic Embed Text",
                "dimensions": 768,
                "model_name": "nomic-embed-text",
            },
        }

    def select_models(self):
        """Let user select models through Streamlit UI"""
        st.sidebar.title("📚 Model Selection")

        # Select LLM
        llm = st.sidebar.radio(
            "Choose LLM Model:",
            options=list(self.llm_models.keys()),
            format_func=lambda x: self.llm_models[x],
        )

        # Select Embeddings
        embedding = st.sidebar.radio(
            "Choose Embedding Model:",
            options=list(self.embedding_models.keys()),
            format_func=lambda x: self.embedding_models[x]["name"],
        )

        return llm, embedding


class SimplePDFProcessor:
    """Handle PDF processing and chunking"""

    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, pdf_file):
        """Read PDF and extract text"""
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def create_chunks(self, text, pdf_file):
        """Split text into chunks"""
        chunks = []
        start = 0

        while start < len(text):
            # Find end of chunk
            end = start + self.chunk_size

            # If not at the start, include overlap
            if start > 0:
                start = start - self.chunk_overlap

            # Get chunk
            chunk = text[start:end]

            # Try to break at sentence end
            if end < len(text):
                last_period = chunk.rfind(".")
                if last_period != -1:
                    chunk = chunk[: last_period + 1]
                    end = start + last_period + 1

            chunks.append(
                {
                    "id": str(uuid.uuid4()),  # cdefield24482kuy
                    "text": chunk,
                    "metadata": {"source": pdf_file.name},
                }
            )

            start = end

        return chunks


class SimpleRAGSystem:
    """Simple RAG implementation"""

    def __init__(self, embedding_model="openai", llm_model="openai"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Initialize ChromaDB
        self.db = chromadb.PersistentClient(path="./chroma_db")

        # Setup embedding function based on model
        self.setup_embedding_function()

        # Setup LLM
        if llm_model == "openai":
            self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY_2"))
        else:
            self.llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

        # Get or create collection with proper handling
        self.collection = self.setup_collection()

    def setup_embedding_function(self):
        """Setup the appropriate embedding function"""
        try:
            if self.embedding_model == "openai":
                self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY_2"),
                    model_name="text-embedding-3-small",
                )
            elif self.embedding_model == "nomic":
                # For Nomic embeddings via Ollama
                self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key="ollama",
                    api_base="http://localhost:11434/v1",
                    model_name="nomic-embed-text",
                )

                # Alternative if needed:
                # self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                #     model_name="all-MiniLM-L6-v2"
                # )
            else:  # chroma default
                self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            st.error(f"Error setting up embedding function: {str(e)}")
            raise e

    def setup_collection(self):
        """Setup collection with proper dimension handling"""
        collection_name = f"documents_{self.embedding_model}"

        try:
            # Try to get existing collection first
            try:
                collection = self.db.get_collection(
                    name=collection_name, embedding_function=self.embedding_fn
                )
                st.info(
                    f"Using existing collection for {self.embedding_model} embeddings"
                )
            except:
                # If collection doesn't exist, create new one
                collection = self.db.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"model": self.embedding_model},
                )
                st.success(
                    f"Created new collection for {self.embedding_model} embeddings"
                )

            return collection

        except Exception as e:
            st.error(f"Error setting up collection: {str(e)}")
            raise e

    def add_documents(self, chunks):
        """Add documents to ChromaDB"""
        try:
            # Ensure collection exists
            if not self.collection:
                self.collection = self.setup_collection()

            # Add documents
            self.collection.add(
                ids=[chunk["id"] for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks],
            )
            return True
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False

    def query_documents(self, query, n_results=3):
        """Query documents and return relevant chunks"""
        try:
            # Ensure collection exists
            if not self.collection:
                raise ValueError("No collection available")

            results = self.collection.query(query_texts=[query], n_results=n_results)
            return results
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None

    def generate_response(self, query, context):
        """Generate response using LLM"""
        try:
            prompt = f"""
            Based on the following context, please answer the question.
            If you can't find the answer in the context, say so, or I don't know.

            Context: {context}

            Question: {query}

            Answer:
            """

            response = self.llm.chat.completions.create(
                model="gpt-4o-mini" if self.llm_model == "openai" else "llama3.2",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )

            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None

    def get_embedding_info(self):
        """Get information about current embedding model"""
        model_selector = SimpleModelSelector()
        model_info = model_selector.embedding_models[self.embedding_model]
        return {
            "name": model_info["name"],
            "dimensions": model_info["dimensions"],
            "model": self.embedding_model,
        }


def main():
    st.set_page_config(page_title="📄 RAG PDF Assistant", layout="wide")

    st.markdown("""
        <div style='display: flex; align-items: center; justify-content: space-between;'>
            <div style='display: flex; align-items: center;'>
                <img src='https://cdn-icons-png.flaticon.com/512/337/337946.png' width='60'>
                <h1 style='padding-left: 15px;'>PDF Question Answering RAG System</h1>
            </div>
            <img src='https://media.giphy.com/media/KGhpQH1Zi2CLk/giphy.gif' width='80'>
        </div>
        <hr style="border: 1px solid #bbb;">
    """, unsafe_allow_html=True)

    # Session Initialization
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = None
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # Sidebar: Model Selection
    st.sidebar.markdown("### ⚙️ Model Settings")
    model_selector = SimpleModelSelector()
    llm_model, embedding_model = model_selector.select_models()

    # Handle Model Change
    if embedding_model != st.session_state.current_embedding_model:
        st.session_state.processed_files.clear()
        st.session_state.current_embedding_model = embedding_model
        st.session_state.rag_system = None
        st.warning("⚠️ Embedding model changed. Please re-upload your documents.")

    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = SimpleRAGSystem(embedding_model, llm_model)

        # Display Embedding Info
        embedding_info = st.session_state.rag_system.get_embedding_info()
        st.sidebar.info(
            f"\n**Model:** {embedding_info['name']}\n\n**Dimensions:** {embedding_info['dimensions']}"
        )

    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return

    # File Upload Section
    st.markdown("## 📂 Upload PDF Document")
    pdf_file = st.file_uploader("Upload your PDF", type="pdf")

    if pdf_file and pdf_file.name not in st.session_state.processed_files:
        processor = SimplePDFProcessor()
        with st.spinner("🔍 Extracting and indexing content..."):
            try:
                progress = st.progress(0)
                text = processor.read_pdf(pdf_file)
                progress.progress(25)

                chunks = processor.create_chunks(text, pdf_file)
                progress.progress(60)

                added = st.session_state.rag_system.add_documents(chunks)
                progress.progress(100)

                if added:
                    st.session_state.processed_files.add(pdf_file.name)
                    st.success(f"✅ Successfully processed `{pdf_file.name}`")
            except Exception as e:
                st.error(f"❌ Failed to process PDF: {str(e)}")

    # Query Section
    if st.session_state.processed_files:
        st.markdown("---")
        st.markdown("## 💬 Ask Questions from Your PDFs")

        query = st.text_input("Type your question:", placeholder="e.g., What is the main topic of section 2?")

        if query:
            with st.spinner("🧠 Generating AI response..."):
                results = st.session_state.rag_system.query_documents(query)

                if results and results["documents"]:
                    response = st.session_state.rag_system.generate_response(
                        query, results["documents"][0]
                    )

                    if response:
                        st.markdown(f"""
                            <div style="background-color:#111827; color:#39ff14; padding:20px; border-radius:10px;
                                        font-family: monospace; font-size: 16px; white-space: pre-wrap;
                                        word-wrap: break-word; max-width: 100%;">
                                <strong>🤖 AI Response:</strong><br><br>
                                <span class="typewriter">{response}</span>
                            </div>
                            <style>
                                .typewriter {{
                                    overflow: hidden;
                                    border-right: .15em solid orange;
                                    animation: typing 2s steps(40, end), blink-caret .75s step-end infinite;
                                }}
                                @keyframes typing {{
                                    from {{ width: 0 }}
                                    to {{ width: 100% }}
                                }}
                                @keyframes blink-caret {{
                                    from, to {{ border-color: transparent }}
                                    50% {{ border-color: orange }}
                                }}
                            </style>
                        """, unsafe_allow_html=True)

                        with st.expander("📚 Relevant Source Passages"):
                            for i, doc in enumerate(results["documents"][0], 1):
                                st.markdown(f"**🔸 Passage {i}:**")
                                st.info(doc)

                        # Save history
                        st.session_state.qa_history.insert(0, (query, response))

        # Display QA History
        if st.session_state.qa_history:
            st.markdown("---")
            st.markdown("## 🕓 Question History")

            for idx, (q, a) in enumerate(st.session_state.qa_history):
                with st.expander(f"Q{idx + 1}: {q}"):
                    st.markdown(f"**💬 Answer:**\n\n{a}")

    else:
        st.info("👆 Please upload a PDF document to begin.")
    
    st.markdown("""
        <hr>
        <div style='text-align: center;'>
            <p>🚀 Built with ❤️ by <strong>Basel Amr Barakat</strong></p>
            <p>
                <a href='https://www.linkedin.com/in/baselamrbarakat' target='_blank'>
                    <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='25' style='vertical-align:middle;'/> LinkedIn
                </a>
                &nbsp;&nbsp;&nbsp;
                <a href='https://github.com/Basel-Amr' target='_blank'>
                    <img src='https://cdn-icons-png.flaticon.com/512/25/25231.png' width='25' style='vertical-align:middle;'/> GitHub
                </a>
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
