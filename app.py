import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
import chromadb
from collections import namedtuple
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# --- Step 1: Configuration for local models ---
# Set the global LlamaIndex settings to use your local Ollama and Hugging Face models
Settings.llm = Ollama(model="llama3")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
# Note: The embedding model will be downloaded automatically the first time you run this.

# Define a simple named tuple for form fields for clarity
FormField = namedtuple("FormField", ["name", "label", "prompt"])

# --- Step 2: RAG Indexing and retrieval ---
@st.cache_resource
def create_rag_index(documents_path: str, persist_dir: str = "./chroma_db"):
    """Loads documents, creates embeddings using a local model, and builds a vector store index."""
    if not os.path.exists(documents_path) or not os.listdir(documents_path):
        st.error(f"No documents found in {documents_path}.")
        st.stop()
    
    documents = SimpleDirectoryReader(documents_path).load_data()
    
    # Initialize ChromaDB client and collection
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection("local_rag_form_data")
    
    # Set up the vector store and storage context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    with st.spinner("Indexing documents..."):
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    st.success("Indexing complete.")
    
    return index

def fill_form_field(query_engine, form_field: FormField) -> str:
    """Uses the RAG query engine to fill a specific form field."""
    prompt_text = f"Provide a short, direct answer for '{form_field.name}' based on the document text. The question is: {form_field.prompt}"
    response = query_engine.query(prompt_text)
    return str(response).strip()

def validate_form_logically(query_engine, form_data: dict) -> dict:
    """Performs logical validation on the completed form data."""
    validation_results = {}
    
    # Example validation 1: Is the "Client Status" consistent with the "Renewal Date"?
    if "client_status" in form_data and "renewal_date" in form_data:
        status = form_data["client_status"]
        date = form_data["renewal_date"]
        validation_query = f"Based on the provided documents, is it logically consistent for a client with a status of '{status}' to have a renewal date of '{date}'?"
        validation_response = query_engine.query(validation_query)
        
        if "yes" in str(validation_response).lower() and "not consistent" not in str(validation_response).lower():
            validation_results["renewal_date_status_consistency"] = {"result": True, "details": "Validated by RAG."}
        else:
            validation_results["renewal_date_status_consistency"] = {"result": False, "details": str(validation_response)}
    
    # Example validation 2: Is the product code correct?
    if "product_code" in form_data:
        code = form_data["product_code"]
        validation_query = f"Based on the documents, is '{code}' a valid product code?"
        validation_response = query_engine.query(validation_query)
        
        if "yes" in str(validation_response).lower():
            validation_results["product_code_validity"] = {"result": True, "details": "Validated by RAG."}
        else:
            validation_results["product_code_validity"] = {"result": False, "details": str(validation_response)}
            
    return validation_results

# --- Step 3: Streamlit UI and application logic ---
st.title("Local RAG-Powered Form Filler")
st.markdown("Use your local documents to automatically fill and validate a form.")

# Initialize the RAG engine
documents_path = "./documents"
index = create_rag_index(documents_path)
query_engine = index.as_query_engine()

# Define the form fields
form_fields = [
    FormField("client_name", "Client Name", "What is the client's name?"),
    FormField("client_status", "Client Status", "What is the client's status?"),
    FormField("subscription_plan", "Subscription Plan", "What is the client's subscription plan?"),
    FormField("renewal_date", "Renewal Date", "What is the client's renewal date?"),
    FormField("product_code", "Product Code", "What is the product code?")
]

# Create a form container in Streamlit
with st.form(key="rag_form"):
    st.subheader("Fill the form automatically")
    
    # Text input for the user's initial query
    user_query = st.text_input("Enter a query to start pre-filling the form (e.g., 'details for Client A')", key="user_query")
    
    # Button to trigger form filling
    fill_button = st.form_submit_button("Fill Form from RAG")

    # Store filled form data in session state
    if "completed_form_data" not in st.session_state:
        st.session_state.completed_form_data = {}
    
    if fill_button and user_query:
        st.session_state.completed_form_data = {} # Reset form data
        st.info("Pre-filling form based on your query...")
        for field in form_fields:
            answer = fill_form_field(query_engine, field)
            st.session_state.completed_form_data[field.name] = answer
            
    # Display and allow edits to the filled form data
    st.subheader("Completed Form")
    edited_form_data = {}
    for field in form_fields:
        default_value = st.session_state.completed_form_data.get(field.name, "")
        edited_form_data[field.name] = st.text_input(field.label, value=default_value, key=f"edit_{field.name}")
    
    # Button to trigger validation
    validate_button = st.form_submit_button("Validate Form")

# --- Step 4: Validation results display ---
if validate_button:
    if edited_form_data:
        with st.spinner("Validating form..."):
            validation_results = validate_form_logically(query_engine, edited_form_data)
        st.subheader("Validation Results")
        for check, result in validation_results.items():
            if result['result']:
                st.success(f"**{check}: PASSED**")
            else:
                st.error(f"**{check}: FAILED** - {result['details']}")
    else:
        st.warning("Please fill the form before validating.")