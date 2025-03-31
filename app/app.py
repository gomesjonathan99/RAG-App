import streamlit as st
from vectordb.vectorstore import VectorStore
from langchain_google_genai import GoogleGenerativeAI
from prompt.app import get_system_prompt
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Custom CSS styling for enhanced dark theme UI
st.markdown("""
<style>
    /* Main app styling - Dark Theme */
    .main {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    
    /* Container styling */
    .block-container {
        background-color: #222222;
        padding: 2rem;
        border-radius: 10px;
    }
    
    /* Title styling */
    h1 {
        color: #88c0d0;
        font-weight: 700;
        padding-bottom: 10px;
        border-bottom: 2px solid #5e81ac;
    }
    
    /* Subheader styling */
    h3 {
        color: #88c0d0;
        margin-top: 20px;
        font-weight: 600;
    }
    
    /* Regular text */
    p, label, span {
        color: #e0e0e0;
    }
    
    /* Divider styling */
    hr {
        margin: 25px 0;
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(94, 129, 172, 0), rgba(94, 129, 172, 0.75), rgba(94, 129, 172, 0));
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: #333333;
        color: #e0e0e0;
        border-radius: 8px;
        border: 1px solid #5e81ac;
        padding: 12px 15px;
        font-size: 16px;
        transition: all 0.3s;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #81a1c1;
        box-shadow: 0 0 0 0.2rem rgba(94, 129, 172, 0.25);
        background-color: #3a3a3a;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #5e81ac;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #81a1c1;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }
    
    /* Status message styling */
    .success-message {
        background-color: #2b3d2b;
        color: #a3be8c;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        border-left: 4px solid #a3be8c;
    }
    
    .error-message {
        background-color: #4d2c2c;
        color: #bf616a;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        border-left: 4px solid #bf616a;
    }
    
    /* Progress indicator */
    .stProgress > div > div > div {
        background-color: #5e81ac;
    }
    
    /* Response box styling */
    .response-box {
        background-color: #2e3440;
        color: #eceff4;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.25);
        font-family: 'Roboto', sans-serif;
        border-left: 5px solid #5e81ac;
        line-height: 1.6;
    }
    
    .response-box h2 {
        color: #88c0d0;
        margin-top: 0;
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    /* Card container for process steps */
    .process-step {
        background-color: #2c3440;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #5e81ac;
    }
    
    /* Section container */
    .section-container {
        background-color: #2a2a2a;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #3a3a3a;
    }
    
    /* Section header */
    .section-header {
        color: #88c0d0;
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }
    
    .section-header span {
        margin-left: 10px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #252525;
    }
    
    /* PDF URL display section */
    .url-display {
        background-color: #2e3440;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 15px;
        font-family: monospace;
        overflow-x: auto;
        white-space: nowrap;
        border: 1px solid #3b4252;
    }
    
    /* App header */
    .app-header {
        background-color: #2e3440;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-bottom: 3px solid #5e81ac;
    }
    
    .app-header h1 {
        margin: 0;
        padding: 0;
        border: none;
    }
    
    /* Helper text */
    .helper-text {
        font-size: 14px;
        color: #a0a0a0;
        margin-top: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App header with title and description
st.markdown('<div class="app-header">', unsafe_allow_html=True)
st.title("📄 PDF URL Processor")
st.markdown('<p style="font-size: 18px; color: #88c0d0;">Extract insights from any PDF document using AI</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
    st.session_state.vector_store = None
    st.session_state.pdf_url = ""

# URL input with better UI

st.markdown('<div class="section-header">📎 <span>Step 1: Provide PDF Source</span></div>', unsafe_allow_html=True)
st.markdown('<p>Enter the URL of the PDF document you want to analyze</p>', unsafe_allow_html=True)

# Fixed the empty label issue by providing a proper label and using label_visibility
pdf_url = st.text_input(
    label="PDF URL Input", 
    placeholder="https://example.com/document.pdf", 
    key="pdf_url_input",
    label_visibility="collapsed"  # Hide the label but keep it for accessibility
)

if pdf_url:
    st.session_state.pdf_url = pdf_url
    # Create a visual progress indicator
    st.markdown('<div class="process-step"><b>🚀 Processing PDF...</b></div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    
    try:
        # Initialize VectorStore
        vector_store = VectorStore(pdf_url)
        pdf_docs = vector_store.pdf_to_docs()
        progress_bar.progress(33)
        
        if pdf_docs:
            st.markdown('<div class="success-message">✅ PDF processed successfully!</div>', unsafe_allow_html=True)
            
            chunks = vector_store.text_splitter(pdf_docs)
            progress_bar.progress(66)
            
            if chunks:
                st.markdown('<div class="success-message">✅ Text split into chunks successfully!</div>', unsafe_allow_html=True)
                
                vector_db_as_retriever = vector_store.create_vector_db(chunks)
                progress_bar.progress(100)
                
                if vector_db_as_retriever:
                    st.markdown('<div class="success-message">✅ Vector database created successfully! Ready to answer questions.</div>', unsafe_allow_html=True)
                    st.session_state.pdf_processed = True
                    st.session_state.vector_store = vector_db_as_retriever
                else:
                    st.markdown('<div class="error-message">❌ Failed to create vector database</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-message">❌ Failed to split text into chunks</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">❌ Failed to process PDF</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="error-message">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Create a visually appealing question section

st.markdown('<div class="section-header">🔍 <span>Step 2: Ask Questions About Your Document</span></div>', unsafe_allow_html=True)

# Display the current PDF URL if it exists
if st.session_state.pdf_url:
    st.markdown('<p><strong>Currently loaded PDF:</strong></p>', unsafe_allow_html=True)
    st.markdown(f'<div class="url-display">{st.session_state.pdf_url}</div>', unsafe_allow_html=True)

# Query input section (always visible, but will only work after PDF processing)
st.markdown('<p>Ask any question about the content of your PDF</p>', unsafe_allow_html=True)

# Fixed the empty label issue by providing a proper label and using label_visibility
query = st.text_input(
    label="Question Input", 
    placeholder="What is the main topic of this document?", 
    key="query_input",
    label_visibility="collapsed"  # Hide the label but keep it for accessibility
)

# Initialize the LLM model
llm = GoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2, max_output_tokens=4096)

if query:
    if st.session_state.pdf_processed:
        # Show a loading spinner
        with st.spinner('Thinking...'):
            system_prompt, prompt = get_system_prompt()
            
            try:
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(st.session_state.vector_store, question_answer_chain)
                results = rag_chain.invoke({"input": query})
                
                # Display the answer in a beautifully styled box
                st.markdown('<div class="section-header">💡 <span>Answer</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="response-box"><h2>Response</h2>{results["answer"]}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-message">⚠️ Please process a PDF first before asking questions</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Add a footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #88c0d0; font-size: 14px;">© 2025 PDF Processor - Powered by Gemini 1.5 Pro</p>', unsafe_allow_html=True)