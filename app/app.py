import streamlit as st
from vectordb.vectorstore import VectorStore
from langchain_google_genai import GoogleGenerativeAI
from prompt.app import get_system_prompt
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Custom CSS styling for UI
st.markdown("""
<style>
.response-box {
  border: 2px solid #4CAF50;
  padding: 20px;
  border-radius: 10px;
  margin-top: 20px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  font-family: Arial, sans-serif;
  color: #F6F8D5;
}
.response-box h2 {
  color: #F6F8D5;
  margin-top: 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize the LLM model
llm = GoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2, max_output_tokens=4096)

st.title("📄 PDF URL Processor")
st.markdown("---")

# Initialize session state
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
    st.session_state.vector_store = None

if not st.session_state.pdf_processed:
    pdf_url = st.text_input("Enter PDF URL:", placeholder="https://example.com/document.pdf")
    st.markdown("---")
    
    if pdf_url:
        st.write("🚀 Processing PDF...")
        
        try:
            # Initialize VectorStore
            vector_store = VectorStore(pdf_url)
            pdf_docs = vector_store.pdf_to_docs()
            
            if pdf_docs:
                st.success("✅ PDF processed successfully!")
                
                chunks = vector_store.text_splitter(pdf_docs)
                if chunks:
                    st.success("✅ Text split into chunks successfully!")
                    
                    vector_db_as_retriever = vector_store.create_vector_db(chunks)
                    if vector_db_as_retriever:
                        st.success("✅ Vector database created successfully!")
                        st.session_state.pdf_processed = True
                        st.session_state.vector_store = vector_db_as_retriever
                    else:
                        st.error("❌ Failed to create vector database")
                else:
                    st.error("❌ Failed to split text into chunks")
            else:
                st.error("❌ Failed to process PDF")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if st.session_state.pdf_processed:
    st.subheader("🔍 Ask a Question")
    query = st.text_input("Enter your query:")
    
    if query:
        system_prompt, prompt = get_system_prompt()
        
        try:
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(st.session_state.vector_store, question_answer_chain)
            results = rag_chain.invoke({"input": query})
            
            st.markdown("---")
            st.markdown(f'<div class="response-box">{results["answer"]}</div>', unsafe_allow_html=True)
            st.markdown("---")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")