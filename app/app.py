""" Streamlit app """
from vectordb.vectorstore import VectorStore
from langchain_google_genai import GoogleGenerativeAI
from prompt.app import get_prompt
import streamlit as st

# Custom CSS styling for a professional look
st.markdown("""
<style>
.response-box {
  border: 2px solid #4CAF50; /* Green border */
  padding: 20px;
  border-radius: 10px;
  margin-top: 20px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  font-family: Arial, sans-serif;
  color: #F6F8D5; /* Dark gray text */
}
.response-box h2 {
  color: #F6F8D5; /* Orange header text */
  margin-top: 0;
}

</style>
""", unsafe_allow_html=True)

# Initialize the LLM model
llm = GoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2, max_output_tokens=4096)

# st.sidebar.header("🔑 Enter API Key")
# gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
# Main UI: User inputs PDF URL
st.title("📄 PDF URL Processor")
st.markdown("---")
# Initialize session state variables
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
    st.session_state.vector_store = None

if not st.session_state.pdf_processed:
    pdf_url = st.text_input("Enter PDF URL:", placeholder="https://example.com/document.pdf")
    st.markdown("---")
    if pdf_url:
        st.write("🚀 Processing PDF...")

        # Initialize VectorStore with the PDF URL
        vector_store = VectorStore(pdf_url)
        
        # Convert PDF to text
        pdf_str = vector_store.pdf_to_text()

        if pdf_str:
            st.success("✅ PDF processed successfully! (pdf -> String)")

            # Split text into chunks
            chunks = vector_store.text_splitter(pdf_str)
            if chunks:
                st.success("✅ Text split into chunks successfully!")

                # Create vector database
                vector_db_as_retriever = vector_store.create_vector_db(chunks)
                if vector_db_as_retriever:
                    st.success("✅ Vector database created successfully!")

                    # Save state and show query input
                    st.session_state.pdf_processed = True
                    st.session_state.vector_store = vector_db_as_retriever
                else:
                    st.error("❌ Failed to create vector database")
            else:
                st.error("❌ Failed to split text into chunks")
        else:
            st.error("❌ Failed to process PDF -> String")

# Show query input after PDF processing
if st.session_state.pdf_processed:
    st.subheader("🔍 Ask a Question")
    query = st.text_input("Enter your query:")

    if query:
        docs = st.session_state.vector_store.get_relevant_documents(query)
        
        if docs:
            st.success("✅ Retrieved relevant documents successfully!")
            
            # Extract text from relevant documents
            docs_to_llm = "\n\n".join([doc.page_content for doc in docs])

            # Generate response using LLM
            prompt = get_prompt()
            
            chain = prompt | llm
            response = chain.invoke({"user_input": query, "docs": docs_to_llm})

            # Display AI Response in styled box
            st.markdown("---")
            st.markdown(f'<div class="response-box font-size: 40px">{response}</div>', unsafe_allow_html=True)
            st.markdown("---")        
        else:
            st.error("❌ No relevant documents found.")
