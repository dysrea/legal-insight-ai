import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# PATH CONSTANTS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DB_FAISS_PATH = os.path.join(ROOT_DIR, "vectorstore", "db_faiss")

# CUSTOM PROMPT
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

IMPORTANT: You are an expert Indian Lawyer. 
1. You MUST cite the specific "Section" numbers (e.g., Section 302 IPC, Section 376) if they are mentioned in the context.
2. Do not give generic advice. Stick strictly to the provided legal text.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def load_llm():
    llm = Ollama(model="llama3")
    return llm

def main():
    st.set_page_config(page_title="Legal Insight AI")
    st.title("⚖️ Legal Insight Assistant")
    st.markdown("### Ask questions about the Indian Penal Code")

    # Load Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    # Load the Database
    # We use allow_dangerous_deserialization = True because we created the DB ourselves
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization = True)
    except Exception as e:
        st.error(f"Error loading Vector Database: {e}")
        st.info("Hint: Did you run 'python src/ingest.py' successfully?")
        return

    # Create QnA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt()}
    )

    # User Input
    user_query = st.text_input("Enter your query here:", placeholder="e.g., What is the punishment for theft?")

    # Logic
    if user_query:
        with st.spinner("Analyzing Legal Precedents..."):
            try:
                # Send query to the chain
                response = qa_chain.invoke({'query': user_query})
                
                # Show Result
                st.markdown(f"AI Response:")
                st.write(response["result"])
                
                # Show Evidence
                with st.expander("View Source Documents (Verification)"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Source {i+1}:**")
                        st.info(doc.page_content)
            
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()