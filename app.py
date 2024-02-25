import streamlit as st
from dotenv import load_dotenv 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI 
from streamlit_chat import message
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFaceHub


def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 10,
        length_function = len 
    )
    chunks = text_splitter.split_text(text)
    return chunks 


def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceBgeEmbeddings(model_name = 'BAAI/bge-small-en')
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id='google/flan-t5-xxl', model_kwargs={'temperature': 0.5, 'max_length': 512}, huggingfacehub_api_token='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')
    # llm = HuggingFaceHub(repo_id = 'google/flan-t5-xxl'. model_kwargs = {'temperature':0.5, 'max_length':512})
    memory = ConversationBufferMemory(memory_key ='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(),
        memory = memory 
    )
    return conversation_chain
    

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(response['answer'])

# Main function
def main():
    load_dotenv()
    st.set_page_config(page_title='College Query Chat', page_icon='🎓')
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header('Admission Query Chat 🎓')
    message('Hello, how can I assist you today?', is_user=True)
    message("Hello! I'm here to help you with your college queries.")

    user_question = st.text_input('Ask a question about college')
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader('Upload college document here and click on "Process": ', accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner('Processing'):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                
                # Get text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()
