import os 
from dotenv import load_dotenv
import streamlit as st
from htmlTemplates import css, user_template, bot_template
from lanchain.text_splitters import CharacterTextSplitter
from lanchain.store.vectorstore import FAISS
from lanchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from lanchain.chains import ConversationalRetrieverChain


def get_pdf_text(pdf_docs):
    text = ""
    if pdf_docs:
        #turn each pdf into a PdfReader object
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                pdf_page = page.extract_text()
                if pdf_page:
                    text += pdf_page
    return text
        
def get_text_chunks(text):
    #to be able to split text into chunks we need an Text Splitter
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    ) #here we just create a TextSplitter ... nothing happened yet
    
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(chunks):
    #first we need an Embedding model
    openai_encoder = openai.OpenAIEmbeddings()

    #we can now create our vector store 
    vectorstore = FAISS.from_texts(
        embedding = openai_encoder,
        text = chunks
    )
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    
    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        return_message=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_queries(user_question):
    # this is where the final processing happens
    
    #get the response
    response = st.session_state.conversation({"question":user_question})
    
    #retrieve the conversation history using the chat_history key we set before
    st.session_state.chat_history = response['chat_history']
    
    #print and format LLM's output
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
    


def main():
    #load our api keys; without the API's won't be able to use the LLM
    load_dotenv()
    
    #setup streamlist config
    st.set_config_page(page_title="Ask Multiple PDFs", page_icon=":books:")
    st.header("Ask Multiple PDFs")

    #load our custom css classes
    st.write(css, unsafe_html_code=true)
    
    #initializing values in session state so that the conversation history survives the re-run
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    user_question = st.text_input("Type your question..")
    if user_question:
        if st.session.state.conversation is None:
            st.warning("Please upload and process your PDF first.")
        else:
            handle_user_queries(user_question)
        
    with st.sidebar:
        # add the file upload field
        pdf_docs = st.file_uploader(
            "Upload Your PDFs and click on process", accept_multiple_files=True
        )
        #create a submit button
        if st.button("Process"):
            #extract text
            text = get_pdf_text(pdf_docs)
            
            #chunk text 
            chunks = get_text_chunks(text)
            
            #create a vectorstore
            #to create a vectorstore do we need a a separate function ?
                #yes, separation of concerns
            vectorstore = get_vectorstore(chunks)
            
            #create a conversation chain
            st.session_state.conversation = get_conversation_chain(
                vectorstore)
            
            
        
        
        

if __name__ == "__main__":
    main()