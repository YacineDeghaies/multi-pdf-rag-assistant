import streamlit as st
import os
from langchain_openai import (OpenAIEmbeddings, ChatOpenAI)
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain ,
)
from htmlTemplates import css
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from uuid import uuid4

def handle_user_query(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation.invoke(
            {"input": user_question},
            config={
                "configurable": {
                    "session_id": st.session_state.session_id
                }
            },
        )
    
histories = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in histories:
        histories[session_id] = InMemoryChatMessageHistory()

    # Return the history associated with this session ID.
    return histories[session_id]
    
def get_text_from_pdfs(pdf_docs):
    text = ""
    # Turn each pdf into a PdfReader object
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_page = page.extract_text()
            if pdf_page:
                text += pdf_page
    return text
            
    
def get_chunks_from_text(text):
    #to split text into chunks, what is required?
    # a TextSplitter
    # what does a splitter requires ?
        # infos on how to split: separator, chunk size, chunk_overlap, etc..
    text_splitter = RecursiveCharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)
    
def get_vectorstore(chunks):
    #To define a vectorstore, what do we first require?
        # an Embeddings model
        # the chunks of text
    embedding_model = OpenAIEmbeddings()
    return FAISS.from_texts(
        texts=chunks,
        embeddings=embedding_model
    )
    
def get_conversation_chain(vectorstore):
    # what do we first need ?
        # a conversation chain object from langchain, what is it called? create_retrieval_chain
        #it's not an object, its a function that will later return a chain
        #what does it require?
            # a history aware retriever
            # a stuff_docs chain
            
    #how to create a history aware retriever?
    #what does a retriever needs in general ?
    #the word history means it needs the chat_history
        #where do we find it ?
        #it will be later supplied by the retrieval chain to the history-aware retriever
        #how do we tell the chain to provide us with chat history
        #through a placeholder? a placeholder needs to be inside a template, e.g., ChatPromptTemplate
        
    contextualize_q_system_prompt = """
    You are provided with the chat history and the user's latest question.
    If the user's latest question depends on the chat history for its meaning, 
    rewrite it as a standalone question.
    The user question should not be dependent on the chat history.
    If the latest user's question does not depend on the chat history, 
    do not rephrase it and return it as it is.
    Do not answer the question.
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        # Include messages here
        ("system", contextualize_q_system_prompt), #the system prompt
        MessagesPlaceholder("chat_history"), #the chat history
        ("human", "{input}") # the latest user question
    ])
    
    # We first need a retriever. Reason for this:
    # We cannot get a history aware retriever from the vectorspace itself
    # We have to connect the retriever with the chat history ourselves
    retriever = vectorstore.as_retriever()
    
    # we also need an LLM for both our chains
    llm = ChatOpenAI()
    
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_q_prompt
    )
    
    qa_system_prompt = """
    You are an assistant for question-answering tasks
    Use the following retrieved context to answer the question.
    If you don't know the answer, say that you don't know.

    Context:
    {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    # create a document-combining chain
    combine_docs_chain = create_stuff_documents_chain(
        llm, qa_prompt
    )
    
    retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
    
    chain_with_history = RunnableWithMessageHistory(
    retrieval_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    ) 
    
    return chain_with_history
    
def main():
   
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs")

    #every browser session gets its own conversation
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())

    #load our customized css 
    st.write(css, unsafe_allow_html=True)


    # Initializing session state variables.
    # we need to save our conversation chain and chat_history by every re-run from streamlit
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    # take user input
    user_question = st.text_input("Ask question to your PDFs")
    if user_question:
        handle_user_query(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        
        #upload PDFs
        pdf_docs = st.file_uploader(
            "Upload your PDF files", accept_multiple_files=True ,type="pdf"
        )

        if st.button("Process"):
            with st.spinner("Processing.."):

                #extract text from PDFs
                text = get_text_from_pdfs(pdf_docs)

                #split text into chunks
                chunks = get_chunks_from_text(text)

                #create a vectorstore to store our Embeddings
                vectorstore = get_vectorstore(chunks)

                #create a conversation chain
                conversation_chain = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()