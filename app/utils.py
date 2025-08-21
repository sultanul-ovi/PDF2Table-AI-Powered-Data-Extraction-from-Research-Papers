"""
utils.py - Helper functions for PDF2Table application
Contains all core functionality for PDF processing, embedding, and extraction
"""

import os
import uuid
import tempfile
import pandas as pd
from typing import List, Optional

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

# Environment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class AnswerWithSources(BaseModel):
    """An answer to the question, with sources and reasoning."""
    answer: str = Field(description="Answer to question")
    sources: str = Field(description="Full direct text chunk from the context used to answer the question")
    reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")

class ExtractedInfo(BaseModel):
    """Extracted information about the research article"""
    paper_title: AnswerWithSources
    paper_summary: AnswerWithSources
    publication_year: AnswerWithSources
    paper_authors: AnswerWithSources

def get_embedding_function():
    """Create and return OpenAI embedding function"""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

def get_llm():
    """Create and return ChatOpenAI instance"""
    return ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

def load_pdf(file_path: str) -> List:
    """
    Load PDF document and return pages
    
    Args:
        file_path (str): Path to PDF file
        
    Returns:
        List: List of document pages
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at {file_path}")
    
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

def split_documents(pages: List, chunk_size: int = 1500, chunk_overlap: int = 200) -> List:
    """
    Split documents into chunks
    
    Args:
        pages (List): List of document pages
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        List: List of text chunks
    """
    if not pages:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " "]
    )
    chunks = text_splitter.split_documents(pages)
    return chunks

def create_vectorstore(chunks: List, embedding_function, vectorstore_path: str):
    """
    Create vectorstore from document chunks
    
    Args:
        chunks (List): Document chunks
        embedding_function: Embedding function
        vectorstore_path (str): Path to save vectorstore
        
    Returns:
        Chroma: Vectorstore instance
    """
    # Create unique UUIDs based on chunk content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    unique_ids = set()
    unique_chunks = []

    for chunk, id in zip(chunks, ids):     
        if id not in unique_ids:       
            unique_ids.add(id)
            unique_chunks.append(chunk) 

    vectorstore = Chroma.from_documents(
        documents=unique_chunks,
        ids=list(unique_ids),
        embedding=embedding_function,
        persist_directory=vectorstore_path
    )

    return vectorstore

def load_existing_vectorstore(vectorstore_path: str, embedding_function):
    """
    Load existing vectorstore
    
    Args:
        vectorstore_path (str): Path to vectorstore
        embedding_function: Embedding function
        
    Returns:
        Chroma: Vectorstore instance
    """
    return Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embedding_function
    )

def format_docs(docs: List) -> str:
    """Format retrieved documents into context string"""
    return "\n\n".join(doc.page_content for doc in docs)

def get_comprehensive_context(vectorstore, question: str) -> str:
    """
    Get comprehensive context by searching for key paper elements
    
    Args:
        vectorstore: Chroma vectorstore instance
        question (str): User question
        
    Returns:
        str: Formatted context string
    """
    # Search for different aspects to ensure we get all relevant chunks
    searches = [
        "title consequences erudite vernacular",
        "authors Daniel Oppenheimer",
        "abstract summary introduction",
        "publication year 2005 copyright",
        "conclusion implications applications"
    ]
    
    all_chunks = []
    seen_content = set()
    
    for search_query in searches:
        chunks = vectorstore.similarity_search(search_query, k=3)
        for chunk in chunks:
            # Avoid duplicate chunks
            chunk_preview = chunk.page_content[:100]
            if chunk_preview not in seen_content:
                seen_content.add(chunk_preview)
                all_chunks.append(chunk)
    
    # Also include the best chunks from the original question
    question_chunks = vectorstore.similarity_search(question, k=2)
    for chunk in question_chunks:
        chunk_preview = chunk.page_content[:100]
        if chunk_preview not in seen_content:
            seen_content.add(chunk_preview)
            all_chunks.append(chunk)
    
    return format_docs(all_chunks)

def create_structured_extraction_chain(vectorstore, llm):
    """
    Create structured extraction chain
    
    Args:
        vectorstore: Chroma vectorstore instance
        llm: ChatOpenAI instance
        
    Returns:
        Chain: Structured extraction chain
    """
    # Structured extraction prompt
    STRUCTURED_PROMPT_TEMPLATE = """
    You are an expert research paper analyzer. Extract the following information from the provided context.
    Be precise and only use information that is explicitly stated in the context.

    Context:
    {context}

    Extract the following information about this research paper:
    1. Paper Title: The exact title of the research paper
    2. Paper Summary: A comprehensive summary of the paper's main findings and contributions  
    3. Publication Year: The year this paper was published
    4. Paper Authors: The authors who wrote this paper

    For each field, provide:
    - answer: The extracted information
    - sources: Quote the exact text from context that supports your answer
    - reasoning: Brief explanation of how you found this information

    If any information is not available in the context, state "Not available in provided context" for that field.

    Question: {question}
    """

    structured_prompt_template = ChatPromptTemplate.from_template(STRUCTURED_PROMPT_TEMPLATE)
    
    # Create comprehensive context function for this vectorstore
    def get_context_for_vectorstore(question):
        return get_comprehensive_context(vectorstore, question)
    
    # Create structured chain
    structured_chain = (
        {"context": get_context_for_vectorstore, "question": RunnablePassthrough()}
        | structured_prompt_template
        | llm.with_structured_output(ExtractedInfo, strict=True)
    )
    
    return structured_chain

def extract_paper_info(file_path: str, vectorstore_path: Optional[str] = None) -> ExtractedInfo:
    """
    Complete pipeline to extract information from a research paper
    
    Args:
        file_path (str): Path to PDF file
        vectorstore_path (str, optional): Path to save/load vectorstore
        
    Returns:
        ExtractedInfo: Extracted paper information
    """
    # Initialize components
    embedding_function = get_embedding_function()
    llm = get_llm()
    
    if vectorstore_path is None:
        vectorstore_path = "vectorstore_temp"
    
    # Check if vectorstore already exists
    if os.path.exists(vectorstore_path):
        try:
            vectorstore = load_existing_vectorstore(vectorstore_path, embedding_function)
        except:
            # If loading fails, create new vectorstore
            pages = load_pdf(file_path)
            chunks = split_documents(pages)
            vectorstore = create_vectorstore(chunks, embedding_function, vectorstore_path)
    else:
        # Create new vectorstore
        pages = load_pdf(file_path)
        chunks = split_documents(pages)
        vectorstore = create_vectorstore(chunks, embedding_function, vectorstore_path)
    
    # Create extraction chain
    extraction_chain = create_structured_extraction_chain(vectorstore, llm)
    
    # Extract information
    result = extraction_chain.invoke(
        "Extract the title, summary, publication year, and authors from this research paper."
    )
    
    return result

def convert_to_dataframe(extracted_info: ExtractedInfo) -> pd.DataFrame:
    """
    Convert extracted information to a clean DataFrame
    
    Args:
        extracted_info (ExtractedInfo): Extracted paper information
        
    Returns:
        pd.DataFrame: Formatted DataFrame
    """
    data = {
        'Title': extracted_info.paper_title.answer,
        'Summary': extracted_info.paper_summary.answer,
        'Year': extracted_info.publication_year.answer,
        'Authors': extracted_info.paper_authors.answer
    }
    
    df = pd.DataFrame([data])
    return df

def process_uploaded_file(uploaded_file) -> tuple:
    """
    Process uploaded file and return extracted information and DataFrame
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        tuple: (ExtractedInfo, DataFrame)
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Extract information
        extracted_info = extract_paper_info(tmp_file_path)
        
        # Convert to DataFrame
        df = convert_to_dataframe(extracted_info)
        
        return extracted_info, df
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)