"""
streamlit_app.py - Main Streamlit interface for PDF2Table
Web interface for uploading PDFs and extracting structured information
"""

import streamlit as st
import pandas as pd
import os
import tempfile
from utils import process_uploaded_file, ExtractedInfo
import time

# Page configuration
st.set_page_config(
    page_title="PDF2Table - AI-Powered Research Paper Extraction",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
}

.sub-header {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}

.feature-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.success-box {
    background-color: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #c3e6cb;
}

.info-box {
    background-color: #d1ecf1;
    color: #0c5460;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #bee5eb;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📄 PDF2Table</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Data Extraction from Research Papers</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        **PDF2Table** extracts structured information from research papers using:
        
        - 🤖 **OpenAI GPT-4o-mini**
        - 🔗 **LangChain RAG**
        - 🧠 **ChromaDB Vectors**
        - 📊 **Streamlit Interface**
        """)
        
        st.header("🎯 Features")
        st.markdown("""
        - Extract **title**, **authors**, **summary**, and **publication year**
        - Smart context retrieval
        - Source attribution
        - Clean table output
        """)
        
        st.header("⚙️ How it Works")
        st.markdown("""
        1. Upload your PDF
        2. AI processes the document
        3. Extracts key information
        4. Displays results in a table
        """)

    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Research Paper")
        st.markdown("Upload a PDF research paper to extract structured information")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Select a research paper in PDF format"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Processing section
    if uploaded_file is not None:
        # Display file info
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"**📄 File:** {uploaded_file.name}")
        st.markdown(f"**📏 Size:** {uploaded_file.size:,} bytes")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Extract Information", type="primary", use_container_width=True):
                process_document(uploaded_file)
    
    else:
        # Demo section when no file is uploaded
        st.markdown("---")
        st.markdown("### 📈 Example Output")
        
        # Example table
        example_data = {
            'Title': ['Consequences of Erudite Vernacular Utilized Irrespective of Necessity: Problems with Using Long Words Needlessly'],
            'Summary': ['Study investigating how complex writing affects reader perceptions, finding that overly complex vocabulary leads to negative evaluations despite attempts to appear intelligent.'],
            'Year': ['2005'],
            'Authors': ['Daniel M. Oppenheimer']
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        # Features showcase
        st.markdown("---")
        st.markdown("### 🌟 Key Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 🎯 Accurate Extraction
            Uses advanced AI to identify and extract key paper elements with high precision.
            """)
        
        with col2:
            st.markdown("""
            #### 📚 Source Attribution
            Provides source text and reasoning for each extracted piece of information.
            """)
        
        with col3:
            st.markdown("""
            #### ⚡ Fast Processing
            Efficient RAG pipeline processes papers in seconds, not minutes.
            """)

def process_document(uploaded_file):
    """Process the uploaded document and display results"""
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: File upload
        status_text.text("📤 Processing uploaded file...")
        progress_bar.progress(20)
        time.sleep(0.5)
        
        # Step 2: PDF processing
        status_text.text("📄 Loading and chunking PDF...")
        progress_bar.progress(40)
        time.sleep(0.5)
        
        # Step 3: Embeddings
        status_text.text("🧠 Creating embeddings...")
        progress_bar.progress(60)
        time.sleep(0.5)
        
        # Step 4: Extraction
        status_text.text("🤖 Extracting information with AI...")
        progress_bar.progress(80)
        
        # Process the file
        extracted_info, df = process_uploaded_file(uploaded_file)
        
        # Step 5: Complete
        status_text.text("✅ Extraction complete!")
        progress_bar.progress(100)
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        display_results(extracted_info, df)
        
    except Exception as e:
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show error
        st.error(f"❌ Error processing file: {str(e)}")
        st.markdown("""
        **Possible issues:**
        - PDF might be corrupted or password-protected
        - OpenAI API key might be missing or invalid
        - Network connectivity issues
        
        Please check your setup and try again.
        """)

def display_results(extracted_info: ExtractedInfo, df: pd.DataFrame):
    """Display the extraction results"""
    
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("### ✅ Information Successfully Extracted!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main results table
    st.markdown("### 📊 Extracted Information")
    st.dataframe(df, use_container_width=True)
    
    # Detailed breakdown with sources
    st.markdown("---")
    st.markdown("### 🔍 Detailed Breakdown with Sources")
    
    # Create tabs for each field
    tab1, tab2, tab3, tab4 = st.tabs(["📰 Title", "📝 Summary", "📅 Year", "👥 Authors"])
    
    with tab1:
        display_field_details("Title", extracted_info.paper_title)
    
    with tab2:
        display_field_details("Summary", extracted_info.paper_summary)
    
    with tab3:
        display_field_details("Publication Year", extracted_info.publication_year)
    
    with tab4:
        display_field_details("Authors", extracted_info.paper_authors)
    
    # Download option
    st.markdown("---")
    st.markdown("### 💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="extracted_paper_info.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON download
        json_data = extracted_info.model_dump_json(indent=2)
        st.download_button(
            label="📥 Download Full Details (JSON)",
            data=json_data,
            file_name="extracted_paper_details.json",
            mime="application/json"
        )

def display_field_details(field_name: str, field_data):
    """Display detailed information for a specific field"""
    
    st.markdown(f"**{field_name}:**")
    st.write(field_data.answer)
    
    st.markdown("**Source Text:**")
    st.text_area(
        "Source",
        field_data.sources,
        height=100,
        key=f"source_{field_name}",
        label_visibility="collapsed"
    )
    
    st.markdown("**AI Reasoning:**")
    st.write(field_data.reasoning)

if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ OpenAI API key not found!")
        st.markdown("""
        Please set your OpenAI API key:
        1. Create a `.env` file in your project directory
        2. Add: `OPENAI_API_KEY=your_api_key_here`
        3. Restart the application
        """)
        st.stop()
    
    main()