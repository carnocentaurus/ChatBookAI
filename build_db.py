#!/usr/bin/env python3
"""
Pre-build the Chroma vector database locally
Run this ONCE on your laptop, then push to GitHub
"""

import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration
HANDBOOK_FILE = "data/handbook.pdf"
DB_PATH = "data/chroma_db"

def load_handbook():
    """Load handbook text from PDF"""
    print(f"📚 Loading handbook from: {HANDBOOK_FILE}")
    
    if not os.path.exists(HANDBOOK_FILE):
        print(f"❌ Error: {HANDBOOK_FILE} not found!")
        return ""
    
    try:
        reader = PdfReader(HANDBOOK_FILE)
        print(f"✅ PDF opened! Pages: {len(reader.pages)}")
        
        text = ""
        for i, page in enumerate(reader.pages):
            print(f"📄 Extracting page {i+1}/{len(reader.pages)}...")
            text += page.extract_text() + "\n"
        
        print(f"✅ Extraction complete! Total characters: {len(text)}")
        return text.strip()
    
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return ""

def build_database():
    """Build the Chroma vector database"""
    print("\n" + "="*60)
    print("🔨 BUILDING VECTOR DATABASE")
    print("="*60 + "\n")
    
    # Load handbook
    handbook_text = load_handbook()
    
    if not handbook_text:
        print("❌ Cannot build database without handbook text!")
        return False
    
    # Load embeddings model
    print("\n🧠 Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Embeddings model loaded")
    
    # Split text into chunks
    print("\n✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(handbook_text)
    print(f"✅ Split into {len(chunks)} chunks")
    
    # Delete old database if exists
    if os.path.exists(DB_PATH):
        print(f"\n🗑️ Deleting old database at {DB_PATH}...")
        import shutil
        shutil.rmtree(DB_PATH)
    
    # Create database
    print(f"\n💾 Creating vector database at {DB_PATH}...")
    print("⏳ This may take 2-5 minutes...")
    
    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print("\n" + "="*60)
    print("🎉 DATABASE BUILT SUCCESSFULLY!")
    print("="*60)
    print(f"\n📊 Database location: {DB_PATH}")
    print(f"📊 Total chunks: {len(chunks)}")
    print(f"📊 Database size: {get_folder_size(DB_PATH):.2f} MB")
    print("\n✅ Ready to push to GitHub!")
    
    return True

def get_folder_size(path):
    """Get folder size in MB"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total += os.path.getsize(filepath)
    return total / (1024 * 1024)

if __name__ == "__main__":
    print("\n🚀 Starting database build process...\n")
    
    # Check if handbook exists
    if not os.path.exists(HANDBOOK_FILE):
        print(f"❌ Error: {HANDBOOK_FILE} not found!")
        print("📁 Make sure the handbook PDF is in the data/ folder")
        exit(1)
    
    # Build database
    success = build_database()
    
    if success:
        print("\n📝 Next steps:")
        print("1. Check that data/chroma_db/ folder was created")
        print("2. Run: git add data/chroma_db")
        print("3. Run: git commit -m 'Add pre-built vector database'")
        print("4. Run: git push")
        print("5. Wait for Render to redeploy (2-3 minutes)")
        print("\n✨ Your memory usage will be MUCH lower!")
    else:
        print("\n❌ Database build failed. Check errors above.")