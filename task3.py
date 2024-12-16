import pdfplumber
import camelot
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import os
# Set OpenAI API key
openai.api_key = os.environ.get('open_api_key')

# Initialize embedding model and FAISS
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384  # Embedding dimension for 'all-MiniLM-L6-v2'
index = faiss.IndexFlatL2(dimension)

# Store chunks for retrieval
chunk_storage = []

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() for page in pdf.pages]

# Function to extract tables from PDF
def extract_tables_from_pdf(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages="all")
    return [table.df for table in tables]

# Function to chunk text
def chunk_text(text, max_length=200):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(sentence)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# Function to embed and store chunks in FAISS
def embed_and_store_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    global chunk_storage
    chunk_storage.extend(chunks)
    index.add(np.array(embeddings))

# Function to retrieve chunks based on user query
def retrieve_chunks(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [chunk_storage[idx] for idx in indices[0]]
    return results

# Function to generate response using GPT
def generate_response(query, retrieved_chunks):
    prompt = (
        f"Answer the query based on the following information:\n\n"
        f"{retrieved_chunks}\n\n"
        f"Query: {query}\n"
        f"Answer:"
    )
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # Use gpt-4 if needed
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=200
)
    return response['choices'][0]['message']['content'].strip()



# Interactive Workflow
def main():
    print("=== PDF Chatbot with RAG Pipeline ===")
    pdf_path = input("Enter the path to the PDF file: ")
    
    # Step 1: Extract text and tables
    print("Extracting text from PDF...")
    pages_text = extract_text_from_pdf(pdf_path)
    
    print("Extracting tables from PDF...")
    tables = extract_tables_from_pdf(pdf_path)

    # Step 2: Chunk and embed the text
    print("Processing text...")
    for page_text in pages_text:
        chunks = chunk_text(page_text)
        embed_and_store_chunks(chunks)

    print("Data is ready for querying.")
    
    # Step 3: Handle user queries
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Exiting the chatbot. Goodbye!")
            break

        # Retrieve chunks
        print("Retrieving relevant data...")
        retrieved_chunks = retrieve_chunks(query)

        # Generate response
        print("Generating response...")
        response = generate_response(query, retrieved_chunks)
        print("\nResponse:\n", response)

# Run the chatbot
if __name__ == "__main__":
    main()