import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import tempfile
import re
from difflib import SequenceMatcher

# Initialize the model
model = ChatOpenAI(model="gpt-4o")

# Streamlit UI
st.title("Demo: PDF Quote Finder with RAG")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Slider for chunk size between 100 and 5000
chunk_size = st.slider("Select chunk size", min_value=100, max_value=5000, value=2000, step=100)

# State for uploaded file and chunk size
if uploaded_file:
    # Check if the file has changed or if chunk size has changed
    if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name or "chunk_size" not in st.session_state or st.session_state.chunk_size != chunk_size:
        
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Load and chunk contents of the PDF
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        # Dynamically recalculate overlap
        chunk_overlap = int(chunk_size * 0.2)

        # Split documents into chunks using the selected chunk size from the slider
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits = text_splitter.split_documents(docs)
        
        # Store chunks in session state
        st.session_state.all_splits = all_splits
        st.session_state.chunk_size = chunk_size
        st.session_state.uploaded_file_name = uploaded_file.name

        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = InMemoryVectorStore(embeddings)
        
        # Add documents to vector store
        vector_store.add_documents(documents=all_splits)
        
        # Store the vector store in session state
        st.session_state.vector_store = vector_store

        st.success(f"Number of chunks created: {len(all_splits)}")
        st.info("PDF content has been indexed for retrieval.")

# State management for query and response
if "response" not in st.session_state:
    st.session_state.response = None

# Query input field
query = st.text_input("Enter your query:")

# Function to highlight matching sentences in the top result
def highlight_matching_lines(response_text, top_result_text, threshold=0.7):
    # Split the texts into sentences
    top_result_sentences = re.split(r'(?<=[\.\?\!])\s+', top_result_text)
    response_sentences = re.split(r'(?<=[\.\?\!])\s+', response_text)

    highlighted_result = ''
    for top_sentence in top_result_sentences:
        # Initialize match_found as False
        match_found = False
        for response_sentence in response_sentences:
            # Clean up the sentences by stripping whitespace
            top_sentence_clean = top_sentence.strip()
            response_sentence_clean = response_sentence.strip()
            # Compute the similarity ratio
            ratio = SequenceMatcher(None, top_sentence_clean, response_sentence_clean).ratio()
            if ratio >= threshold:
                match_found = True
                break  # Break since we've found a matching sentence

        if match_found:
            # Highlight the matching sentence
            highlighted_result += f"<span style='background-color: yellow;'>{top_sentence}</span> "
        else:
            highlighted_result += f"{top_sentence} "

    return highlighted_result

if query and "vector_store" in st.session_state:  # Ensure vector store exists
    # Retrieve relevant documents based on the query
    retrieved_docs = st.session_state.vector_store.similarity_search_with_score(query, k=1)

    if retrieved_docs:
        doc, score = retrieved_docs[0]
        metadata = doc.metadata
        
        # Prepare the context for the prompt
        context = doc.page_content

        # Define the prompt template
        prompt_template = """
        Please find the most relevant sentences from the context provided that match the human input based on meaning. 
        Note that the human input may be a translation back to English from Korean, so the wording might differ from the original English text.
        Focus on the semantic meaning and use your understanding of paraphrasing to identify matching sentences, even if the wording is different.
        Please clean up any line break characters in your response and enclose it in double quotations. Do not add an introduction or conclusion in your response. 
        If no matching lines are found, respond with 'No matching quote found'.

        Human Input: {question}

        Context: {context}

        Answer:
        """
        formatted_prompt = prompt_template.format(question=query, context=context)

        # Generate the response
        response = model.invoke(formatted_prompt)
        st.session_state.response = response.content

        # Highlight matching sentences in the top result
        highlighted_top_result = highlight_matching_lines(response.content, doc.page_content)

        st.write("### Quote:")
        st.write(st.session_state.response)
        st.write("### Top Result:")
        st.markdown(highlighted_top_result, unsafe_allow_html=True)
        st.write(f"**Score:** {score:.4f} \t**Page:** {metadata['page'] + 1}")

    else:
        st.write("No relevant documents found.")