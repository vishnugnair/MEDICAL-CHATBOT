from flask import Flask, render_template, request
import os
from dotenv import load_dotenv

# LangChain + Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAI

# Local modules
from store_index import create_index_if_not_exists, load_or_create_docsearch, index_name
from src.prompt import prompt

app = Flask(__name__)

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 1. Create or verify Pinecone index
create_index_if_not_exists()

# 2. Load documents and create docsearch
docsearch, embeddings = load_or_create_docsearch()

# 3. Build retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwags={"k": 3})

# 4. Initialize OpenAI LLM
llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0.4, max_tokens=500)

# 5. Create Q&A chain and RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route('/')
def index():
    """
    Render the home page (index.html).
    """
    return render_template('index.html', response=None)

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle the user query from the form, run the RAG chain, and return the answer.
    """
    user_query = request.form.get('question')
    if not user_query:
        return render_template('index.html', response="No question provided.")

    # Run the RAG chain with the user query
    response = rag_chain.invoke({"input": user_query})
    answer = response["answer"]

    return render_template('index.html', response=answer, query=user_query)

if __name__ == '__main__':
    # Run on localhost:8080
    app.run(host='0.0.0.0', port=8080, debug=True)
