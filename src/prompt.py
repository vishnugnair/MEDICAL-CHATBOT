from langchain_core.prompts import ChatPromptTemplate

# System prompt template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don’t know the answer, say that you "
    "don’t know. Use three sentences maximum and keep the "
    "answer concise.\n\n"
    "{context}"
)

# Create a ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
