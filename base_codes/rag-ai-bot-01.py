########################################################################################################################
# 01 - SECURELY PULL IN API KEYS & GIVE LANGCHAIN PROJECT NAME
########################################################################################################################

from dotenv import load_dotenv
load_dotenv()


########################################################################################################################
# 02 - LOAD DOCUMENT
########################################################################################################################

from langchain_community.document_loaders import TextLoader

raw_filename = "abc-grocery-help-desk-data.md"
loader = TextLoader(raw_filename, encoding="utf-8")
docs = loader.load()
text = docs[0].page_content

########################################################################################################################
# 03 - SPLIT DOCUMENT INTO CHUNKS
########################################################################################################################

from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("###", "id")],
    strip_headers= True)

chunked_docs = splitter.split_text(text)
print(len(chunked_docs), "Q/A chuncs") # 32 length
print(chunked_docs[0].page_content)    # output without page_content and metadata



########################################################################################################################
# 04 - TURN EACH CHUNK INTO AN EMBEDDING VECTOR & STORE IN VECTOR DB
########################################################################################################################

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# create the embeddings
embedding = OpenAIEmbeddings(model ="text-embedding-3-small")

# create vector database
vectorstore = Chroma.from_documents(documents=chunked_docs,
                                    embedding=embedding,
                                    collection_metadata={"hnsw:space":"cosine"},
                                    persist_directory="abc_vector_db_chroma",
                                    collection_name="abc_help_qa")


# code to load DB once saved

"""
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="abc_vector_db_chroma",
                     collection_name="abc_help_qa",
                     embedding_function=embeddings)

"""

########################################################################################################################
# 05 - TEST RETRIEVAL
########################################################################################################################


query = "What hours are you open on Easters?"


# 01. best k chunks based on similarity distance
top_docs_regular = vectorstore.similarity_search(query, k=4)

for i in top_docs_regular:
    print(i.metadata)
    print(i.page_content)


# 02. best k chunks based on similarity distance with distance score provided (lower = a better match)
top_docs_distance = vectorstore.similarity_search_with_score(query, k=4)

# print the chunks
for i in top_docs_distance:
    print(i[0].metadata)
    print(i[1])
    print(i[0].page_content)

# 03. best k chunks based on similarity distance with relevance (normalised similarity) score provided (higher = a better match)
top_docs_relevance = vectorstore.similarity_search_with_relevance_scores(query, k=4, score_threshold=0.40)

# print the chunks
for i in top_docs_relevance:
    print(i[0].metadata)
    print(i[1])
    print(i[0].page_content)


########################################################################################################################
# 06 - SET UP THE LLM ASSISTANT
########################################################################################################################

from langchain_openai import ChatOpenAI

abc_assistant_llm = ChatOpenAI(model="gpt-5",
                               temperature=0,
                               max_tokens=None,
                               max_retries=1)


########################################################################################################################
# 07 - SET UP THE PROMPT TEMPLATE
########################################################################################################################

from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(
"""
System Instructions: You are a helpful assistant for ABC Grocery - your job is to find the best solutions & answers for the customer's query.
Answer ONLY using the provided context. If the answer is not in the context, say that you don't have this information and encourage the customer to email human@abc-grocery.com

Context: {context}

Question: {input}

Answer:
"""
)

########################################################################################################################
# 08 - SET UP THE RETRIEVER
########################################################################################################################

# document retriever
retriever = vectorstore.as_retriever(search_type ="similarity_score_threshold", search_kwargs={"k": 6, "score_threshold": 0.25})


########################################################################################################################
# 09 - SET UP THE RETRIEVAL CHAIN
########################################################################################################################

from langchain_core.runnables import RunnableLambda
from operator import itemgetter

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# RAG answer chain: {input} -> retrieve -> format -> prompt -> model -> string
rag_answer_chain = (
    {
        "context": itemgetter("input") | retriever | RunnableLambda(format_docs),
        "input": itemgetter("input"),
    }
    | prompt_template
    | abc_assistant_llm
)


########################################################################################################################
# 10 - QUERY THE ASSISTANT
########################################################################################################################

user_prompt = ("What time can I come into the store today?")
user_prompt = ("What products do you sell?")
user_prompt = ("The shoes I bought are damaged.  What should I do?")
user_prompt = ("What is a baby dolphin called?")

# invoke the retrieval chain and get response
response = rag_answer_chain.invoke({"input": user_prompt})
print(response.content)

########################################################################################################################
# 11 - CHECK THE CONTEXT USED
########################################################################################################################

from langchain_core.runnables import RunnableParallel

# to also bring through context and user query for analysis
rag_with_context = RunnableParallel(answer=rag_answer_chain,
                                    context=itemgetter("input") | retriever,
                                    input=itemgetter("input"))

user_prompt = ("What time can I come into the store today?")

# invoke
response = rag_with_context.invoke({"input": user_prompt})
print(response["answer"].content)

print(response["context"])
