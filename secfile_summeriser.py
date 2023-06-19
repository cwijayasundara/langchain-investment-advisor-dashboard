import os
import pinecone

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

import investor_util

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')  # find next to api key in console
PINECONE_ENV = os.getenv('PINECONE_ENV')  # find next to api key in console
index_name = 'semantic-search-openai'
EMBEDDING_MODEL_NAME = 'text-embedding-ada-002'

llm = ChatOpenAI(temperature=0.0)

# embedding model
embed = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_key=OPENAI_API_KEY
)

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

# connect to index assuming its already created
index = pinecone.Index(index_name)
print('Pinecone index status is', index.describe_index_stats())

text_field = "text"
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)


# run the summarizer chain
def summerise_large_pdf(fileUrl):
    url = fileUrl
    loader = PyPDFLoader(url)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    return chain.run(texts)


# I want to store the summery text to a list
summery_text_list = []


def fill_in_ten_k_summery_text_list():
    for url in investor_util.ten_k_file_url_dict.values():
        summery_text = summerise_large_pdf(url)
        summery_text_list.append(summery_text)
    return summery_text_list


summery_text_list = fill_in_ten_k_summery_text_list()

# I want to loop the summery text list and print out the summery text
for summery_text in summery_text_list:
    print(summery_text)


# doc_db = Pinecone.from_documents(
#     summery_text_list,
#     embed,
#     index_name=index_name
# )

# query = "Search the summary of the FY23_Q2 Consolidated_Financial_Statements for Apple "
# search_docs = doc_db.similarity_search(query, top_k=3)
# print(search_docs)
