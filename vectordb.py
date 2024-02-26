import pandas as pd
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
# from pinecone import Pinecone
from tqdm.auto import tqdm
from pathlib import Path  
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone_key import api_key,pinecone_env,index
from embedding import embedd_llm

directory_path = Path("Data")
loader = DirectoryLoader(directory_path, glob="*.pdf",  loader_cls=PyPDFLoader,) 
# Load all PDFs at once
documents = loader.load()


#chunking and splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs_split = text_splitter.split_documents(documents)

pinecone=pinecone.init(api_key=api_key,environment=pinecone_env)
index= index

'''the below "only two lines" are from Hassan's code,@hassanspaceimam for pinecone-client=2.2.4 version'''
docsearch=Pinecone.from_texts([t.page_content for t in docs_split], embedd_llm, index_name=index)
vectorstore = Pinecone.from_existing_index(index, embedd_llm)

'''pinecone setup for 'datalab server' "pinecone-client==3.0.0". pinecone-client=2.2.4 may depricate in near future'''
# pinecone = Pinecone(api_key=api_key,environment=pinecone_env)
# index= pinecone.Index(index)

# #loading data into dataframe
# datalist = []
# sourcelist = []

# # Extract text and metadata from each document
# for document in documents:
#     page_content = document.page_content
#     metadata = document.metadata
#     source = metadata['source']
#     text = page_content

#     datalist.append(text)
#     sourcelist.append(source)

# # Create the DataFrame
# dataframe_conversion = pd.DataFrame(data={"ID": range(1, len(datalist) + 1),
#                                         "Text": datalist,
#                                         "Source": sourcelist})
# dataframe_conversion = dataframe_conversion.replace('\n','.', regex=True).replace('\n\n','.',regex=True).replace('\n\n\n','.',regex=True)


# #data loaded into vectordb
# # embedd_llm = HuggingFaceEmbeddings()

# batch_size=10
# for i in tqdm(range(0,len(dataframe_conversion),batch_size)):
#     i_end=min(len(dataframe_conversion),i+batch_size)
#     batch=dataframe_conversion.iloc[i:i_end]
#     ids = [f"{x['ID']}" for _, x in batch.iterrows()]
#     content=[dataframe_conversion["Text"] for _,x in batch.iterrows()]
#     source=[dataframe_conversion["Source"] for _,x in batch.iterrows()]
#     combined_input = [f"{content} | {source}" for content, source in zip(content, source)]
#     embeds = embedd_llm.embed_documents(combined_input)
#     metadata = [
#         {'id':chunk['ID'],'text': "Text",'source': chunk['Source']}for _,chunk in batch.iterrows()
#      ]

#     index.upsert(vectors=zip(ids,embeds,metadata))



# text_field = 'text'

# vectorstore = Pinecone(
#     index, embedd_llm.embed_query, text_field
# )

