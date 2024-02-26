from langchain.prompts import PromptTemplate


prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}


**Please ensure the answer does not contain newline characters (\n) or hash characters (#).**

Only return the topest five most relevant answer below in only 4 sentences:

"""


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}
