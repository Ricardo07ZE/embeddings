# imports
import pandas as pd
import tiktoken
import openai
import os

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191


# Functions definition
def get_embedding(text, model=embedding_model):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


# Document file
dir_name = os.path.dirname(__file__)  # get current directory
document_to_embedded = os.path.join(dir_name, 'Refined_text_santander.csv')
facts_df = pd.read_csv(document_to_embedded, encoding='latin1')

# Prepare encoding
encoding = tiktoken.get_encoding(embedding_encoding)

# Tokens count
facts_df['Tokens'] = facts_df.apply(
    lambda x: len(encoding.encode(x["Content"])),
    axis=1
)

# Getting embeddings
facts_df['Embeddings'] = facts_df.apply(
        lambda x: get_embedding(x["Content"]),
        axis=1
    )

facts_df.to_csv('refined_santander_embeddings.csv', encoding='ISO-8859-1')
