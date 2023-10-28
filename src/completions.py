# imports
import ast

import pandas as pd
import tiktoken
import openai
import os

from scipy import spatial

# embedding model parameters
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

# Embeddings file
dir_name = os.path.dirname(__file__)  # get current directory
document_to_embedded = os.path.join(dir_name, 'refined_santander_embeddings.csv')
embeddings_df = pd.read_csv(document_to_embedded, index_col=0, encoding='ISO-8859-1')

# print(embeddings_df)

# convert embeddings from CSV str type back to list type
embeddings_df['embedding'] = embeddings_df['Embeddings'].apply(ast.literal_eval)

# print(embeddings_df)

embeddings_df.to_csv('testing.csv', encoding='ISO-8859-1')


# search function
def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatedness = [
        (row["Content"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatedness.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatedness)
    return strings[:top_n], relatednesses[:top_n]


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Contesta de ser posible la siguiente pregunta con la informacion presentada, de no ser posible ' \
                   'responde "Sin informacion disponible" \n Informacion: \n'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n{string}\n'
        if (
                num_tokens(message + next_article + question, model=model)
                > token_budget
        ):
            break
        else:
            message += next_article
    # print(f"Message: {message}")
    return message + question


def ask(
        query: str,
        df: pd.DataFrame = embeddings_df,
        model: str = GPT_MODEL,
        token_budget: int = 1024 - 500,
        print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "Responde como si fueras un chatbot de servicio"},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message


# # examples
# strings, relatednesses = strings_ranked_by_relatedness("aplicacion movil", embeddings_df, top_n=5)
# for string, relatedness in zip(strings, relatednesses):
#     print(f"{relatedness=:.3f}")
#     print(string)

answer = ask('Para que sirve la aplicación movil?', print_message=True)
print(answer)
