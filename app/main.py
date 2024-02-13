import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import torch
import torch.nn as nn
import json
import nltk

# Define the Skipgram neural network model
class SkipgramModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_dimension):
        super(SkipgramModel, self).__init__()
        # Match these names with the keys in the state_dict
        self.embedding_center = nn.Embedding(vocabulary_size, embedding_dimension)
        self.embedding_outside = nn.Embedding(vocabulary_size, embedding_dimension)

    def forward(self, center_words, context_words, vocabulary_indices):
        # The rest of your forward method remains unchanged
        center_vect = self.center_embedding(center_words)
        context_vect = self.context_embedding(context_words)
        vocab_vect = self.context_embedding(vocabulary_indices)
        dot_product_context = torch.exp(context_vect.bmm(center_vect.transpose(1, 2)).squeeze(2))
        dot_product_vocab = vocab_vect.bmm(center_vect.transpose(1, 2)).squeeze(2)
        sum_dot_product = torch.sum(torch.exp(dot_product_vocab), 1)
        loss = -torch.mean(torch.log(dot_product_context / sum_dot_product))
        return loss

# Load and process the text corpus
def process_corpus(filepath):
    corpus_collection = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                corpus_collection.append(line)
    return corpus_collection

# Tokenize and preprocess input text
def text_tokenize(input_text):
    tokenized = nltk.word_tokenize(input_text.lower())
    return tokenized

# Calculate text embedding
def calculate_text_embedding(input_text, skipgram_model, word_to_index):
    tokens = text_tokenize(input_text)  # Assuming text_tokenize is your tokenization function
    embeddings_list = []
    for token in tokens:
        index = word_to_index.get(token, word_to_index.get('<UNK>'))  # Handle unknown tokens
        word_tensor = torch.LongTensor([index])
        if 0 <= index < skipgram_model.embedding_center.weight.shape[0]:  # Use corrected attribute names
            center_embed = skipgram_model.embedding_center(word_tensor)
            context_embed = skipgram_model.embedding_outside(word_tensor)
            embed = (center_embed + context_embed) / 2
            embeddings_list.append(embed.detach().numpy())
        else:
            embeddings_list.append(np.zeros(skipgram_model.embedding_center.weight.shape[1]))  # Use corrected attribute name
    if embeddings_list:
        embeddings_array = np.array(embeddings_list)
        average_embedding = np.mean(embeddings_array, axis=0)
    else:
        average_embedding = np.zeros(skipgram_model.embedding_center.weight.shape[1])  # Use corrected attribute name
    return average_embedding.flatten()


# Find passages similar to the query
def find_similarities(query, text_collection, skipgram_model, word_to_index, top_n=10):
    query_embed = calculate_text_embedding(query, skipgram_model, word_to_index)
    similarity_scores = []
    for passage in text_collection:
        passage_embed = calculate_text_embedding(passage, skipgram_model, word_to_index)
        similarity = np.dot(query_embed, passage_embed)
        similarity_scores.append(similarity)
    top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:top_n]
    similar_passages = [(text_collection[idx], similarity_scores[idx] * 100) for idx in top_indices]
    return similar_passages

# Load model configuration and initialize the model
config_file = 'word2vec_config_skipgram.json'
with open(config_file, 'r') as file:
    config = json.load(file)

skipgram_model = SkipgramModel(vocabulary_size=config['voc_size'], embedding_dimension=config['emb_size'])
model_weights = 'word2vec_model_skipgram.pth'
skipgram_model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
skipgram_model.eval()

# Load word to index mapping and corpus
word_to_index_file = 'word2index_skipgram.json'
with open(word_to_index_file, 'r') as file:
    word_to_index = json.load(file)

text_collection = process_corpus('corpus.txt')

# Initialize Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Custom styles for components
CUSTOM_STYLE = {
    'queryInput': {
        'marginBottom': '1rem',
    },
    'findButton': {
        'width': '100%',
        'marginBottom': '1rem',
    },
    'resultsDiv': {
        'marginTop': '1rem',
    },
    'footer': {
        'marginTop': '2rem',
        'textAlign': 'center',
        'padding': '1rem',
        'backgroundColor': '#f8f9fa',
        'borderTop': '1px solid #e9ecef',
        'fontSize': '0.9rem',
    }
}

# Define the layout of the Dash app
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Advanced Text Search"), className="mb-3")),
    dbc.Row(dbc.Col(dcc.Input(id="input_query", type="text", placeholder="Type your query here", style=CUSTOM_STYLE['queryInput']))),
    dbc.Row(dbc.Col(html.Button("Find Similar", id="find_button", n_clicks=0, className="btn btn-primary", style=CUSTOM_STYLE['findButton']), width={"size": 6, "offset": 3})),
    dbc.Row(dbc.Col(html.Div(id="query_results", style=CUSTOM_STYLE['resultsDiv']))),
    dbc.Row(dbc.Col(html.Div("Developed by Tanzil Al Sabah - ST123845 - AIT - IoT", style=CUSTOM_STYLE['footer'])))
], fluid=True, className="py-3")

@app.callback(
    Output("query_results", "children"),
    Input("find_button", "n_clicks"),
    State("input_query", "value"),
)
def update_query_results(n_clicks, input_query):
    if n_clicks > 0 and input_query:
        similar_passages = find_similarities(input_query, text_collection, skipgram_model, word_to_index, top_n=10)
        return [html.Div([
            html.P(f"Result {rank+1}: {passage}", className="passage-text"),
        ], className="passage-result") for rank, (passage, score) in enumerate(similar_passages)]
    return []

if __name__ == '__main__':
    app.run_server(debug=True)
