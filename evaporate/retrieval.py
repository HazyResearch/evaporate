import torch
from transformers import AutoTokenizer, AutoModel


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_embeddings(sentences):
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever')

    sentences = [
        "Where was Marie Curie born?",
        "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
        "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
    ]

    # Apply tokenizer
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    outputs = model(**inputs)

    # Mean pooling

    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    return embeddings

def get_most_similarity(target_sentence, sentences):
    target_embedding = get_embeddings([target_sentence])[0]
    embeddings = get_embeddings(sentences)
    max_similarity = torch.nn.functional.cosine_similarity(target_embedding, embeddings, dim = -1)
    most_similar_sentence = max_similarity.argmax()
    return sentences[most_similar_sentence]

#write a main function to test the code with if __name__ == "__main__":
if __name__ == "__main__":
    target_sentence = "Where was Marie Curie born?"
    sentences = [
        "Where was Marie Curie born?",
        "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
        "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
    ]
    print(get_most_similarity(target_sentence, sentences))
