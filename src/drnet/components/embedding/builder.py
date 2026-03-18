from .embedding import (
    IDXEmbedding,
    IDXEmbeddingWithHistory,
)


def embedding_builder(history, **kwargs):
    cls = (
        IDXEmbeddingWithHistory
        if history==True
        else IDXEmbedding
    )
    return cls(**kwargs)