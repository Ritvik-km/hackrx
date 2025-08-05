# app/rag_pipeline/embedder.py
from typing import List
from app.config import settings
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

from google import genai
from google.genai import types
from app.utils.gemini_cache import GeminiEmbeddingCache

from tenacity import retry, wait_exponential, stop_after_attempt


class GitHubEmbeddingModel:
    def __init__(self):
        self.client = EmbeddingsClient(
            endpoint=settings.EMBEDDING_ENDPOINT,
            credential=AzureKeyCredential(settings.GITHUB_TOKEN),
        )
        self.model = settings.EMBEDDING_MODEL

    def embed_documents(self, texts: List[str], task_type: str = None) -> List[List[float]]:
        response = self.client.embed(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str, task_type: str = None) -> List[float]:
        return self.embed_documents([text])[0]


class GoogleEmbeddingModel:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.model = "gemini-embedding-001"
        # self.cache = GeminiEmbeddingCache()

    @retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(5))
    def embed_single(self, text, config):
        result = self.client.models.embed_content(
            model=self.model,
            contents=[text],  # ✅ MUST be a list
            config=config
        )
        return result.embeddings[0]  # ✅ Use [0] since it's a single-item call


    # def embed_batch(self, batch_texts, batch_indices, config, all_embeddings):
    #     from time import sleep
    #     for j, text in enumerate(batch_texts):
    #         try:
    #             result = self.embed_single(text, config)
    #             global_index = batch_indices[j]
    #             all_embeddings[global_index] = result.values
    #             self.cache.set(text, result.values) 
    #             sleep(0.2)  # optional delay
    #         except Exception as e:
    #             print(f"❌ Error embedding text at index {batch_indices[j]}: {e}")
    #             raise
    def embed_batch(self, texts: List[str], indices: List[int], config, all_embeddings):
        try:
          
            result = self.client.models.embed_content(
                model=self.model,
                contents=texts,         # ✅ pass full batch here
                config=config
            )
            for j, embed in enumerate(result.embeddings):
                global_index = indices[j]
                all_embeddings[global_index] = embed.values
                # self.cache.set(texts[j], embed.values)
        except Exception as e:
            print(f"❌ Error embedding batch: {e}")
            raise

    def embed_documents(self, texts: List[str], output_dim: int = 3072, task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=output_dim
        )

        all_embeddings = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            # cached = self.cache.get(text)
            # if cached:
            #     all_embeddings[i] = cached
            # else:
            uncached_texts.append(text)
            uncached_indices.append(i)

        MAX_TOKENS_PER_BATCH = 8000
        current_batch = []
        current_batch_indices = []
        current_token_count = 0

        for text, global_index in zip(uncached_texts, uncached_indices):
            token_count = len(text.split())
            if current_token_count + token_count > MAX_TOKENS_PER_BATCH:
                self.embed_batch(current_batch, current_batch_indices, config, all_embeddings)
                current_batch = []
                current_batch_indices = []
                current_token_count = 0

            current_batch.append(text)
            current_batch_indices.append(global_index)
            current_token_count += token_count

        if current_batch:
            self.embed_batch(current_batch, current_batch_indices, config, all_embeddings)

        return all_embeddings



    def embed_query(self, text: str, output_dim: int = 3072, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
        # cached = self.cache.get(text)
        # if cached:
        #     return cached

        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=output_dim
        )
        result = self.client.models.embed_content(
            model=self.model,
            contents=[text],
            config=config
            )
        embedding = result.embeddings[0].values
        # self.cache.set(text, embedding)
        return embedding
    
    
# Working but without caching -----------------------------------------------------------
# class GoogleEmbeddingModel:
#     def __init__(self):
#         self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
#         self.model = "gemini-embedding-001"

#     def embed_documents(self, texts: List[str], output_dim: int = 3072, task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
#         config = types.EmbedContentConfig(
#             task_type=task_type,
#             output_dimensionality=output_dim
#         )
#         # result = self.client.models.embed_content(
#         #     model=self.model,
#         #     contents=texts,
#         #     config=config
#         # )
#         # return [e.values for e in result.embeddings]
#         all_embeddings = []
#         batch_size = 100  # Gemini's API limit

#         for i in range(0, len(texts), batch_size):
#             batch = texts[i:i + batch_size]
#             try:
#                 result = self.client.models.embed_content(
#                     model=self.model,
#                     contents=batch,
#                     config=config
#                 )
#                 all_embeddings.extend([e.values for e in result.embeddings])
#             except Exception as e:
#                 print(f"❌ Error embedding batch {i//batch_size + 1}: {e}")
#                 raise

#         return all_embeddings

    # def embed_query(self, text: str, output_dim: int = 3072, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
    #     config = types.EmbedContentConfig(
    #         task_type=task_type,
    #         output_dimensionality=output_dim
    #     )
    #     result = self.client.models.embed_content(
    #         model=self.model,
    #         contents=[text],
    #         config=config
    #     )
    #     return result.embeddings[0].values


# class GoogleEmbeddingModel:
    # def __init__(self):
    #     self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    #     self.model = "gemini-embedding-001"

    # def embed_documents(self, texts: List[str], output_dim: int = 3072) -> List[List[float]]:
    #     # Use EmbedContentConfig to specify task type and dimensionality
    #     config = types.EmbedContentConfig(
    #         task_type="RETRIEVAL_DOCUMENT",
    #         output_dimensionality=output_dim
    #     )
    #     result = self.client.models.embed_content(
    #         model=self.model,
    #         contents=texts,
    #         config=config
    #     )
    #     # Each embedding is a types.Embedding with .values
    #     return [e.values for e in result.embeddings]

    # # def embed_query(self, text: str, output_dim: int = 3072) -> List[float]:
    # #     return self.embed_documents([text], output_dim=output_dim)[0]

    # def embed_query(self, text: str, output_dim: int = 3072) -> List[float]:
    #     config = types.EmbedContentConfig(
    #         task_type="RETRIEVAL_QUERY",  # optimized for queries
    #         output_dimensionality=output_dim
    #     )
    #     result = self.client.models.embed_content(
    #         model=self.model,
    #         contents=[text],
    #         config=config
    #     )
    #     return result.embeddings[0].values

