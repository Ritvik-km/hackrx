# app/rag_pipeline/embedder.py
from typing import List
import time
import random
from app.config import settings
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

from google import genai
from google.genai import types
from google.genai.errors import ClientError
from app.utils.gemini_cache import GeminiEmbeddingCache

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


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
        self.base_delay = 1.0  # Base delay between requests
        # self.cache = GeminiEmbeddingCache()

    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter"""
        delay = min(60, self.base_delay * (2 ** attempt))
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter

    @retry(
        retry=retry_if_exception_type(ClientError),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5)
    )
    def embed_single(self, text, config):
        """Embed a single text with retry logic"""
        result = self.client.models.embed_content(
            model=self.model,
            contents=[text],
            config=config
        )
        return result.embeddings[0]

    def embed_batch_with_fallback(self, texts: List[str], indices: List[int], config, all_embeddings):
        """
        Try to embed a batch, falling back to individual embeddings on rate limit
        """
        try:
            # Try batch embedding first
            result = self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config=config
            )
            
            for j, embed in enumerate(result.embeddings):
                global_index = indices[j]
                all_embeddings[global_index] = embed.values
                # self.cache.set(texts[j], embed.values)
                
            print(f"✅ Successfully embedded batch of {len(texts)} texts")
            
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"⚠️ Rate limit hit for batch of {len(texts)}, falling back to individual embeddings with delays")
                self._embed_individually_with_delays(texts, indices, config, all_embeddings)
            else:
                print(f"❌ Non-rate-limit error in batch: {e}")
                raise
        except Exception as e:
            print(f"❌ Unexpected error in batch: {e}")
            raise

    def _embed_individually_with_delays(self, texts: List[str], indices: List[int], config, all_embeddings):
        """
        Embed texts individually with progressive delays to handle rate limits
        """
        for i, (text, global_index) in enumerate(zip(texts, indices)):
            attempt = 0
            max_attempts = 3
            
            while attempt < max_attempts:
                try:
                    result = self.embed_single(text, config)
                    all_embeddings[global_index] = result.values
                    # self.cache.set(text, result.values)
                    print(f"✅ Embedded individual text {i+1}/{len(texts)}")
                    break
                    
                except ClientError as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        attempt += 1
                        if attempt < max_attempts:
                            delay = self._exponential_backoff(attempt)
                            print(f"⏳ Rate limited on individual text {i+1}/{len(texts)}, waiting {delay:.1f}s (attempt {attempt}/{max_attempts})")
                            time.sleep(delay)
                        else:
                            print(f"❌ Failed to embed text after {max_attempts} attempts: {str(e)[:100]}")
                            raise
                    else:
                        print(f"❌ Non-rate-limit error on individual text: {e}")
                        raise
            
            # Add a small delay between successful individual embeddings to be respectful
            if i < len(texts) - 1:  # Don't delay after the last item
                time.sleep(0.5)

    def embed_documents(self, texts: List[str], output_dim: int = 3072, task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=output_dim
        )

        all_embeddings = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []

        # Check cache (currently commented out but structure preserved)
        for i, text in enumerate(texts):
            # cached = self.cache.get(text)
            # if cached:
            #     all_embeddings[i] = cached
            # else:
            uncached_texts.append(text)
            uncached_indices.append(i)

        if not uncached_texts:
            return all_embeddings

        print(f"🔢 Embedding {len(uncached_texts)} uncached texts...")
        
        # Adaptive batch sizing based on total tokens
        total_tokens = sum(len(text.split()) for text in uncached_texts)
        print(f"🔢 Total tokens to embed: {total_tokens}")
        
        # Adjust batch size based on total load
        if total_tokens > 50000:
            MAX_TOKENS_PER_BATCH = 5000  # Smaller batches for large loads
        elif total_tokens > 20000:
            MAX_TOKENS_PER_BATCH = 6000  # Medium batches
        else:
            MAX_TOKENS_PER_BATCH = 8000  # Original size for small loads

        current_batch = []
        current_batch_indices = []
        current_token_count = 0
        batch_count = 0

        for text, global_index in zip(uncached_texts, uncached_indices):
            token_count = len(text.split())
            
            # Check if adding this text would exceed batch limit
            if current_token_count + token_count > MAX_TOKENS_PER_BATCH and current_batch:
                batch_count += 1
                print(f"🚀 Processing batch {batch_count} with {len(current_batch)} texts ({current_token_count} tokens)")
                
                self.embed_batch_with_fallback(current_batch, current_batch_indices, config, all_embeddings)
                
                # Reset for next batch
                current_batch = []
                current_batch_indices = []
                current_token_count = 0
                
                # Add delay between batches to respect rate limits
                time.sleep(self.base_delay)

            current_batch.append(text)
            current_batch_indices.append(global_index)
            current_token_count += token_count

        # Process final batch if it exists
        if current_batch:
            batch_count += 1
            print(f"🚀 Processing final batch {batch_count} with {len(current_batch)} texts ({current_token_count} tokens)")
            self.embed_batch_with_fallback(current_batch, current_batch_indices, config, all_embeddings)

        print(f"✅ Completed embedding {len(uncached_texts)} texts in {batch_count} batches")
        return all_embeddings

    def embed_query(self, text: str, output_dim: int = 3072, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
        # cached = self.cache.get(text)
        # if cached:
        #     return cached

        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=output_dim
        )
        
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=[text],
                config=config
            )
            embedding = result.embeddings[0].values
            # self.cache.set(text, embedding)
            return embedding
            
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("⚠️ Rate limited on query embedding, retrying with delay...")
                time.sleep(self._exponential_backoff(1))
                # Retry once
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=[text],
                    config=config
                )
                embedding = result.embeddings[0].values
                # self.cache.set(text, embedding)
                return embedding
            else:
                raise
            

# # app/rag_pipeline/embedder.py
# from typing import List
# from app.config import settings
# from azure.ai.inference import EmbeddingsClient
# from azure.core.credentials import AzureKeyCredential

# from google import genai
# from google.genai import types
# from app.utils.gemini_cache import GeminiEmbeddingCache

# from tenacity import retry, wait_exponential, stop_after_attempt


# class GitHubEmbeddingModel:
#     def __init__(self):
#         self.client = EmbeddingsClient(
#             endpoint=settings.EMBEDDING_ENDPOINT,
#             credential=AzureKeyCredential(settings.GITHUB_TOKEN),
#         )
#         self.model = settings.EMBEDDING_MODEL

#     def embed_documents(self, texts: List[str], task_type: str = None) -> List[List[float]]:
#         response = self.client.embed(input=texts, model=self.model)
#         return [item.embedding for item in response.data]

#     def embed_query(self, text: str, task_type: str = None) -> List[float]:
#         return self.embed_documents([text])[0]


# class GoogleEmbeddingModel:
#     def __init__(self):
#         self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
#         self.model = "gemini-embedding-001"
#         # self.cache = GeminiEmbeddingCache()

#     @retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(5))
#     def embed_single(self, text, config):
#         result = self.client.models.embed_content(
#             model=self.model,
#             contents=[text],  # ✅ MUST be a list
#             config=config
#         )
#         return result.embeddings[0]  # ✅ Use [0] since it's a single-item call


#     # def embed_batch(self, batch_texts, batch_indices, config, all_embeddings):
#     #     from time import sleep
#     #     for j, text in enumerate(batch_texts):
#     #         try:
#     #             result = self.embed_single(text, config)
#     #             global_index = batch_indices[j]
#     #             all_embeddings[global_index] = result.values
#     #             self.cache.set(text, result.values) 
#     #             sleep(0.2)  # optional delay
#     #         except Exception as e:
#     #             print(f"❌ Error embedding text at index {batch_indices[j]}: {e}")
#     #             raise
#     def embed_batch(self, texts: List[str], indices: List[int], config, all_embeddings):
#         try:
          
#             result = self.client.models.embed_content(
#                 model=self.model,
#                 contents=texts,         # ✅ pass full batch here
#                 config=config
#             )
#             for j, embed in enumerate(result.embeddings):
#                 global_index = indices[j]
#                 all_embeddings[global_index] = embed.values
#                 # self.cache.set(texts[j], embed.values)
#         except Exception as e:
#             print(f"❌ Error embedding batch: {e}")
#             raise

#     def embed_documents(self, texts: List[str], output_dim: int = 3072, task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
#         config = types.EmbedContentConfig(
#             task_type=task_type,
#             output_dimensionality=output_dim
#         )

#         all_embeddings = [None] * len(texts)
#         uncached_texts = []
#         uncached_indices = []

#         for i, text in enumerate(texts):
#             # cached = self.cache.get(text)
#             # if cached:
#             #     all_embeddings[i] = cached
#             # else:
#             uncached_texts.append(text)
#             uncached_indices.append(i)

#         MAX_TOKENS_PER_BATCH = 8000
#         current_batch = []
#         current_batch_indices = []
#         current_token_count = 0

#         for text, global_index in zip(uncached_texts, uncached_indices):
#             token_count = len(text.split())
#             if current_token_count + token_count > MAX_TOKENS_PER_BATCH:
#                 self.embed_batch(current_batch, current_batch_indices, config, all_embeddings)
#                 current_batch = []
#                 current_batch_indices = []
#                 current_token_count = 0

#             current_batch.append(text)
#             current_batch_indices.append(global_index)
#             current_token_count += token_count

#         if current_batch:
#             self.embed_batch(current_batch, current_batch_indices, config, all_embeddings)

#         return all_embeddings



#     def embed_query(self, text: str, output_dim: int = 3072, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
#         # cached = self.cache.get(text)
#         # if cached:
#         #     return cached

#         config = types.EmbedContentConfig(
#             task_type=task_type,
#             output_dimensionality=output_dim
#         )
#         result = self.client.models.embed_content(
#             model=self.model,
#             contents=[text],
#             config=config
#             )
#         embedding = result.embeddings[0].values
#         # self.cache.set(text, embedding)
#         return embedding


