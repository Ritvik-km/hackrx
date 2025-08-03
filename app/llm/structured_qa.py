import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from openai import AsyncOpenAI
import tiktoken
import asyncio

from typing import List, Dict
from langchain.schema import Document
import json
from dotenv import load_dotenv
import os

load_dotenv()

endpoint = "https://models.github.ai/inference"
# model = "openai/gpt-4.1"
model = "openai/gpt-4o"
# model = "meta/Llama-4-Scout-17B-16E-Instruct"
token = os.environ["GITHUB_TOKEN"]

client = AsyncOpenAI(
    base_url=endpoint,
    api_key=token,
)

# client = ChatCompletionsClient(
#     endpoint=endpoint,
#     credential=AzureKeyCredential(token),
# )

async def ask_llm_structured(questions: List[str], retrieved_clauses: List[Document]) -> Dict:
    enc = tiktoken.encoding_for_model("gpt-4")
    max_token_budget = 5000  # leave room for instructions + questions
    used_tokens = 0
    safe_chunks = []

    for clause in retrieved_clauses:
        tokens = len(enc.encode(clause.page_content))
        if used_tokens + tokens <= max_token_budget:
            safe_chunks.append(clause)
            used_tokens += tokens
        else:
            break

    print(f"🧠 Included {len(safe_chunks)} clauses ({used_tokens} tokens) out of {len(retrieved_clauses)}")

    # Generate instruction prompt
    instruction = (
        "You are a domain-aware assistant that answers structured insurance, legal, HR, or compliance queries.\n"
        "For each of the user's questions, answer in simple language using the context below.\n"
        # "Return a list of answers in JSON format like: { \"answers\": [q1_answer, q2_answer, ...] }\n"
        "Return a plain JSON object (no markdown), like: { \"answers\": [q1_answer, q2_answer, ...] }\n"

    )

    # Combine context and questions
    context = "\n\n".join([doc.page_content for doc in safe_chunks])
    joined_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

    try:
        print("📨 Sending request to LLM...")
        # response = client.complete(
        response = await client.chat.completions.create(
            messages=[
                # SystemMessage(content=instruction),
                # UserMessage(content=f"Context:\n{context}\n\nQuestions:\n{joined_questions}")
                {"role": "system", "content": instruction},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestions:\n{joined_questions}"}
            ],
            temperature=0.4,
            top_p=1.0,
            model=model,
        )
        print("✅ LLM responded.")

    except Exception as e:
        print("❌ Error from LLM call:", e)
        raise

    # return json.loads(response.choices[0].message.content)
    try:
        content = response.choices[0].message.content

        if content.strip().startswith("```"):
            content = content.strip().strip("`").strip("json").strip()

        return json.loads(content)
    except json.JSONDecodeError as je:
        print("❌ JSON decode error:", je)
        print("🔎 Raw content:\n", content)
        raise

async def answer_single_question(question: str, context: str, instruction: str) -> str:
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
            ],
            temperature=0.4,
            top_p=1.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error answering: {question}\n", e)
        return "Error"


async def ask_llm_structured_parallel(questions: List[str], retrieved_clauses: List[Document]) -> Dict:
    enc = tiktoken.encoding_for_model("gpt-4")
    max_token_budget = 5000
    used_tokens = 0
    safe_chunks = []

    for clause in retrieved_clauses:
        tokens = len(enc.encode(clause.page_content))
        if used_tokens + tokens <= max_token_budget:
            safe_chunks.append(clause)
            used_tokens += tokens
        else:
            break

    print(f"🧠 Included {len(safe_chunks)} clauses ({used_tokens} tokens) out of {len(retrieved_clauses)})")

    context = "\n\n".join([doc.page_content for doc in safe_chunks])
    instruction = (
        "You are a domain-aware assistant that answers structured insurance, legal, HR, or compliance queries.\n"
        "Respond in a complete sentence, without markdown. Return only the answer string."
    )

    tasks = [answer_single_question(q, context, instruction) for q in questions]
    answers = await asyncio.gather(*tasks)

    return { "answers": answers }


# def ask_llm_structured(questions: List[str], retrieved_clauses: List[Document]) -> Dict:
#     import re
#     enc = tiktoken.encoding_for_model("gpt-4")
#     max_token_budget = 6000
#     used_tokens = 0
#     safe_chunks = []

#     for clause in retrieved_clauses:
#         tokens = len(enc.encode(clause.page_content))
#         if used_tokens + tokens <= max_token_budget:
#             safe_chunks.append(clause)
#             used_tokens += tokens
#         else:
#             break

#     print(f"🧠 Included {len(safe_chunks)} clauses ({used_tokens} tokens) out of {len(retrieved_clauses)}")

#     instruction = (
#         "You are a domain-aware assistant that answers structured insurance, legal, HR, or compliance queries.\n"
#         "For each of the user's questions, answer in simple language using the context below.\n"
#         "Return a plain JSON object (no markdown), like: { \"answers\": [q1_answer, q2_answer, ...] }"
#     )

#     context = "\n\n".join([doc.page_content for doc in safe_chunks])
#     joined_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

#     try:
#         print("📨 Streaming request to LLM...")

#         stream = client.chat.completions.create(
#             messages=[
#                 {"role": "system", "content": instruction},
#                 {"role": "user", "content": f"Context:\n{context}\n\nQuestions:\n{joined_questions}"}
#             ],
#             model=model,
#             temperature=0.4,
#             top_p=1.0,
#             stream=True,
#         )

#         # 🧩 Accumulate streamed content
#         collected = ""
#         for part in stream:
#             if not part.choices or not hasattr(part.choices[0], "delta"):
#                 continue

#             token = getattr(part.choices[0].delta, "content", "")
#             print(token, end="", flush=True)
#             collected += token or ""


#     except Exception as e:
#         print("❌ LLM streaming error:", e)
#         raise

#     # 🧹 Strip Markdown formatting if accidentally included
#     if collected.strip().startswith("```"):
#         collected = collected.strip().strip("`").strip("json").strip()

#     try:
#         # Try to parse valid JSON object from collected text
#         json_match = re.search(r"\{.*\}", collected, re.DOTALL)
#         if not json_match:
#             raise ValueError("No valid JSON object found in streamed response.")

#         return json.loads(json_match.group(0))

#     except json.JSONDecodeError as je:
#         print("❌ JSON decode error:", je)
#         print("🔎 Raw streamed content:\n", collected)
#         raise
