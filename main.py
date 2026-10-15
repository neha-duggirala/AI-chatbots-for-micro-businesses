from ingestion.vector_store import build_vector_stores
from rag_chain import build_rag_chain

query = "What time will my milk be delivered"

# 1. Build Vector Stores
stores = build_vector_stores()
print("=" * 60)
print(f"📦 VECTOR STORES INITIALIZED")
print("-" * 60)
print(f"• Total Stores Built: {len(stores)}")
print(f"• Store Keys       : {list(stores.keys())}")
print("=" * 60)

selected = list(stores.values())
chain = build_rag_chain(selected)

# 2. Pretty Print Query & Response
print(f"\n💬 QUERY")
print(f"\"{query}\"")
print("-" * 60)

print("🤖 GENERATED ANSWER:")
response = chain.invoke(query)

# Handle response whether it's a dictionary or a string
if isinstance(response, dict):
    # Extract common answer keys used in chains
    answer_text = response.get("answer") or response.get("output") or str(response)
else:
    answer_text = str(response)

print(f"\n{answer_text.strip()}")
print("=" * 60)