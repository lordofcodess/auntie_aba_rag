"""Test that cut-off point queries work."""
import os
os.environ["GEMINI_API_KEY"] = "dummy"

import sys, types
fake_genai = types.ModuleType("genai")
class FakeClient:
    def __init__(self, **kw): pass
fake_genai.Client = FakeClient
sys.modules["google.genai"] = fake_genai

sys.path.insert(0, ".")
from rag_chat import HandbookRAG

rag = HandbookRAG()

queries = [
    "what is the cut-off point for computer engineering?",
    "admission requirements for computer engineering at UG",
    "computer engineering aggregate",
    "BSc Computer Engineering cut-off point",
]

for q in queries:
    print(f"\n{'=' * 80}")
    print(f"QUERY: {q}")
    print("=" * 80)
    chunks = rag.retrieve(q, top_k=5)
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        print(f"[{i}] {m.get('source_file')} | {m.get('department') or 'no-dept'} | {m.get('content_type')}")
        snippet = c['text'][:150].replace('\n', ' ')
        print(f"    {snippet}")
