from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import numpy as np
import os

app = FastAPI()

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class RequestBody(BaseModel):
query_id: str
query: str
candidates: list[str]

@app.post("/")
async def rank_passages(req: RequestBody):
texts = [req.query] + req.candidates

```
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

embeddings = [np.array(d.embedding, dtype=np.float32)
              for d in response.data]

query_emb = embeddings[0]
candidate_embs = embeddings[1:]

query_emb = query_emb / np.linalg.norm(query_emb)

scores = []

for idx, emb in enumerate(candidate_embs):
    emb = emb / np.linalg.norm(emb)
    sim = float(np.dot(query_emb, emb))
    scores.append((sim, idx))

scores.sort(key=lambda x: x[0], reverse=True)

top3 = [idx for _, idx in scores[:3]]

return {"ranking": top3}
```
