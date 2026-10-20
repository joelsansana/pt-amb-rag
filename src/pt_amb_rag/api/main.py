import json
import time

from fastapi import Depends, FastAPI
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse

from pt_amb_rag.api.ask import ask
from pt_amb_rag.api.auth import require_token
from pt_amb_rag.api.schemas import AskRequest, SearchRequest, SourceChunk
from pt_amb_rag.retrieval.vector_store import embed, search

app = FastAPI(title="pt-amb-rag", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


# Plain `def`: FastAPI runs it in a threadpool, so the blocking
# embed/search calls don't stall the event loop.
@app.post("/search", dependencies=[Depends(require_token)])
def search_endpoint(req: SearchRequest):
    q_emb = embed([req.question])[0]
    results = search(
        q_emb,
        k=req.k,
        regulation=req.regulation,
        article_num=req.article_num,
        part=req.part,
    )
    return {
        "results": [
            SourceChunk(score=r.score, **(r.payload or {})).model_dump()
            for r in results
        ]
    }


@app.post("/ask", dependencies=[Depends(require_token)])
async def ask_endpoint(req: AskRequest):
    async def event_stream():
        t0 = time.monotonic()
        try:
            result = await run_in_threadpool(
                ask,
                req.question,
                k=req.k,
                regulation=req.regulation,
                article_num=req.article_num,
                part=req.part,
            )
            sources = [SourceChunk(**s).model_dump() for s in result["sources"]]
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"detail": str(e)})}
            return

        yield {"event": "answer", "data": result["answer"]}
        yield {"event": "sources", "data": json.dumps(sources, default=str)}
        yield {
            "event": "done",
            "data": json.dumps(
                {"latency_ms": int((time.monotonic() - t0) * 1000)}
            ),
        }

    return EventSourceResponse(event_stream())
