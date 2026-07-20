#!/usr/bin/env python3
"""
Phase 2: 增强后端 — GraphRAG Q&A + 知识图谱管理
FastAPI backend with:
  - POST /api/query  → structured JSON with source citations
  - GET  /api/health → status
  - GET  /api/stats  → GraphRAG parquet stats
  - GET  /api/kg/list → available KGs metadata
  - POST /api/query/stream → SSE streaming
"""
import subprocess, json, os, re, time, asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="API-KG Q&A Backend v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GRAPH_ROOT = r"C:\Users\sino\Documents\graphrag_kg"
KG_FILES = {
    "cephalosporin": {"entities": 43, "relationships": 64, "apis": 25},
    "carbapenem_v2": {"entities": 46, "relationships": 63, "apis": 8},
    "statin_v2": {"entities": 29, "relationships": 33, "apis": 8},
    "arb": {"entities": 21, "relationships": 28, "apis": 9},
    "quinolone_skill": {"entities": 43, "relationships": 38, "apis": 11},
    "quinolone_evolution": {"entities": 40, "relationships": 45, "apis": 19},
    "hiv_arv": {"entities": 35, "relationships": 39, "apis": 12},
}

class QueryRequest(BaseModel):
    question: str
    method: str = "local"

@app.get("/api/health")
def health():
    output_exists = os.path.exists(f"{GRAPH_ROOT}/output/entities.parquet")
    return {
        "status": "ok",
        "graphrag_root": GRAPH_ROOT,
        "indexed": output_exists
    }

@app.get("/api/kg/list")
def list_kgs():
    return {"knowledge_graphs": KG_FILES, "total": len(KG_FILES)}

@app.get("/api/stats")
def get_stats():
    data_dir = f"{GRAPH_ROOT}/output"
    if not os.path.exists(f"{data_dir}/entities.parquet"):
        return {"error": "GraphRAG not indexed yet"}
    try:
        e = pd.read_parquet(f"{data_dir}/entities.parquet")
        r = pd.read_parquet(f"{data_dir}/relationships.parquet")
        cr = pd.read_parquet(f"{data_dir}/community_reports.parquet")
        return {
            "entities": int(len(e)),
            "relationships": int(len(r)),
            "reports": int(len(cr)),
            "entity_types": e["type"].value_counts().to_dict() if "type" in e.columns else {}
        }
    except Exception as ex:
        return {"error": str(ex)}

@app.post("/api/query")
def query_graphrag(req: QueryRequest):
    try:
        result = subprocess.run(
            ["python", "-m", "graphrag", "query",
             "--method", req.method, req.question],
            capture_output=True, text=True, timeout=180,
            cwd=GRAPH_ROOT
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr[-300:])

        raw_output = result.stdout.strip()
        answer = raw_output

        sources = []
        for kg_name in KG_FILES:
            if kg_name in raw_output:
                sources.append(f"kg_{kg_name}")

        entities_mentioned = re.findall(r'(?:KSM|DI|API|SUP|DRUG|GEN)_\w+', raw_output)

        return {
            "answer": answer,
            "question": req.question,
            "sources": sources if sources else ["graphrag_community_report"],
            "entities_mentioned": list(set(entities_mentioned))[:10],
            "method": req.method
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Query timed out after 180s")
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    async def generate():
        yield f"data: {json.dumps({'type':'start','question':req.question}, ensure_ascii=False)}\n\n"
        start = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", "graphrag", "query",
                "--method", req.method, req.question,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=GRAPH_ROOT
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
            elapsed = time.time() - start
            answer = stdout.decode('utf-8', errors='replace').strip()
            if proc.returncode != 0:
                answer = f"查询出错: {stderr.decode('utf-8',errors='replace')[-200:]}"
            yield f"data: {json.dumps({'type':'result','answer':answer,'elapsed':f'{elapsed:.1f}s'}, ensure_ascii=False)}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type':'error','answer':'查询超时(180s)'})}\n\n"
        yield "data: {\"type\":\"done\"}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
