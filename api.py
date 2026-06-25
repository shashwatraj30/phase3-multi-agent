import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile
from pydantic import BaseModel
from main import run_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

class ChatRequest(BaseModel):
    question: str
    context: str

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage
        from dotenv import load_dotenv
        load_dotenv()

        # Truncate context to last 3000 chars — prevents token blowup on long sessions
        context = req.context[-3000:] if len(req.context) > 3000 else req.context

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        response = llm.invoke([HumanMessage(content=(
            "You are a research assistant helping a user understand academic papers.\n\n"
            f"Here is the analysis context from the uploaded papers:\n{context}\n\n"
            f"User question: {req.question}\n\n"
            "Answer concisely and accurately based only on the provided context."
        ))])
        return {"status": "success", "answer": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    temp_dir = tempfile.mkdtemp()
    pdf_paths = []

    try:
        for file in files:
            if not file.filename.endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")
            path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
            with open(path, "wb") as f:
                f.write(await file.read())
            pdf_paths.append(path)

        result = run_pipeline(pdf_paths)
        return {"status": "success", "report": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)