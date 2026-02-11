from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import json
import uuid
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timedelta

from model import get_model, generate_questions_local

# ---------- In-memory exam storage ----------
exams_store: Dict[str, Dict[str, Any]] = {}

# ---------- Pydantic models ----------
class ExamCreateResponse(BaseModel):
    exam_id: str
    questions: List[Dict[str, Any]]
    created_at: datetime

class AnswerItem(BaseModel):
    question_id: int
    selected_option_index: int  # 0=A, 1=B, 2=C, 3=D

class ExamSubmitRequest(BaseModel):
    answers: List[AnswerItem]

class ExamSubmitResponse(BaseModel):
    score: float
    total_questions: int
    correct_count: int
    wrong_count: int
    percentage: float
    wrong_answers: List[Dict[str, str]]

# ---------- FastAPI app ----------
app = FastAPI(
    title="Exam Generation API (Local LLM)",
    description=(
        "Generate and take exams using a local HuggingFace model. "
        "Supports Bangla and English. Upload knowledge files, set parameters, get scored results."
    ),
    version="3.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Pre-load the model on startup so first request is fast."""
    print("🚀 Loading local model on startup...")
    get_model()
    print("✅ Model ready.")

@app.post("/exam/create", response_model=ExamCreateResponse)
async def create_exam(
    knowledge_files: List[UploadFile] = File(..., description="One or more .txt files (Bangla or English)"),
    pattern_file: Optional[UploadFile] = File(None, description="Optional .txt file with extra generation instructions"),
    num_questions: int = Form(10, ge=1, le=50, description="Number of questions to generate"),
    duration_minutes: int = Form(30, ge=1, le=180, description="Exam duration in minutes"),
    correct_mark: float = Form(1.0, ge=0, description="Points per correct answer"),
    wrong_penalty: float = Form(0.0, ge=0, description="Points deducted per wrong answer"),
    language: str = Form("auto", description="Language hint: 'bangla', 'english', or 'auto'"),
):
    # 1. Read knowledge files
    context_parts = []
    for f in knowledge_files:
        if not f.filename.lower().endswith(".txt"):
            raise HTTPException(400, f"Only .txt files allowed. Got: {f.filename}")
        context_parts.append(f.file.read().decode("utf-8"))
    context = "\n\n".join(context_parts)

    # 2. Read optional pattern file
    pattern = ""
    if pattern_file:
        if not pattern_file.filename.lower().endswith(".txt"):
            raise HTTPException(400, "Pattern file must be .txt")
        pattern = pattern_file.file.read().decode("utf-8")

    # 3. Generate questions via local model
    try:
        questions = generate_questions_local(context, num_questions, pattern, language)
    except Exception as e:
        raise HTTPException(500, f"Question generation failed: {str(e)}")

    if not questions:
        raise HTTPException(500, "Model returned no questions. Try reducing num_questions or simplifying context.")

    # 4. Store exam session
    exam_id = str(uuid.uuid4())
    start_time = datetime.utcnow()

    exams_store[exam_id] = {
        "questions": questions,
        "start_time": start_time,
        "duration_minutes": duration_minutes,
        "correct_mark": correct_mark,
        "wrong_penalty": wrong_penalty,
        "submitted": False,
        "answers": None,
        "result": None,
        "language": language,
    }

    return ExamCreateResponse(
        exam_id=exam_id,
        questions=questions,
        created_at=start_time,
    )

@app.post("/exam/submit/{exam_id}", response_model=ExamSubmitResponse)
async def submit_exam(exam_id: str, request: ExamSubmitRequest):
    if exam_id not in exams_store:
        raise HTTPException(404, "Exam not found")

    exam = exams_store[exam_id]
    if exam["submitted"]:
        raise HTTPException(400, "Exam already submitted")

    # Check time limit
    elapsed = datetime.utcnow() - exam["start_time"]
    if elapsed > timedelta(minutes=exam["duration_minutes"]):
        raise HTTPException(400, f"Exam time expired (duration: {exam['duration_minutes']} min)")

    questions = exam["questions"]
    correct_mark = exam["correct_mark"]
    wrong_penalty = exam["wrong_penalty"]

    answers_dict = {item.question_id: item.selected_option_index for item in request.answers}
    correct_count = 0
    wrong_count = 0
    wrong_details = []

    for idx, q in enumerate(questions):
        correct_idx = ord(q["correct"].strip().upper()) - ord("A")
        selected_idx = answers_dict.get(idx)

        if selected_idx is None:
            wrong_count += 1
            wrong_details.append({
                "question": q["question"],
                "your_answer": "No answer provided",
                "correct_answer": q["options"][correct_idx],
                "explanation": q.get("explanation", ""),
            })
        elif selected_idx == correct_idx:
            correct_count += 1
        else:
            wrong_count += 1
            wrong_details.append({
                "question": q["question"],
                "your_answer": q["options"][selected_idx],
                "correct_answer": q["options"][correct_idx],
                "explanation": q.get("explanation", ""),
            })

    total = len(questions)
    score = correct_count * correct_mark - wrong_count * wrong_penalty
    max_score = total * correct_mark
    percentage = (score / max_score * 100) if max_score > 0 else 0.0

    result = {
        "score": score,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "percentage": percentage,
        "wrong_answers": wrong_details,
    }
    exam["submitted"] = True
    exam["answers"] = answers_dict
    exam["result"] = result

    return ExamSubmitResponse(total_questions=total, **result)

@app.get("/exam/{exam_id}")
async def get_exam_status(exam_id: str):
    if exam_id not in exams_store:
        raise HTTPException(404, "Exam not found")
    exam = exams_store[exam_id]
    elapsed = (datetime.utcnow() - exam["start_time"]).total_seconds()
    remaining = max(0, exam["duration_minutes"] * 60 - elapsed)
    return {
        "exam_id": exam_id,
        "created_at": exam["start_time"],
        "duration_minutes": exam["duration_minutes"],
        "time_remaining_seconds": int(remaining),
        "submitted": exam["submitted"],
        "num_questions": len(exam["questions"]),
        "language": exam.get("language", "auto"),
        "result": exam["result"],
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
