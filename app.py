# -*- coding: utf-8 -*-
import os
import io
import sqlite3
import hashlib
from datetime import date, datetime

from flask import (
    Flask, request, jsonify, send_from_directory,
    session, render_template
)
from openai import OpenAI

from docx import Document as DocxDocument
from PyPDF2 import PdfReader
import openpyxl
import re

# ------------ Config ------------

DATABASE_PATH = "nextgen_ai_teachers_aigrader.db"
FREE_DAILY_LIMIT = 5  # free plan limit

client = OpenAI()  # uses OPENAI_API_KEY from environment

app = Flask(__name__, static_folder="static", static_url_path="")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")  # replace in production

# ------------ Default Prompt ------------

DEFAULT_PROMPT_TEMPLATE = """You are an AI assistant that grades student work against a rubric for a human teacher.

You will always be given:
RUBRIC:
{rubric}

STUDENT SUBMISSION:
{student}

Your job is to carefully read the ENTIRE student submission and grade it ONLY using the rubric. Always look for:
- What the work actually does or says (semantic understanding).
- Whether specific requirements, constraints, or features in the rubric are present (rule checking).

GENERAL GRADING PRINCIPLES:
- Obey the rubric exactly. If it defines point values, levels (e.g., Exemplary / Proficient / Developing), or specific requirements, follow them.
- If the rubric is vague, make a reasonable assumption and explain it briefly in the justification.
- Never award full credit when clear rubric requirements are missing.
- Partial credit is allowed when some, but not all, expectations for a criterion are met.
- If something is completely missing, award 0 points for that criterion.

TREAT DIFFERENT TYPES OF WORK CAREFULLY:
- For CODE (C++, Java, Python, etc.), consider:
  - Does it compile or look syntactically valid?
  - Does it logically implement the required behavior?
  - Are required methods, classes, variables, and control structures present?
  - Are naming and style conventions followed if the rubric mentions them?
- For ESSAYS or WRITING, consider:
  - Does it answer the prompt?
  - Does it follow structure and length requirements?
  - Does it include required evidence, explanation, or formatting specified in the rubric?

OUTPUT FORMAT (PLAIN TEXT ONLY):
Return plain text with NO Markdown and NO bullets.

For each rubric criterion or row, output in this exact pattern:

Criterion: <short name or description of the rubric criterion>
Score: <earned_points>/<max_points>
Evidence: <1–3 clear sentences citing specific features of the student work, or explaining what is missing>

Leave exactly one blank line between criteria.

AFTER ALL CRITERIA, add a final section:

Total Score: <sum_earned_points>/<sum_max_points>

Teacher Comment Summary:
<2–4 complete sentences that a teacher can paste into Canvas or Google Classroom. Mention strengths first, then specific areas to improve, and, if helpful, a next step or suggestion for the student.>

FORMATTING RULES (VERY IMPORTANT):
- Do NOT use ** or any other Markdown.
- Do NOT use bullets like *, -, +.
- Do NOT use numbered lists like 1), 2), etc.
- Do NOT include code fences or backticks.
- Use only normal sentences and line breaks.

If the rubric does not clearly state total points, make a reasonable interpretation, explain it briefly in the first Evidence line of the first criterion, and still follow the same output structure."""

# ------------ Simple SHA256 Password Hashing ------------

def hash_password(password: str) -> str:
    """
    Hash a password using SHA256.
    NOTE: This is a simple hash, not salted. Fine for a small demo / local app.
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Compare a plain-text password to a SHA256 hash.
    """
    return hash_password(password) == stored_hash

# ------------ DB Helpers ------------

def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            plan TEXT NOT NULL DEFAULT 'free', -- 'free' or 'pro'
            uses_today INTEGER NOT NULL DEFAULT 0,
            last_use_date TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS activation_codes (
            code TEXT PRIMARY KEY,
            plan TEXT NOT NULL,             -- e.g. 'pro'
            redeemed_by INTEGER,            -- user id
            redeemed_at TEXT,               -- ISO timestamp
            FOREIGN KEY(redeemed_by) REFERENCES users(id)
        );
    """)

    conn.commit()
    conn.close()


# Call at startup
init_db()

# ------------ File Text Extraction ------------

def extract_text_from_file(file_storage):
    filename = file_storage.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    raw = file_storage.read()
    file_storage.seek(0)

    # Plain text / code / CSV
    if ext in [".txt", ".cpp", ".java", ".py", ".md", ".xml", ".html", ".json", ".csv"]:
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return "[Error decoding text file.]"

    # DOCX
    if ext == ".docx":
        try:
            buffer = io.BytesIO(raw)
            doc = DocxDocument(buffer)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return "[Error reading DOCX file.]"

    # PDF
    if ext == ".pdf":
        try:
            buffer = io.BytesIO(raw)
            reader = PdfReader(buffer)
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text)
            return "\n".join(pages)
        except Exception:
            return "[Error reading PDF file.]"

    # XLSX
    if ext == ".xlsx":
        try:
            buffer = io.BytesIO(raw)
            wb = openpyxl.load_workbook(buffer, data_only=True)
            sheet = wb.active
            lines = []
            for row in sheet.iter_rows(values_only=True):
                line = "\t".join("" if cell is None else str(cell) for cell in row)
                lines.append(line)
            return "\n".join(lines)
        except Exception:
            return "[Error reading XLSX file.]"

    return f"[Unsupported or unknown file type: {ext}]"

# ------------ AI Call ------------

# Primary and fallback models for grading
PRIMARY_MODEL = os.environ.get("OPENAI_PRIMARY_MODEL", "gpt-4o")
FALLBACK_MODEL = os.environ.get("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")

def build_full_prompt(prompt_template: str, rubric_text: str, student_text: str) -> str:
    """
    Safely interpolate rubric and student text into the prompt template.

    Escapes curly braces in the rubric and student text so that str.format
    does not treat them as formatting placeholders.
    """
    def _escape(text: str) -> str:
        return (text or "").replace("{", "{{").replace("}", "}}")

    return prompt_template.format(
        rubric=_escape(rubric_text),
        student=_escape(student_text),
    )

def call_model(full_prompt: str) -> str:
    """
    Call OpenAI and return plain text.
    Uses chat.completions and falls back to a secondary model on error.
    """
    last_error = None
    for model_name in (PRIMARY_MODEL, FALLBACK_MODEL):
        if not model_name:
            continue
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = str(e)
            continue
    raise RuntimeError(f"OpenAI call failed: {last_error}")

# ------------ Auth Helpers ------------

def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row


def update_user_usage(user_id):
    today_str = date.today().isoformat()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT uses_today, last_use_date, plan FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return {"error": "User not found"}

    uses_today = row["uses_today"] or 0
    last_use_date = row["last_use_date"]
    plan = row["plan"]

    if last_use_date != today_str:
        uses_today = 0

    if plan == "free" and uses_today >= FREE_DAILY_LIMIT:
        conn.close()
        return {"error": "Free plan daily limit reached. Upgrade to Pro for unlimited grading."}

    uses_today += 1
    cur.execute("""
        UPDATE users SET uses_today = ?, last_use_date = ? WHERE id = ?
    """, (uses_today, today_str, user_id))
    conn.commit()
    conn.close()

    return {
        "plan": plan,
        "uses_today": uses_today,
        "limit": FREE_DAILY_LIMIT
    }

# ------------ Routes ------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/grader")
def grader_page():
    return render_template("grader.html")

# --- Auth APIs ---

@app.post("/api/register")
def register():
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    pw_hash = hash_password(password)

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO users (email, password_hash, plan)
            VALUES (?, ?, 'free')
        """, (email, pw_hash))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Email already registered."}), 400

    cur.execute("SELECT id FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()

    session["user_id"] = row["id"]
    return jsonify({"message": "Registered successfully.", "plan": "free"})


@app.post("/api/login")
def login():
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()

    if not row or not verify_password(password, row["password_hash"]):
        return jsonify({"error": "Invalid email or password."}), 400

    session["user_id"] = row["id"]
    return jsonify({
        "message": "Logged in successfully.",
        "plan": row["plan"]
    })


@app.post("/api/logout")
def logout():
    session.pop("user_id", None)
    return jsonify({"message": "Logged out."})


@app.get("/api/me")
def me():
    user = get_current_user()
    if not user:
        return jsonify({"loggedIn": False})
    return jsonify({
        "loggedIn": True,
        "email": user["email"],
        "plan": user["plan"],
        "uses_today": user["uses_today"],
        "limit": FREE_DAILY_LIMIT
    })

# --- Activation Code Redeem ---

@app.post("/api/redeem")
def redeem():
    user = get_current_user()
    if not user:
        return jsonify({"error": "Login required."}), 401

    data = request.get_json() or {}
    code = (data.get("code") or "").strip()

    if not code:
        return jsonify({"error": "Activation code is required."}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM activation_codes WHERE code = ?", (code,))
    code_row = cur.fetchone()

    if not code_row:
        conn.close()
        return jsonify({"error": "Invalid activation code."}), 400

    if code_row["redeemed_by"] is not None:
        conn.close()
        return jsonify({"error": "Code has already been redeemed."}), 400

    # Upgrade user plan
    cur.execute("UPDATE users SET plan = ? WHERE id = ?", (code_row["plan"], user["id"]))
    cur.execute("""
        UPDATE activation_codes
        SET redeemed_by = ?, redeemed_at = ?
        WHERE code = ?
    """, (user["id"], datetime.utcnow().isoformat(), code))
    conn.commit()
    conn.close()

    return jsonify({"message": f"Code redeemed! Your plan is now {code_row['plan']}."})

# --- Grading API ---

@app.post("/api/grade")
def grade():
    """
    Grade a student submission against a rubric using the AI model.

    - Requires the user to be logged in.
    - Enforces free vs. pro usage limits via update_user_usage.
    - Accepts rubric and student content as text, file upload, or both.
    - Uses a robust prompt that supports any rubric and any file type.
    """
    user = get_current_user()
    if not user:
        return jsonify({"error": "Login required."}), 401

    # Check and update usage (free vs pro)
    usage_info = update_user_usage(user["id"])
    if "error" in usage_info:
        # Daily limit or user issue
        return jsonify(usage_info), 403

    # Text fields
    rubric_text = (request.form.get("rubricText") or "").strip()
    student_text = (request.form.get("studentText") or "").strip()
    prompt_template = (request.form.get("promptTemplate") or "").strip() or DEFAULT_PROMPT_TEMPLATE

    # File uploads
    rubric_file = request.files.get("rubricFile")
    student_file = request.files.get("studentFile")

    rubric_file_text = ""
    student_file_text = ""

    if rubric_file and getattr(rubric_file, "filename", ""):
        rubric_file_text = (extract_text_from_file(rubric_file) or "").strip()

    if student_file and getattr(student_file, "filename", ""):
        student_file_text = (extract_text_from_file(student_file) or "").strip()

    # Combine pasted text + file text
    combined_rubric = "\n\n".join(
        part for part in [rubric_text, rubric_file_text] if part
    ).strip()

    combined_student = "\n\n".join(
        part for part in [student_text, student_file_text] if part
    ).strip()

    # Basic validation
    if not combined_rubric:
        return jsonify({"error": "Rubric is required. Paste rubric text or upload a rubric file."}), 400

    if not combined_student:
        return jsonify({"error": "Student work is required. Paste student text or upload a student file."}), 400

    # Build full prompt safely (handles curly braces in rubric or student text)
    full_prompt = build_full_prompt(prompt_template, combined_rubric, combined_student)

    try:
        raw_result = call_model(full_prompt) or ""
    except Exception as e:
        return jsonify({"error": f"Error while calling AI model: {e}"}), 500

    # Defensive cleaning in case the model still returns some Markdown-style artifacts
    clean_result = raw_result.replace("**", "").replace("---", "")

    import re as _re_local
    clean_result = _re_local.sub(r"^[*\-\+]+\s*", "", clean_result, flags=_re_local.MULTILINE)
    clean_result = clean_result.strip()

    return jsonify({
        "result": clean_result,
        "plan": usage_info["plan"],
        "uses_today": usage_info["uses_today"],
        "limit": usage_info["limit"],
    })



if __name__ == "__main__":
    app.run(debug=True)
