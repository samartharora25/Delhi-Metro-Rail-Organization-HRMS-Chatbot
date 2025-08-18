import os
import sys
import traceback
from typing import List, Dict, Any

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from supabase import create_client, Client
from dotenv import load_dotenv
load_dotenv() 
# Ensure DMRC_Chatbot package is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DMRC_DIR = os.path.join(BASE_DIR, "DMRC_Chatbot")
if DMRC_DIR not in sys.path:
    sys.path.append(DMRC_DIR)

# Now import the RAG system
try:
    from src.langchain_rag import HRDocumentRAG  # type: ignore
except Exception as e:
    HRDocumentRAG = None
    _import_error = f"Failed to import HRDocumentRAG: {e}"
else:
    _import_error = None

# Load environment variables
load_dotenv()
# Also try loading DMRC_Chatbot/.env if present
dmrc_env = os.path.join(DMRC_DIR, ".env")
if os.path.exists(dmrc_env):
    load_dotenv(dmrc_env, override=False)

app = Flask(__name__, template_folder="templates")
# Secret key for session cookies (set in .env)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

# Flask-Login setup
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

# Supabase client (from env)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
# Dev-only bypass login configuration
ALLOW_DEV_LOGIN = (os.getenv("ALLOW_DEV_LOGIN", "false").lower() == "true")
DEFAULT_LOGIN_EMAIL = os.getenv("DEFAULT_LOGIN_EMAIL", "")
DEFAULT_LOGIN_PASSWORD = os.getenv("DEFAULT_LOGIN_PASSWORD", "")
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_ANON_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
else:
    print("⚠️ SUPABASE_URL or SUPABASE_ANON_KEY not set. Auth will not work.")


class User(UserMixin):
    def __init__(self, user_id: str, email: str | None = None):
        self.id = user_id
        self.email = email


@login_manager.user_loader
def load_user(user_id: str):
    # Minimal loader: we only store the user id in session
    return User(user_id)


# Global state
rag = None  # type: HRDocumentRAG | None
initialization_error: str | None = None


# ---------- New UI routes ----------
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/login", methods=["GET", "POST"]) 
def login():
    if request.method == "POST":
        if supabase is None:
            flash("Auth not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY.", "error")
            return render_template("login.html")

        # Read from form or JSON
        email = (request.form.get("email") or (request.json.get("email") if request.is_json else "")) or ""
        password = (request.form.get("password") or (request.json.get("password") if request.is_json else "")) or ""
        email = email.strip()

        if not email or not password:
            flash("Please enter both email and password.", "error")
            return render_template("login.html"), 400

        # Dev-only default login bypass (guarded by env flag)
        if ALLOW_DEV_LOGIN and DEFAULT_LOGIN_EMAIL and DEFAULT_LOGIN_PASSWORD:
            if email == DEFAULT_LOGIN_EMAIL and password == DEFAULT_LOGIN_PASSWORD:
                print(f"[Auth] DEV bypass login for {email}")
                login_user(User(user_id=f"dev-{email}", email=email))
                return redirect(url_for("chat"))

        try:
            auth_res = supabase.auth.sign_in_with_password({"email": email, "password": password})
            user = getattr(auth_res, "user", None)
            if not user:
                detail = getattr(auth_res, "error", None) or getattr(auth_res, "message", None)
                raise ValueError(detail or "Invalid email or password.")
            login_user(User(user_id=user.id, email=email))
            return redirect(url_for("chat"))
        except Exception as e:
            # Log exact reason to the server console for diagnosis
            print(f"[Auth] Login failed for {email}: {e}")
            lower = str(e).lower()
            if "not confirmed" in lower or "email not confirmed" in lower:
                flash("Email not confirmed. Confirm the user in Supabase (Auth > Users) or disable confirmations for dev.", "error")
            elif "invalid login credentials" in lower or "invalid email or password" in lower:
                flash("Invalid email or password.", "error")
            else:
                flash(f"Login failed: {str(e)}", "error")
            return render_template("login.html"), 401

    return render_template("login.html")


@app.route("/chat")
@login_required
def chat():
    return render_template(
        "index.html",
        system_initialized=rag is not None and initialization_error is None,
        initialization_error=initialization_error,
    )


# ---------- Existing status/index kept for compatibility ----------
@app.route("/index")
def index():
    return render_template(
        "index.html",
        system_initialized=rag is not None and initialization_error is None,
        initialization_error=initialization_error,
    )


def _format_sources(docs: List[Any]) -> List[Dict[str, Any]]:
    sources = []
    for d in docs or []:
        meta = getattr(d, "metadata", {}) or {}
        sources.append({
            "chapter": meta.get("chapter_title") or meta.get("type") or "Unknown",
            "section": meta.get("subtopic_title"),
            "pages": meta.get("page_range", "-"),
            "content_preview": (getattr(d, "page_content", "") or "")[:300]
        })
    return sources


def init_rag() -> None:
    global rag, initialization_error

    if _import_error:
        initialization_error = _import_error
        return

    try:
        # Prefer existing chunks.json if present, else fall back to sample PDF
        chunks_path = os.path.join(DMRC_DIR, "chunks.json")
        pdf_path = os.path.join(DMRC_DIR, "data", "Sample HR Policy Manual.pdf")
        chroma_db_path = os.path.join(DMRC_DIR, "chroma_db")

        kwargs: Dict[str, Any] = {
            "chroma_db_path": chroma_db_path,
            "collection_name": "hr_documents",
        }
        if os.path.exists(chunks_path):
            kwargs["chunks_file"] = chunks_path
        elif os.path.exists(pdf_path):
            kwargs["pdf_path"] = pdf_path
        else:
            initialization_error = (
                "No chunks.json or sample PDF found. Please place 'chunks.json' in DMRC_Chatbot/ or a PDF at "
                "DMRC_Chatbot/data/Sample HR Policy Manual.pdf"
            )
            return

        rag_instance = HRDocumentRAG(**kwargs)  # type: ignore

        # Build/load vector store and retrieval chain; don't force rebuild by default
        rag_instance.build_rag_system(
            force_rebuild=False,
            groq_model="llama3-8b-8192",
            temperature=0.1,
            max_tokens=1024,
        )

        rag = rag_instance
        initialization_error = None
    except Exception as e:
        initialization_error = f"RAG initialization failed: {e}\n{traceback.format_exc()}"
        rag = None


@app.route("/api/status", methods=["GET"]) 
def api_status():
    status = {
        "system_initialized": rag is not None and initialization_error is None,
        "initialization_error": initialization_error,
    }
    if rag is not None:
        try:
            status["stats"] = rag.get_stats()
        except Exception as e:
            status["stats_error"] = str(e)
    return jsonify(status)


@app.route("/api/chat", methods=["POST"]) 
def api_chat():
    if rag is None or initialization_error is not None:
        return jsonify({
            "error": True,
            "message": initialization_error or "RAG system not initialized",
            "sources": [],
        }), 503

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": True, "message": "Empty message", "sources": []}), 400

    try:
        result = rag.query(message, return_source_docs=True)  # type: ignore

        if isinstance(result, dict):
            answer = result.get("answer") or result.get("result") or ""
            docs = result.get("source_documents") or []
            return jsonify({
                "error": False,
                "message": answer,
                "sources": _format_sources(docs),
            })
        else:
            # Fallback: similarity docs list
            docs = result or []
            return jsonify({
                "error": False,
                "message": "I found relevant passages. However, the LLM answer was unavailable. Here are the top matches.",
                "sources": _format_sources(docs),
            })

    except Exception as e:
        return jsonify({
            "error": True,
            "message": f"Processing error: {e}",
            "sources": [],
        }), 500


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


if __name__ == "__main__":
    # Initialize on startup
    init_rag()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
