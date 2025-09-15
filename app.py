# app.py — AI-Powered Study Assistant (Streamlit + LangChain)
# ------------------------------------------------------------
# Features
# - Upload a PDF ➜ extract text (PyPDF2)
# - Summarize into concise bullet points (LangChain + OpenAI)
# - Generate MCQs with explanations (schema-validated via Pydantic)
# - Adjustable params (bullets, difficulty, chunk sizes, etc.)
# - Download results (JSON)
# - Caching for speed
# ------------------------------------------------------------

import os
import io
import re
import gc
import json
import hashlib
from typing import List

import streamlit as st
from pydantic import BaseModel, Field, conlist, model_validator

# Optional imports (installed via requirements)
from PyPDF2 import PdfReader

# LangChain / OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

# -----------------------------
# Config & Constants
# -----------------------------
APP_TITLE = "AI Study Assistant – PDFs ➜ Notes + MCQs"
DEFAULT_MODEL = "gpt-4o-mini"

# Defaults (can be overridden via UI)
N_BULLETS_DEFAULT = 6
N_QUESTIONS_DEFAULT = 4
MAX_POINT_WORDS_DEFAULT = 12
CHUNK_MAX_CHARS_DEFAULT = 800
CHUNK_OVERLAP_DEFAULT = 150
SAFE_SINGLE_MAX = 1500

# Cache for LangChain
set_llm_cache(SQLiteCache("lc_cache.sqlite"))

# -----------------------------
# Data Models (Pydantic)
# -----------------------------
class MCQ(BaseModel):
    question: str
    options: conlist(str, min_length=3, max_length=6)
    correct_index: int = Field(..., ge=0)
    explanation: str

    @model_validator(mode="after")
    def _idx_in_range(self):
        if not (0 <= self.correct_index < len(self.options)):
            raise ValueError(
                f"correct_index must be between 0 and {len(self.options) - 1}"
            )
        return self

class MCQSet(BaseModel):
    topic_summary: List[str]
    questions: List[MCQ]

class Summary(BaseModel):
    bullets: List[str]

# -----------------------------
# Utility Helpers
# -----------------------------

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def to_point(t: str, max_words: int) -> str:
    t = re.sub(r"^\s*(?:[-–—•*]|\d+[\.)])\s*", "", t.strip())
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[\.;:,\-–—]\s*$", "", t)
    words = t.split()
    return " ".join(words[:max_words]) if len(words) > max_words else t


def safe_slice(text: str, max_chars: int = SAFE_SINGLE_MAX) -> str:
    return text[:max_chars] if len(text) > max_chars else text


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    chunks, i, n = [], 0, len(text)
    step = max(1, max_chars - min(overlap, max_chars - 1))
    while i < n:
        j = min(n, i + max_chars)
        chunks.append(text[i:j])
        i += step
    return chunks

# -----------------------------
# PDF Extraction (with cache)
# -----------------------------
@st.cache_data(show_spinner=False)
def extract_pdf_text_cached(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:
            pass
        if getattr(reader, "is_encrypted", False):
            raise ValueError("PDF is encrypted. Decrypt or supply password.")

    pages = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = t.replace("\x00", "").replace("\r", "\n")
        # fix hyphenated line breaks like "in-
        # formation" ➜ "information"
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
        # compact excessive tabs before linebreaks
        t = re.sub(r"[\t]+\n", "\n\n", t).strip()
        pages.append(f"[Page {i+1}]\n{t}")
    return "\n\n".join(pages).strip()

# -----------------------------
# LLM + Chains
# -----------------------------

def make_llm(api_key: str, model: str, temperature: float = 0.2, max_tokens: int = 512):
    return ChatOpenAI(
        model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key
    )


def build_summary_chain(api_key: str, model: str):
    summary_parser = PydanticOutputParser(pydantic_object=Summary)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a concise study assistant. Return exactly the schema requested. "
                "Summarize the user's study material into {n_bullets} concise points. "
                "{format_instructions}",
            ),
            ("human", "Study material:\n\n{raw_text}"),
        ]
    )
    chain = prompt | make_llm(api_key, model, temperature=0.1, max_tokens=256) | summary_parser
    return chain


def build_mcq_chain(api_key: str, model: str):
    # Strict structured output directly to MCQSet
    mcq_llm_strict = make_llm(api_key, model, temperature=0.0, max_tokens=900).with_structured_output(MCQSet)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Create {n_questions} multiple-choice questions based ONLY on the bullets. "
                "Requirements per question:\n"
                "- fields: question, options (exactly 4 strings), correct_index (0-3), explanation (≤ 25 words)\n"
                "- exactly one correct option\n"
                "- no new facts beyond the bullets\n"
                "Return data that matches the MCQSet schema.",
            ),
            ("human", "Bullets:\n{bullets}\n\nDifficulty: {difficulty}"),
        ]
    )
    chain = prompt | mcq_llm_strict
    return chain


# Summarization logic across chunks

def summarize_text(api_key: str, model: str, raw_text: str, n_bullets: int, max_words: int,
                    chunk_max_chars: int, chunk_overlap: int) -> List[str]:
    summary_chain = build_summary_chain(api_key, model)

    if len(raw_text) <= chunk_max_chars:
        out = summary_chain.invoke(
            {
                "raw_text": safe_slice(raw_text),
                "n_bullets": n_bullets,
                "format_instructions": PydanticOutputParser(pydantic_object=Summary).get_format_instructions(),
            }
        )
        return [to_point(b, max_words) for b in out.bullets]

    seen, window = set(), []
    for ch in chunk_text(raw_text, max_chars=chunk_max_chars, overlap=chunk_overlap):
        part = summary_chain.invoke(
            {
                "raw_text": ch,
                "n_bullets": n_bullets,
                "format_instructions": PydanticOutputParser(pydantic_object=Summary).get_format_instructions(),
            }
        )
        for b in part.bullets:
            bp = to_point(b, max_words)
            k = bp.lower()
            if k not in seen:
                seen.add(k)
                window.append(bp)
                if len(window) > n_bullets * 3:
                    window.pop(0)

    if len(window) <= n_bullets:
        return window

    compress_input = "\n".join("- " + x for x in window)
    compressed = summary_chain.invoke(
        {
            "raw_text": compress_input,
            "n_bullets": n_bullets,
            "format_instructions": PydanticOutputParser(pydantic_object=Summary).get_format_instructions(),
        }
    )
    return [to_point(b, max_words) for b in compressed.bullets]


def generate_mcqs(api_key: str, model: str, bullets: List[str], n_questions: int, difficulty: str) -> MCQSet:
    mcq_chain = build_mcq_chain(api_key, model)
    return mcq_chain.invoke(
        {
            "bullets": "\n- " + "\n- ".join(bullets),
            "n_questions": n_questions,
            "difficulty": difficulty,
        }
    )

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
st.title("🧠 AI Study Assistant")
st.caption("Turn PDFs into concise notes and auto-generated quizzes. LangChain + OpenAI")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    # API key handling (prefer env var, allow manual override)
    default_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=default_key,
        help="Used only locally to call OpenAI. Not uploaded.",
    )
    model = st.text_input("Model", value=DEFAULT_MODEL)

    st.markdown("---")
    n_bullets = st.number_input("Number of summary bullets", 3, 12, N_BULLETS_DEFAULT)
    n_questions = st.number_input("Number of MCQs", 2, 10, N_QUESTIONS_DEFAULT)
    difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard", "mixed"], index=3)

    st.markdown("---")
    max_point_words = st.slider("Max words per bullet", 6, 25, MAX_POINT_WORDS_DEFAULT)
    chunk_max_chars = st.slider("Chunk size (chars)", 400, 2000, CHUNK_MAX_CHARS_DEFAULT, step=50)
    chunk_overlap = st.slider("Chunk overlap (chars)", 50, 800, CHUNK_OVERLAP_DEFAULT, step=10)

    st.markdown("---")
    st.info(
        "Tip: You can also set OPENAI_API_KEY as an environment variable before running the app.",
        icon="💡",
    )

# File uploader
uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    file_hash = sha1_bytes(file_bytes)
    st.write(f"**File:** {uploaded.name}  |  **Size:** {len(file_bytes)/1024:.1f} KB  |  **SHA1:** `{file_hash[:10]}...`")

    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar to proceed.")
        st.stop()

    with st.spinner("Extracting text from PDF..."):
        try:
            raw_text = extract_pdf_text_cached(file_bytes)
        except Exception as e:
            st.error(f"PDF extraction failed: {e}")
            st.stop()

    st.success(f"Characters loaded: {len(raw_text):,}")

    # Actions
    col1, col2 = st.columns(2)
    do_summarize = col1.button("📝 Summarize")
    do_mcq = col2.button("❓ Generate MCQs")

    # Keep state across interactions
    if "bullets_state" not in st.session_state:
        st.session_state["bullets_state"] = []
    if "mcq_state" not in st.session_state:
        st.session_state["mcq_state"] = None

    if do_summarize:
        with st.spinner("Summarizing..."):
            bullets = summarize_text(
                api_key=api_key,
                model=model,
                raw_text=raw_text,
                n_bullets=int(n_bullets),
                max_words=int(max_point_words),
                chunk_max_chars=int(chunk_max_chars),
                chunk_overlap=int(chunk_overlap),
            )
            st.session_state["bullets_state"] = bullets
        st.success("Summary ready ✅")

    bullets = st.session_state.get("bullets_state", [])

    if bullets:
        st.subheader("Summary (Concise Points)")
        for i, b in enumerate(bullets, 1):
            st.markdown(f"**{i}.** {b}")

        # Download bullets JSON
        bullets_json = json.dumps({"bullets": bullets}, indent=2)
        st.download_button(
            label="⬇️ Download Summary (JSON)",
            data=bullets_json,
            file_name="summary.json",
            mime="application/json",
        )

    if do_mcq and not bullets:
        st.warning("Please generate the summary first, then MCQs.")

    if do_mcq and bullets:
        with st.spinner("Generating MCQs..."):
            try:
                mcq_set = generate_mcqs(
                    api_key=api_key,
                    model=model,
                    bullets=bullets,
                    n_questions=int(n_questions),
                    difficulty=difficulty,
                )
                st.session_state["mcq_state"] = mcq_set
            except Exception as e:
                st.error(f"MCQ generation failed: {e}")
                st.stop()
        st.success("MCQs ready ✅")

    mcq_set = st.session_state.get("mcq_state")

    if mcq_set:
        st.subheader("Topic Summary (from MCQSet)")
        for i, b in enumerate(mcq_set.topic_summary, 1):
            st.markdown(f"**{i}.** {b}")

        st.subheader("MCQs")
        for i, q in enumerate(mcq_set.questions, 1):
            with st.expander(f"Q{i}. {q.question}"):
                for j, opt in enumerate(q.options):
                    label = chr(65 + j) + ". " + opt
                    if j == q.correct_index:
                        st.markdown(f"✅ **{label}**")
                    else:
                        st.markdown(label)
                st.markdown(f"**Answer:** {chr(65 + q.correct_index)}")
                st.markdown(f"**Why:** {q.explanation}")

        # Download MCQs JSON
        mcq_json = mcq_set.model_dump_json(indent=2)
        st.download_button(
            label="⬇️ Download MCQs (JSON)",
            data=mcq_json,
            file_name="mcqs.json",
            mime="application/json",
        )

    # Memory cleanup button
    with st.expander("Advanced: Memory / Debug"):
        if st.button("Free memory (gc.collect)"):
            del raw_text
            gc.collect()
            st.info("Garbage collected.")

else:
    st.info("Upload a PDF to get started.")

# -----------------------------
# How to run (shown in app footer)
# -----------------------------
st.markdown(
    "---\n**Run locally:**\n\n"
    "1. Create a virtual env (optional)\n\n"
    "2. Install deps:\n"
    "```bash\n"
    "pip install streamlit langchain==0.3.21 langchain-openai==0.3.9 langchain-community==0.3.19 PyPDF2 tqdm pydantic\n"
    "```\n"
    "3. Set your API key (macOS/Linux):\n"
    "```bash\nexport OPENAI_API_KEY=YOUR_KEY\n``\n"
    "4. Run the app:\n"
    "```bash\nstreamlit run app.py\n```\n"
)
