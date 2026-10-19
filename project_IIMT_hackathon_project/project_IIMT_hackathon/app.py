"""Adaptive Learning and Knowledge Support Streamlit app."""

import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from project_IIMT_hackathon.learning import (
    ModelServiceError,
    NotesFileError,
    answer_from_evidence,
    extract_pdf_text,
    generate_quiz,
    retrieve_evidence,
)


SAMPLE_NOTES = """Topic: Photosynthesis Overview
Photosynthesis is the process used by plants, algae, and certain bacteria to harness energy from sunlight into chemical energy.
Key Stages:
1. Light-dependent reactions occur in the thylakoid membrane, where sunlight is absorbed by chlorophyll to produce ATP and NADPH.
2. The Calvin cycle occurs in the stroma and uses ATP and NADPH to convert carbon dioxide into glucose.
Chloroplasts are the cellular organelles where photosynthesis takes place.
Mitochondria are responsible for cellular respiration, not photosynthesis."""
MAX_NOTES_CHARACTERS = 100_000


try:
    app_secrets = dict(st.secrets)
except StreamlitSecretNotFoundError:
    app_secrets = {}
for setting_name in ("OLLAMA_HOST", "OLLAMA_MODEL", "OLLAMA_API_KEY"):
    if setting_name in app_secrets and not os.environ.get(setting_name):
        os.environ[setting_name] = str(app_secrets[setting_name])


def reset_learning_progress() -> None:
    st.session_state.chat_history = []
    st.session_state.quiz = None
    st.session_state.quiz_submitted = False


def load_sample_notes() -> None:
    st.session_state.notes = SAMPLE_NOTES
    st.session_state.notes_editor = SAMPLE_NOTES
    st.session_state.topic_editor = "Photosynthesis"
    st.session_state.active_topic = "Photosynthesis"
    reset_learning_progress()


st.set_page_config(
    page_title="Adaptive Learning & Knowledge Support",
    page_icon="📘",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17324d;
        --muted: #53677a;
        --teal: #087e83;
        --blue: #176b88;
        --line: #dbe5eb;
        --paper: #ffffff;
    }
    .stApp {
        background: #f5f8fa;
        color: var(--ink);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
            Helvetica, Arial, sans-serif;
    }
    [data-testid="stSidebar"] {
        background: var(--paper);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: var(--ink);
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.025em;
    }
    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }
    .stTextArea textarea,
    .stTextInput input {
        background: #fff;
        color: var(--ink);
        border: 1px solid #cbd8e1;
        border-radius: 7px;
    }
    .stButton button,
    [data-testid="stFormSubmitButton"] button {
        min-height: 2.6rem;
        border: 0;
        border-radius: 7px;
        background: #173f66;
        color: #fff;
        font-weight: 650;
        transition: background-color 150ms ease;
    }
    .stButton button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background: #0c6576;
        color: #fff;
    }
    .stAlert {
        border-radius: 8px;
    }
    .workflow-step {
        min-height: 58px;
        padding: 11px 8px;
        border: 1px solid #dbe5eb;
        border-radius: 8px;
        background: #fff;
        color: #17324d;
        font-size: 0.82rem;
        font-weight: 650;
        text-align: center;
    }
    .section-kicker {
        margin-bottom: 0.25rem;
        color: #087e83;
        font-size: 0.75rem;
        font-weight: 750;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    @media (max-width: 700px) {
        .workflow-step {
            font-size: 0.72rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("notes", "")
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("quiz", None)
st.session_state.setdefault("quiz_submitted", False)
st.session_state.setdefault("active_topic", "")
st.session_state.setdefault("notes_editor", st.session_state.notes)
st.session_state.setdefault("topic_editor", st.session_state.active_topic)
st.session_state.setdefault("quiz_id", 0)

with st.sidebar:
    st.markdown('<p class="section-kicker">Study setup</p>', unsafe_allow_html=True)
    st.header("Your learning session")
    st.caption("Choose one topic and add the notes you want to study.")
    st.text_input("Selected topic", key="topic_editor", placeholder="e.g. Photosynthesis")

    uploaded_file = st.file_uploader(
        "Upload notes",
        type=["txt", "md", "pdf"],
        help="Upload plain text, Markdown, or a text-based PDF (up to 5 MB).",
        max_upload_size=5,
    )
    if uploaded_file is not None and st.button(
        "Load uploaded notes", use_container_width=True
    ):
        try:
            file_bytes = uploaded_file.getvalue()
            if uploaded_file.name.casefold().endswith(".pdf"):
                uploaded_notes = extract_pdf_text(file_bytes)
            else:
                uploaded_notes = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            st.error("This file is not valid UTF-8 text. Save it as UTF-8 and try again.")
        except NotesFileError as error:
            st.error(str(error))
        else:
            if len(uploaded_notes) > MAX_NOTES_CHARACTERS:
                st.error(
                    f"Extracted notes exceed {MAX_NOTES_CHARACTERS:,} characters. "
                    "Use a shorter file focused on one topic."
                )
            else:
                st.session_state.notes_editor = uploaded_notes
                st.success("Notes loaded into the editor. Commit them to start studying.")

    st.text_area(
        "Paste or edit course notes",
        key="notes_editor",
        height=250,
        placeholder="Paste notes for one topic here...",
    )
    st.button(
        "Load sample notes",
        on_click=load_sample_notes,
        use_container_width=True,
    )

    if st.button("Commit notes to session", type="primary", use_container_width=True):
        new_notes = st.session_state.notes_editor.strip()
        topic = st.session_state.topic_editor.strip()
        if not new_notes:
            st.error("Add or upload notes before committing the study session.")
        elif len(new_notes) > MAX_NOTES_CHARACTERS:
            st.error(
                f"Keep notes under {MAX_NOTES_CHARACTERS:,} characters for reliable "
                "local study sessions."
            )
        elif not topic:
            st.error("Enter one selected topic before committing the study session.")
        else:
            st.session_state.notes = new_notes
            st.session_state.active_topic = topic
            reset_learning_progress()
            st.success("Study session updated. Previous answers and quizzes were cleared.")

    st.divider()
    st.caption(
        "AI uses a configured Ollama service. "
        f"Configured model: `{os.environ.get('OLLAMA_MODEL', 'llama3.2:3b')}`."
    )
    st.caption(
        "Notes are sent only to the configured Ollama endpoint. For cloud hosting, "
        "configure a secure, reachable endpoint in app secrets."
    )

st.markdown('<p class="section-kicker">Study companion from real notes</p>', unsafe_allow_html=True)
st.title("Adaptive Learning and Knowledge Support")
st.markdown(
    "Turn course material into a focused tutoring session: ask questions, inspect "
    "the evidence, practice, and learn from feedback."
)

workflow = [
    "1 · Upload / paste notes",
    "2 · Ask a question",
    "3 · Inspect evidence",
    "4 · Practice quiz",
    "5 · Receive feedback",
]
workflow_columns = st.columns(len(workflow))
for column, step in zip(workflow_columns, workflow):
    with column:
        st.markdown(f'<div class="workflow-step">{step}</div>', unsafe_allow_html=True)

st.write("")
if not st.session_state.notes.strip():
    st.info(
        "Start by selecting one topic and adding its notes in the sidebar. "
        "The sample notes are available for a quick demonstration."
    )
    st.subheader("Demonstration checklist")
    st.markdown(
        "- Load the sample notes or paste your own material.\n"
        "- Ask a question that the notes answer and inspect its source lines.\n"
        "- Generate a practice question and submit an answer.\n"
        "- Ask something unsupported and see the system state its limitation."
    )
else:
    note_lines = len(st.session_state.notes.splitlines())
    detail_columns = st.columns([2, 1, 1])
    with detail_columns[0]:
        st.markdown(f"**Active topic:** {st.session_state.active_topic}")
    with detail_columns[1]:
        st.caption(f"{note_lines} source lines")
    with detail_columns[2]:
        st.caption("Local AI · Ollama")

    qa_tab, quiz_tab = st.tabs(["Ask & inspect evidence", "Practice & feedback"])

    with qa_tab:
        st.subheader("Ask about your notes")
        st.caption(
            "Answers are generated from retrieved note passages. If the notes do "
            "not support an answer, the tutor will say so."
        )
        with st.form("question_form"):
            question = st.text_input(
                "Your question",
                placeholder="Ask something about the selected topic...",
            )
            ask_submitted = st.form_submit_button("Ask question")

        if ask_submitted:
            cleaned_question = question.strip()
            if not cleaned_question:
                st.error("Enter a question to continue.")
            else:
                evidence = retrieve_evidence(st.session_state.notes, cleaned_question)
                result = {
                    "question": cleaned_question,
                    "answer": "",
                    "evidence": evidence,
                    "error": "",
                }
                if not evidence:
                    result["answer"] = (
                        "The notes do not contain enough information to answer "
                        "this question. Try a question about another part of the "
                        "selected topic."
                    )
                else:
                    with st.spinner("Checking the source and preparing an answer..."):
                        try:
                            result["answer"] = answer_from_evidence(
                                st.session_state.active_topic,
                                cleaned_question,
                                evidence,
                            )
                        except ModelServiceError as error:
                            result["error"] = str(error)
                st.session_state.chat_history.append(result)

        for chat in reversed(st.session_state.chat_history):
            with st.container(border=True):
                st.markdown(f"**Question:** {chat['question']}")
                if chat.get("error"):
                    st.error(f"Could not generate an answer: {chat['error']}")
                elif chat.get("answer"):
                    st.info(chat["answer"])
                if chat.get("evidence"):
                    with st.expander("Inspect supporting evidence"):
                        for passage in chat["evidence"]:
                            st.markdown(
                                f"**Source line {passage['line']}**  \n"
                                f"> {passage['text']}"
                            )

    with quiz_tab:
        st.subheader("Practice this topic")
        st.caption(
            "Generate a question from the active notes. The correct option and "
            "feedback are checked against a quote from your material."
        )
        if st.button("Generate practice quiz", type="primary", key="generate_quiz"):
            st.session_state.quiz = None
            st.session_state.quiz_submitted = False
            with st.spinner("Creating a source-grounded practice question..."):
                try:
                    st.session_state.quiz = generate_quiz(
                        st.session_state.active_topic,
                        st.session_state.notes,
                    )
                    st.session_state.quiz_id += 1
                except ModelServiceError as error:
                    st.error(f"Could not generate a practice quiz: {error}")

        quiz = st.session_state.quiz
        if quiz:
            st.markdown(f"**{quiz['question']}**")
            with st.form(f"quiz_response_{st.session_state.quiz_id}"):
                selected_option = st.radio(
                    "Choose the best answer",
                    quiz["options"],
                    key=f"quiz_choice_{st.session_state.quiz_id}",
                )
                submitted = st.form_submit_button("Submit answer")

            if submitted:
                st.session_state.quiz_submitted = True
                st.session_state.quiz_response = selected_option

            if st.session_state.quiz_submitted:
                if st.session_state.quiz_response == quiz["answer"]:
                    st.success("Correct. Your answer is supported by the notes.")
                else:
                    st.warning(
                        "Not quite. Revisit the source evidence and use this hint "
                        "to try the concept again."
                    )
                st.markdown(f"**Feedback:** {quiz['explanation']}")
                st.markdown(f"**Supporting note:** “{quiz['source_quote']}”")
        else:
            st.info("Generate a practice question to check your understanding.")
