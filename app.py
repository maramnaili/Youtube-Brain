"""Streamlit interface for the YouTube Brain video Q&A demo."""

import re

import streamlit as st

from youtube_brain.config import load_settings
from youtube_brain.generation import ABSTENTION
from youtube_brain.pipeline import VideoKnowledge, VideoPipeline
from youtube_brain.timestamps import format_timestamp

st.set_page_config(page_title="Video Brain", page_icon="🧠", layout="wide")


@st.cache_resource
def get_settings():
    return load_settings()


@st.cache_resource
def get_pipeline():
    return VideoPipeline(get_settings())


def link_timestamps(text: str, video_id: str) -> str:
    """Turn validated MM:SS citations into links that seek the embedded player."""
    def replace(match: re.Match[str]) -> str:
        timestamp = match.group(1)
        minutes, seconds = (int(value) for value in timestamp.split(":"))
        total_seconds = minutes * 60 + seconds
        return f"[{timestamp}](?v={video_id}&t={total_seconds})"

    return re.sub(r"\[(\d{2,}:\d{2})\]", replace, text)


def load_video(
    process, source_label: str
) -> None:
    """Run ingestion and prepare the per-video retriever for the current session."""
    try:
        with st.spinner(f"Processing {source_label}: captions, transcript, index and insights..."):
            knowledge, was_cached = process()
            retriever = get_pipeline().retriever(knowledge)
        st.session_state.knowledge = knowledge
        st.session_state.retriever = retriever
        st.session_state.video_source = source_label
        st.session_state.last_answer = None
        st.session_state.last_evidence = []
        st.success("Loaded from cache." if was_cached else "Video processed and indexed.")
    except (ValueError, RuntimeError) as error:
        st.error(str(error))


settings = get_settings()
pipeline = get_pipeline()

st.title("🧠 Video Brain")
st.caption("Ask a video a question. Get a grounded answer and jump straight to the moment.")

with st.form("youtube_form"):
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    youtube_submitted = st.form_submit_button("Process video", type="primary")
if youtube_submitted:
    load_video(lambda: pipeline.ingest_youtube(url), "YouTube video")

with st.expander("Or upload an audio/video file"):
    uploaded = st.file_uploader(
        "MP3, MP4, WAV, or M4A (up to 200 MB)",
        type=["mp3", "mp4", "wav", "m4a"],
    )
    if st.button("Transcribe uploaded file", disabled=uploaded is None):
        if uploaded is not None:
            contents = uploaded.getvalue()
            load_video(
                lambda: pipeline.ingest_upload(uploaded.name, contents),
                uploaded.name,
            )



knowledge: VideoKnowledge | None = st.session_state.get("knowledge")
if knowledge:
    st.divider()
    video_id = knowledge.video_id
    st.header(st.session_state.get("video_source", "Video"))
    seek_seconds = 0
    try:
        requested_video_id = st.query_params.get("v")
        if requested_video_id in (None, video_id):
            seek_seconds = max(0, int(st.query_params.get("t", "0")))
    except ValueError:
        st.query_params.pop("t", None)

    left, right = st.columns([2, 1], gap="large")
    with left:
        if not video_id.startswith("upload-"):
            st.video(f"https://www.youtube.com/watch?v={video_id}", start_time=seek_seconds)
        else:
            st.info("Uploaded audio/video processed. The transcript and timestamps are available below.")
        st.subheader("Ask this video")
        with st.form("question_form"):
            question = st.text_input("Your question")
            asked = st.form_submit_button("Ask")
        if asked:
            try:
                evidence = pipeline.retrieve(st.session_state.retriever, question)
                answer = pipeline.answer(question, evidence)
                st.session_state.last_answer = answer
                st.session_state.last_evidence = evidence
            except (RuntimeError, ValueError) as error:
                st.error(str(error))

        if st.session_state.get("last_answer"):
            answer = st.session_state.last_answer
            if answer == ABSTENTION:
                st.info(ABSTENTION)
            else:
                st.markdown(link_timestamps(answer, video_id))
            evidence = st.session_state.get("last_evidence", [])
            if evidence:
                with st.expander("Retrieved passages"):
                    for item in evidence:
                        timestamp = format_timestamp(item.chunk.start_time)
                        st.markdown(
                            f"**[{timestamp}](?v={video_id}&t={int(item.chunk.start_time)})** "
                            f"— {item.chunk.text}"
                        )

    with right:
        st.subheader("Chapters")
        for chapter in knowledge.chapters:
            timestamp = format_timestamp(chapter.start_time)
            st.markdown(
                f"- [{timestamp}](?v={video_id}&t={int(chapter.start_time)}) "
                f"{chapter.title}"
            )
        st.subheader("Key claims")
        if knowledge.claims:
            for claim in knowledge.claims:
                timestamp = format_timestamp(claim.timestamp)
                st.markdown(
                    f"- **[{timestamp}](?v={video_id}&t={int(claim.timestamp)})** "
                    f"{claim.claim}  \n  > {claim.quote}"
                )
        else:
            st.caption("No key claims could be verified against the transcript.")
