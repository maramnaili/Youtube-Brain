# YouTube Brain

YouTube Brain is a multilingual video question-answering prototype. It indexes a video's transcript, retrieves relevant passages, and answers questions with timestamp citations you can use to jump back to the video.

## Features

- Process YouTube videos using captions, with local speech-to-text as a fallback.
- Upload MP3, MP4, WAV, or M4A files for local transcription.
- Search transcripts using multilingual embeddings and BM25, combined with reciprocal rank fusion.
- Generate answers, chapters, and transcript-supported key claims with Ollama or an OpenAI-compatible chat API.
- Cache transcripts and embeddings locally to avoid repeating work.

## Requirements

- Python 3.10 or newer (Python 3.11 recommended).
- [Ollama](https://ollama.com/) and a chat model for local answer generation. The default model is `qwen2.5:3b`.
- `ffmpeg` for local transcription when YouTube captions are unavailable and for uploaded media.
- Network access on first use to download the embedding model and, when needed, the Whisper model.

## Install and run on Windows

Run these commands from the project directory in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
ollama pull qwen2.5:3b
```

If PowerShell prevents environment activation, either allow local scripts for your user or invoke the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Start Ollama in one terminal:

```powershell
ollama serve
```

Start the app in another terminal:

```powershell
streamlit run app.py
```

Open the local address printed by Streamlit (normally <http://localhost:8501>). Paste a YouTube URL, choose an example, or upload a supported media file. Ask a question after processing; answer citations and chapters link to their corresponding video times.

On first use, the configured multilingual embedding model is downloaded. If captions are unavailable, YouTube audio is downloaded and transcribed with faster-whisper; uploaded media is also transcribed locally. These models may require significant disk space, memory, and download time. All generated cache files are stored under `data/cache/`.

## Configuration

Edit `config.yaml` to change the embedding and Whisper models, chunking, retrieval, example videos, upload limit, or language-model provider.

The default language-model configuration uses Ollama at `http://localhost:11434` with `qwen2.5:3b`. To use another Ollama model, pull it and update `llm.model`. To use an OpenAI-compatible endpoint, set `llm.provider` to `openai-compatible`, configure `llm.base_url` and `llm.model`, and provide the API key as the environment variable named by `llm.api_key_env`. Do not put API keys in `config.yaml`.

LLM requests use the configured `llm.timeout_seconds` value. Set it to a positive number of seconds to limit request duration, or to `null` to wait indefinitely.

On Windows, an embedding-model error with code 1455 indicates insufficient virtual memory. Increase the Windows paging-file size, ensure the drive has free space, and restart before retrying.

## Tests

Install the project dependencies as shown above, then run:

```powershell
python -m unittest discover -s tests -v
```

The tests cover timestamp utilities, upload validation, transcript chunking, retrieval fusion, and language-model request and response handling. They do not download video or model data and do not require a running Ollama server.

## Retrieval evaluation

Add labeled examples to `data/evaluation/questions.json`. Each answerable example needs a YouTube URL, question, and reference time interval; unanswerable examples need `answerable: false`. For example:

```json
{
  "questions": [
    {
      "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
      "question": "What does the speaker say about the main idea?",
      "answerable": true,
      "reference_start": 125,
      "reference_end": 160
    },
    {
      "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
      "question": "What color is the speaker's car?",
      "answerable": false
    }
  ]
}
```

Run the evaluation from the project directory:

```powershell
python scripts/evaluate.py
```

It compares dense, BM25, and hybrid retrieval, then writes `reports/evaluation.csv` and `reports/evaluation.md`. This requires network access to process uncached videos and a working configured language model. Recall@K and MRR use overlap with the labeled time interval; timestamp accuracy allows a 30-second margin. Review answer faithfulness and relevance manually.

## Docker

Build and run the Streamlit app:

```powershell
docker build -t youtube-brain .
docker run --rm -p 8501:8501 -v "${PWD}/data:/app/data" youtube-brain
```

The image includes `ffmpeg` and the Python application, but does not include Ollama or model weights. Ollama must run separately and be reachable from the container; `localhost` inside a container refers to the container itself. Configure the Ollama base URL in `config.yaml` to an address reachable from the container (for example, `http://host.docker.internal:11434` where supported).

## Repository contents

- `app.py` — Streamlit interface.
- `src/youtube_brain/` — ingestion, chunking, retrieval, storage, generation, and pipeline modules.
- `tests/` — unit tests.
- `scripts/evaluate.py` — labeled retrieval evaluation.
- `config.yaml` — application configuration.
- `requirements.txt` and `pyproject.toml` — Python dependencies and package metadata.
