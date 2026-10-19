# Streamlit Community Cloud deployment

## App settings

Deploy the `main` branch of `Omkar-Jha/test1` in Streamlit Community Cloud and
set the app file path to:

```text
project_IIMT_hackathon_project/project_IIMT_hackathon/app.py
```

The repository-root `requirements.txt` installs Streamlit and PDF text
extraction dependencies.

## Configure AI

The app uses Ollama's chat API. A cloud-hosted app cannot reach Ollama running
on your own computer through `localhost`. In the Streamlit app's **Settings →
Secrets**, configure an Ollama endpoint that is reachable from the internet
over HTTPS:

```toml
OLLAMA_HOST = "https://your-secure-ollama-endpoint"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_API_KEY = "your-endpoint-bearer-token"
```

`OLLAMA_API_KEY` is optional when the endpoint is protected by another
appropriate mechanism. Do not expose a local Ollama server directly to the
public internet; use an authenticated HTTPS endpoint. Do not put real tokens in
the repository. Without a reachable Ollama endpoint, the app page can load, but
AI answers and quiz generation will show an actionable connection error.

## Publish code updates

Streamlit Community Cloud deploys code from GitHub. Add and push the app,
learning module, tests, project metadata, and repository-root requirements
file; then create or reboot the Cloud app using the path above. Do not commit
virtual environments, Python cache files, or secrets.
