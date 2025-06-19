# Medical Summarization API

This project provides a REST API for summarizing medical conversations. It loads a quantized language model and exposes an endpoint that returns a structured JSON summary of patient information.

## Features

- **Quantized model loading** using the `transformers` library and `bitsandbytes` for reduced memory usage.
- **FastAPI** service exposing a `/summarize` endpoint.
- **ngrok integration** to make the API publicly accessible during development.
- Summaries follow a predefined Pydantic schema with fields such as symptoms, duration and risk factors.

## Requirements

- Python 3.10+
- CUDA capable GPU (if running with GPU acceleration)
- See `requirements.txt` for the full list of Python dependencies.

## Installation

1. Clone the repository.

```bash
git clone <repo-url>
cd medical-summarization
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Configuration

Update `config.py` or set environment variables to configure the model and ngrok tokens:

- `HF_TOKEN` – Hugging Face access token
- `NGROK_AUTH_TOKEN` – ngrok authentication token

These can also be exported in your shell before running the application:

```bash
export HF_TOKEN=<your-hf-token>
export NGROK_AUTH_TOKEN=<your-ngrok-token>
```

## Running the API

Run the main script to launch the API. It will automatically load the model, start ngrok, and serve the FastAPI app on the configured port.

```bash
python main.py
```

The public ngrok URL will be printed in the console, e.g. `https://abcd1234.ngrok.io`. You can send POST requests to `https://abcd1234.ngrok.io/summarize` with a JSON body:

```json
{
  "Translated_conversation": "Patient describes persistent headache..."
}
```

The API returns a JSON summary according to the schema defined in `SummaryModel`.

## Project Structure

- `api.py` – defines the FastAPI routes and request model.
- `main.py` – entry point that loads the model and runs the API with ngrok.
- `model_loader.py` – loads the quantized model and tokenizer.
- `summarization.py` – contains helper functions for template creation and summary generation.
- `config.py` – configuration constants for the model and API.
- `utils.py` – utility functions (e.g. clearing the GPU cache).

