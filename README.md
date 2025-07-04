
 # Medical Summarization API
 
-This project provides a REST API for summarizing medical conversations. It loads a quantized language model and exposes an endpoint that returns a structured JSON summary of patient information.
+This project exposes a REST service that summarizes medical conversations. It leverages a quantized language model and returns the output in a structured JSON format.
 
 ## Features
 
-- **Quantized model loading** using the `transformers` library and `bitsandbytes` for reduced memory usage.
-- **FastAPI** service exposing a `/summarize` endpoint.
-- **ngrok integration** to make the API publicly accessible during development.
-- Summaries follow a predefined Pydantic schema with fields such as symptoms, duration and risk factors.
+- **Quantized model loading** using the `transformers` library and `bitsandbytes`.
+- **FastAPI** service available at `/summarize`.
+- **ngrok integration** so the service can be reached from the public internet during development.
+- Summaries follow a strict Pydantic schema.
 
 ## Requirements
 
-- Python 3.10+
-- CUDA capable GPU (if running with GPU acceleration)
-- See `requirements.txt` for the full list of Python dependencies.
+The main Python dependencies are listed in `requirements.txt`:
+
+```
+streamlit==1.36.0
+requests==2.31.0
+fastapi==0.115.12
+nest-asyncio==1.6.0
+pyngrok==7.2.5
+uvicorn==0.34.2
+bitsandbytes==0.45.5
+accelerate==0.26.0
+transformers
+pydantic
+torch
+```
+
+Python 3.10+ and a CUDA capable GPU are recommended if running with hardware acceleration.
 
 ## Installation
 
-1. Clone the repository.
 
 ```bash
 git clone https://github.com/TarekMohammedgg/medical-summarization
 cd medical-summarization
```

-2. Install the required packages:

```bash
 pip install -r requirements.txt
 ```
 
 ## Configuration
 
-Update `config.py` or set environment variables to configure the model and ngrok tokens:
+Edit `config.py` or set the following environment variables before running:
 
-- `HF_TOKEN` – Hugging Face access token
-- `NGROK_AUTH_TOKEN` – ngrok authentication token
+- `HF_TOKEN` – Hugging Face token used to download the model.
+- `NGROK_AUTH_TOKEN` – your ngrok authentication token.
 
-These can also be exported in your shell before running the application:
+Example:
 
 ```bash
 export HF_TOKEN=<your-hf-token>
 export NGROK_AUTH_TOKEN=<your-ngrok-token>
 ```
 
 ## Running the API
 
-Run the main script to launch the API. It will automatically load the model, start ngrok, and serve the FastAPI app on the configured port.
+Execute the entry point:
 
 ```bash
 python main.py
 ```
 
-The public ngrok URL will be printed in the console, e.g. `https://abcd1234.ngrok.io`. You can send POST requests to `https://abcd1234.ngrok.io/summarize` with a JSON body:
+A public ngrok URL will be printed. Send a POST request to `/summarize` with a JSON body containing `Translated_conversation`.
+
+Example request:
 
 ```json

{ "Translated_conversation": "Patient: How are you? \nAssistant: I'm fine, praise be to God. Can you tell me what your problem is?\nPatient: I have an itch in my left shoulder.\nAssistant: When did this itch start?\nPatient: It started about a week ago, but the pain was mild at first, but it increased in the last few days.\nAssistant: Have you noticed any changes in the area you're experiencing, like a change in color, size, texture, or the appearance of scales or any other changes?\nPatient: It was normal at first, but with all the scratching, it became red.\nAssistant: Are there any other symptoms accompanying the itch, such as pain, bleeding, or any other sensation?\nPatient: No, I haven't used any ointments, but I scratch it with cool water from time to time, and there are no side effects with the itch.\nAssistant: Is there anything that increases the itch, such as tight clothing or friction?\nPatient: No, because my work is from home, and I haven't gotten sunburned or any burns in that area, but the problem started when I ate fish.\nAssistant: Is there a family history of similar cases, such as skin cancer or unusual moles?\nPatient: It's an intense itch, and yes, my father has an allergy to seafood.\nAssistant: Is there anything else you want to add?\nPatient: No.\nAssistant: Thank you for the information. Bring the picture that shows the itch so we can consider it, and we'll discuss it with the doctor." }

```
 
-The API returns a JSON summary according to the schema defined in `SummaryModel`.
+The response is a JSON object describing the conversation details using the schema defined in `SummaryModel`.
```json
{ "Patient_symptoms": [ "Intense itch in left shoulder", "Mild pain that increased in the last few days", "Redness due to scratching" ], "Symptom_location": "Left shoulder", "Duration": "About a week", "Symptom_progression": "Mild pain at first, increased in the last few days", "Risk_factors": "Fish consumption, family history of seafood allergy" }
```


+## How it works
+
+1. **`summarization.py`**
+   - Defines `SummaryModel`, a Pydantic model describing the fields expected in the summary.
+   - `create_summary_template()` builds a chat-style prompt combining the user conversation with the JSON schema.
+   - `generate_summary()` runs the language model on that prompt, extracts the JSON block from the output and returns it as a Python dictionary.
+
+2. **`model_loader.py`**
+   - Loads the quantized language model and tokenizer using `transformers` and `bitsandbytes`.
+
+3. **`api.py`**
+   - Declares a `SummarizationInput` Pydantic model for the request body.
+   - Exposes the `/summarize` endpoint. The endpoint creates a prompt using `create_summary_template()`, generates the model response with `generate_summary()`, and returns the JSON result.
+
+4. **`main.py`**
+   - Loads the model with `load_quantized_model()`.
+   - Starts an ngrok tunnel and launches the FastAPI application.
 
-## Project Structure
+5. **`config.py`**
+   - Stores configuration constants such as the model identifier and default tokens.
 
-- `api.py` – defines the FastAPI routes and request model.
-- `main.py` – entry point that loads the model and runs the API with ngrok.
-- `model_loader.py` – loads the quantized model and tokenizer.
-- `summarization.py` – contains helper functions for template creation and summary generation.
-- `config.py` – configuration constants for the model and API.
-- `utils.py` – utility functions (e.g. clearing the GPU cache).
+6. **`utils.py`**
+   - Provides utility helpers like clearing the GPU cache.
 
+This pipeline first summarizes the conversation locally, ensures the model response is valid JSON according to the Pydantic schema, and finally serves that response through the FastAPI API.
