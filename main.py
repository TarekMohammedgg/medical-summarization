# %%writefile main.py

import nest_asyncio
from pyngrok import ngrok
import uvicorn
from google.colab import userdata
from config import BASE_MODEL_ID, HF_TOKEN, NGROK_PORT, NGROK_AUTH_TOKEN
from model_loader import load_quantized_model
from api import create_api
from utils import clear_gpu_cache

def main():
    """Main function to set up and run the summarization API."""
    # Apply nest_asyncio for Colab compatibility
    nest_asyncio.apply()

    # Clear GPU cache
    clear_gpu_cache()

    # Load model and tokenizer
    model, tokenizer = load_quantized_model(
        model_name=BASE_MODEL_ID,
        auth_token=HF_TOKEN
    )
    print(f"Model loaded on device: {model.device}")

    # Set up ngrok
    ngrok_auth_token = userdata.get("NGROK_AUTH_TOKEN") or NGROK_AUTH_TOKEN
    if not ngrok_auth_token:
        raise ValueError("NGROK_AUTH_TOKEN not provided")
    
    ngrok.set_auth_token(ngrok_auth_token)
    
    # Close existing tunnels
    for tunnel in ngrok.get_tunnels():
        ngrok.disconnect(tunnel.public_url)
    
    # Create new tunnel
    public_url = ngrok.connect(NGROK_PORT)
    print("🚀 Your API is live at:", public_url)

    # Create and run FastAPI app
    app = create_api(model, tokenizer)
    uvicorn.run(app, host="0.0.0.0", port=NGROK_PORT)

if __name__ == "__main__":
    main()