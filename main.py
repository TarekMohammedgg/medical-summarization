import nest_asyncio
from pyngrok import ngrok
import uvicorn
import os
from config import BASE_MODEL_ID, HF_TOKEN, NGROK_PORT, NGROK_AUTH_TOKEN
from model_loader import load_quantized_model
from api import create_api
from utils import clear_gpu_cache

def main():
    """Main function to set up and run the summarization API."""
    nest_asyncio.apply()
    clear_gpu_cache()
    
    # Fetch tokens from environment variables
    hf_token = os.environ.get("HF_TOKEN", HF_TOKEN)
    ngrok_auth_token = os.environ.get("NGROK_AUTH_TOKEN", NGROK_AUTH_TOKEN)
    
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not provided")
    if not ngrok_auth_token:
        raise ValueError("NGROK_AUTH_TOKEN environment variable not provided")
    
    # Load model and tokenizer
    model, tokenizer = load_quantized_model(
        model_name=BASE_MODEL_ID,
        auth_token=hf_token
    )
    print(f"Model loaded on device: {model.device}")
    
    # Set up ngrok
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