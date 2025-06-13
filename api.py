from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from summarization import SummaryModel, create_summary_template, generate_summary
from transformers import PreTrainedTokenizer, PreTrainedModel

class SummarizationInput(BaseModel):
    """Input model for summarization API."""
    Translated_conversation: str

def create_api(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        model: Pre-trained language model
        tokenizer: Model tokenizer
    
    Returns:
        Configured FastAPI app
    """
    app = FastAPI()

    @app.post("/summarize")
    def summarize(input_data: SummarizationInput):
        try:
            en_text = input_data.Translated_conversation
            summary_prompt = create_summary_template(en_text, SummaryModel)
            summary_output = generate_summary(summary_prompt, tokenizer, model)
            return summary_output
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app