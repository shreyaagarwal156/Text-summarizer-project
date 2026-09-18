import re

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_DIR = "./saved_summary_model"
MAX_INPUT_TOKENS = 512
MAX_SUMMARY_TOKENS = 150

app = FastAPI(
    title="Text Summarizer App",
    description="Text summarization using a fine-tuned T5 model",
    version="1.0",
)


# --- device ---------------------------------------------------------------
# CUDA first, then Apple Silicon, then CPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Loading model from {MODEL_DIR} onto {device}...")

model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)

model.to(device)
model.eval()  # turns off dropout, etc.


# --- schema ---------------------------------------------------------------
class DialogueInput(BaseModel):
    dialogue: str = Field(..., min_length=1)


# --- core -----------------------------------------------------------------
def clean_data(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)   # strip html tags first
    text = re.sub(r"\r\n", " ", text)    # normalise line breaks
    text = re.sub(r"\s+", " ", text)     # collapse whitespace
    # If you fine-tuned on lowercased text, add .lower() back here.
    return text.strip()


@torch.no_grad()
def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=MAX_SUMMARY_TOKENS,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# --- routes ---------------------------------------------------------------
# Plain `def`, not `async def`: generation is blocking work, so FastAPI runs it
# in a threadpool instead of stalling the event loop for every other request.
@app.post("/summarize/")
def summarize(dialogue_input: DialogueInput):
    dialogue = dialogue_input.dialogue.strip()

    if not dialogue:
        raise HTTPException(status_code=400, detail="Add some text to summarize.")

    try:
        summary = summarize_dialogue(dialogue)
    except Exception as exc:
        print(f"Summarization failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="The model could not process that text. Try shortening it.",
        )

    if not summary:
        raise HTTPException(
            status_code=422,
            detail="The model returned an empty summary. Try adding more context.",
        )

    return {"summary": summary}


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)