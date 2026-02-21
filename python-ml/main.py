import base64
import io
import time
from contextlib import asynccontextmanager

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

pipe = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    print("Loading model...")
    pipe = DiffusionPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()
    print("Model loaded.")
    yield
    del pipe
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def generate(prompt: str = "a lovely cat", seed: int = 42, steps: int = 4):
    start = time.time()
    generator = torch.Generator("cuda").manual_seed(seed)
    result = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=1.0,
        generator=generator,
    )
    image = result.images[0]
    elapsed = time.time() - start

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return f"""<!DOCTYPE html>
<html>
<head><title>SD - {prompt}</title></head>
<body style="background:#111;color:#eee;font-family:sans-serif;display:flex;flex-direction:column;align-items:center;padding:2rem">
  <h2>{prompt}</h2>
  <img src="data:image/png;base64,{b64}" style="max-width:512px" />
  <p>Generated in {elapsed:.2f}s (seed={seed}, steps={steps})</p>
</body>
</html>"""
