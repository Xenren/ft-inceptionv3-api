from fastapi import FastAPI, HTTPException, Response, UploadFile, status

from model.model import InceptionClassifier

app = FastAPI()
model = InceptionClassifier()


@app.get("/_status/livez", status_code=status.HTTP_200_OK)
async def healthz():
    return {"ok": True}


@app.get("/_status/readyz", status_code=status.HTTP_200_OK)
async def readyz(response: Response):
    if not model:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"ok": False, "reason": "failed to load model"}

    return {"ok": True}


@app.post("/predict")
async def predict(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="unsupported file type")

    img_bytes = await file.read()
    res = model.inference(img_bytes)

    return {"prediction": res.pred_label, "confidence": res.confidence}
