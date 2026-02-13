from fastapi import FastAPI, HTTPException, UploadFile

from model.model import InceptionClassifier

app = FastAPI()
model = InceptionClassifier()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="unsupported file type")

    img_bytes = await file.read()
    res = model.inference(img_bytes)

    return {"prediction": res.pred_label, "confidence": res.confidence}
