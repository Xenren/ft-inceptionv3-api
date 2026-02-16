import sys

from fastapi import FastAPI, HTTPException, Response, UploadFile, status

from model.model import InceptionClassifier

app = FastAPI()


def load_model_with_retry(num_retries: int):
    max_retries = 5
    if num_retries <= max_retries:
        try:
            return InceptionClassifier()
        except RuntimeError as e:
            print(f"\nFailed to load model: Error loading state_dict: {e}")
            load_model_with_retry(num_retries + 1)
        except FileNotFoundError:
            print("\nFailed to load model: Checkpoint file not found.")
        except Exception as e:
            print(f"\nFailed to load model: An unexpected error occurred: {e}")
            load_model_with_retry(num_retries + 1)

    print("\nMaximum retries reached: shutting down")
    sys.exit()


model = load_model_with_retry(num_retries=0)


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
