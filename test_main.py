from fastapi.testclient import TestClient

from main import app
from model.model import LABEL_LOOKUP

client = TestClient(app)


def test__status_livez():
    res = client.get("/_status/livez")
    assert res.status_code == 200
    assert res.json() == {"ok": True}


def test__status_readyz():
    res = client.get("/_status/readyz")
    assert res.status_code == 200
    assert res.json() == {"ok": True}


def test_predict():
    test_img_path = "./nv-test-image.jpg"
    with open(test_img_path, "rb") as img:
        res = client.post(
            "/predict", files={"file": (test_img_path, img, "image/jpeg")}
        )

    assert res.status_code == 200

    pred = res.json()
    assert pred["prediction"] in LABEL_LOOKUP.values()
    assert pred["confidence"] >= 0 and pred["confidence"] <= 1
