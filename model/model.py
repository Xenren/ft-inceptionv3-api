import io
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import inception_v3

# has to match order from training exactly
CLASS_CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

LABEL_LOOKUP = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}

IDX_TO_CODE = CLASS_CODES
IDX_TO_LABEL = [LABEL_LOOKUP[c] for c in IDX_TO_CODE]


class InferenceResult:
    def __init__(
        self, pred_idx: int, pred_code: str, pred_label: str, confidence: float
    ) -> None:
        self.pred_idx = pred_idx
        self.pred_code = pred_code
        self.pred_label = pred_label
        self.confidence = confidence


class InceptionClassifier:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_ft = inception_v3(weights=None, aux_logits=True)

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(LABEL_LOOKUP))
        if model_ft.aux_logits and model_ft.AuxLogits is not None:
            num_ftrs_aux = model_ft.AuxLogits.fc.in_features  # pyright: ignore
            model_ft.AuxLogits.fc = nn.Linear(
                num_ftrs_aux, len(LABEL_LOOKUP)  # pyright: ignore
            )
        model_ft.load_state_dict(
            torch.load("model/skin_cancer_classifer_inceptionv3.pt", weights_only=True)
        )
        model_ft = model_ft.to(self.device)
        model_ft.eval()
        self.model = model_ft

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose(
            [
                transforms.Resize((320, 320)),  # resize slightly larger
                transforms.CenterCrop(299),  # crop to target size
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ]
        )

    def inference(self, img_bytes: bytes) -> InferenceResult:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input: torch.Tensor = self.transform(img)  # pyright: ignore
        input = torch.unsqueeze(input, 0)

        with torch.inference_mode():
            output = self.model(input)  # logits
            probs = F.softmax(output, dim=1)  # convert logits to probabilities
            print(probs)
            pred_idx = int(
                torch.argmax(probs).item()
            )  # idx of highest prob is model prediction

        return InferenceResult(
            pred_idx,
            IDX_TO_CODE[pred_idx],
            IDX_TO_LABEL[pred_idx],
            float(probs[0][pred_idx].item()),
        )
