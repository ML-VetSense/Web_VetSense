"""
FastAPI server for VetAI with real PyTorch model inference
Run with: uvicorn main:app --reload --port 8000
"""

from pyexpat import features
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from xgboost import XGBClassifier
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import joblib
import base64, io, numpy as np, matplotlib.pyplot as plt, cv2
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ----------------------------------------------------------
# CONFIGURACIÓN BASE
# ----------------------------------------------------------
app = FastAPI(title="VetAI Diagnostic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------
# CARGA DE MODELOS
# ----------------------------------------------------------
#MODEL_TABULAR_PATH = "../models/best_model_tabular.pth"
#MODEL_TABULAR_PATH = "../models/xgboost_model.pkl"
MODEL_TABULAR_PATH = "../models/the_xgboost_bundle.pkl"
SCALER_PATH = "../models/the_scaler.pkl"
ENCODERS_PATH = "../models/the_label_encoders.pkl"
MODEL_IMAGE_PATH = "../models/best_efficientnet_model_b.pth"

# === Imagen (EfficientNet-B0 modificado) ===
num_classes_full = 19
image_model = models.efficientnet_b0(weights='IMAGENET1K_V1')
num_features = image_model.classifier[1].in_features
image_model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(512, num_classes_full)
)
image_model.load_state_dict(torch.load(MODEL_IMAGE_PATH, map_location=device))
image_model.to(device)
image_model.eval()
print("✅ Modelo de imagen cargado correctamente.")

# === Tabular ===    
tabular_model = None
FEATURE_NAMES = None
numeric_cols = None
categorical_cols = None
label_encoder_target = None
class_names = None

try:
    bundle = joblib.load(MODEL_TABULAR_PATH)

    # Si es un diccionario con todos los componentes
    if isinstance(bundle, dict) and "model" in bundle:
        tabular_model = bundle["model"]
        FEATURE_NAMES = bundle.get("feature_names", None)
        numeric_cols = bundle.get("numeric_cols", None)
        categorical_cols = bundle.get("categorical_cols", None)
        label_encoder_target = bundle.get("target_encoder", None) or bundle.get("label_encoder_target", None)
        class_names = bundle.get("class_names", None) or (label_encoder_target.classes_ if label_encoder_target is not None else None)  
        
        print("✅ Modelo tabular y metadatos cargados correctamente (bundle).")
        if FEATURE_NAMES is not None:
            print(f"📊 Modelo espera {len(FEATURE_NAMES)} features.")

    else:
        # Podría ser un modelo suelto (pipeline) salvado sin metadatos
        if isinstance(bundle, XGBClassifier) or hasattr(bundle, "predict_proba"):
            tabular_model = bundle
            print("✅ Modelo tabular cargado (modelo individual).")
        else:
            raise ValueError("Formato de bundle no reconocido.")

except Exception as e:
    print(f"⚠️ Error al cargar modelo tabular o preprocesadores: {e}")
    tabular_model = None

# ----------------------------------------------------------
# TRANSFORMACIONES Y CLASES
# ----------------------------------------------------------
transform_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

classes_full = [
    "Dermatophytosis", "Mange_Scabies", "Flea_Allergy", "Dermatitis",
    "Allergic_Dermatitis", "Bacterial_Dermatosis", "Hot_Spots",
    "Dental_Disease", "Eye_Infection", "Ear_Mites",
    "Tick_Infestation", "Parvovirus", "Distemper", "Feline_Leukemia",
    "Feline_Panleukopenia", "Kennel_Cough", "Intestinal_Worms",
    "Urinary_Tract_Infection", "Healthy"
]

classes_visible = [
    "Dermatophytosis", "Mange_Scabies", "Flea_Allergy", "Dermatitis",
    "Allergic_Dermatitis", "Bacterial_Dermatosis", "Hot_Spots",
    "Eye_Infection", "Ear_Mites", "Tick_Infestation", "Healthy"
]
visible_indices = [classes_full.index(c) for c in classes_visible]

# ----------------------------------------------------------
# ESTRUCTURAS DE ENTRADA
# ----------------------------------------------------------
class ClinicalRequest(BaseModel):
    Animal_Type: Optional[str] = None
    Breed: Optional[str] = None
    Age: Optional[float] = None
    Gender: Optional[str] = None
    Weight: Optional[float] = None
    Duration: Optional[str] = None
    Appetite_Loss: Optional[int] = 0
    Vomiting: Optional[int] = 0
    Diarrhea: Optional[int] = 0
    Lethargy: Optional[int] = 0
    Coughing: Optional[int] = 0
    Sneezing: Optional[int] = 0
    Skin_Lesion_Present: Optional[int] = 0
    Skin_Lesion_Type: Optional[str] = "None"
    Skin_Lesion_Location: Optional[str] = "None"
    Hair_Loss: Optional[int] = 0
    Itching_Scratching: Optional[int] = 0
    Lesion_Color: Optional[str] = "None"
    Nasal_Discharge: Optional[int] = 0
    Eye_Discharge: Optional[int] = 0
    Eye_Redness: Optional[int] = 0
    Ear_Discharge: Optional[int] = 0
    Head_Shaking: Optional[int] = 0
    Bad_Breath: Optional[int] = 0
    Drooling: Optional[int] = 0
    Difficulty_Eating: Optional[int] = 0
    Frequent_Urination: Optional[int] = 0
    Blood_in_Urine: Optional[int] = 0
    Straining_to_Urinate: Optional[int] = 0
    Increased_Thirst: Optional[int] = 0
    Weakness: Optional[int] = 0
    Body_Temperature: Optional[float] = None
    Heart_Rate: Optional[float] = None
    Respiratory_Rate: Optional[float] = None


class ImageRequest(BaseModel):
    image: str  # base64 encoded image
    

# ----------------------------------------------------------
# FUNCIONES AUXILIARES 🧩
# ----------------------------------------------------------
import pandas as pd
def preprocess_input(data: ClinicalRequest):
    """
    Preprocesa los datos clínicos con los encoders y scaler reales
    """

    if FEATURE_NAMES is None:
        # si no hay feature names, construimos DataFrame directo
        df = pd.DataFrame([data.dict()])
        return df
    
    # start from request dict
    req = data.dict()

    
    # create empty row filled with defaults depending on numeric/categorical
    row = {}
    for col in FEATURE_NAMES:
        if col in req and req[col] is not None:
            row[col] = req[col]
        else:
            # si sabemos que es numérica -> default 0; si categórica -> "None" o 0 según bundle info
            if numeric_cols and col in numeric_cols:
                # si Body_Temperature/Heart_Rate/Respiratory_Rate pueden ser None, usa mean-like fallback 0
                row[col] = 0.0
            elif categorical_cols and col in categorical_cols:
                # para categorías usamos "Unknown" o "None" string; OneHotEncoder handle_unknown="ignore" en training
                row[col] = "Unknown"
            else:
                # fallback genérico
                row[col] = 0

    df = pd.DataFrame([row], columns=FEATURE_NAMES)
    return df


# ----------------------------------------------------------
# FUNCIÓN GRAD-CAM
# ---------------z-------------------------------------------
def generate_gradcam(model, input_tensor, target_class):
    """
    Generates Grad-CAM heatmap for the predicted class
    """
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.features[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item() if target_class is None else target_class

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].detach()
    acts = activations[0].detach()

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    forward_handle.remove()
    backward_handle.remove()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

# ----------------------------------------------------------
# ENDPOINT DE ESTADO
# ----------------------------------------------------------
@app.get("/")
def root():
    return {"message": "VetAI Diagnostic API 🧠", "status": "running"}

# ----------------------------------------------------------
# PREDICCIÓN CLÍNICA (TABULAR)
# ----------------------------------------------------------
@app.post("/predict_clinical")
async def predict_clinical(request: ClinicalRequest):
    """
    Predict disease from clinical data using the tabular model
    """
    if tabular_model is None:
        raise HTTPException(status_code=500, detail="Modelo tabular no cargado.")

    try:
         # 🧩 Preprocesamiento
        X_df = preprocess_input(request)
        
        # Debug info
        if FEATURE_NAMES is not None:
            sent_cols = list(X_df.columns)
            print(f"🧾 Columnas enviadas: {len(sent_cols)}")
            print(sent_cols)
            print(f"📊 Columnas esperadas por el modelo: {len(FEATURE_NAMES)}")
            # detect extra / missing
            extra = [c for c in sent_cols if c not in FEATURE_NAMES]
            missing = [c for c in FEATURE_NAMES if c not in sent_cols]
            if extra:
                print(f"⚠️ EXTRA: {extra}")
            if missing:
                print(f"⚠️ MISSING: {missing}")

        
        # 🔮 Predicción
        probs = tabular_model.predict_proba(X_df)[0]
        
        
        # Get class names from bundle / label encoder
        if class_names is not None:
            _class_names = class_names
        elif label_encoder_target is not None:
            _class_names = label_encoder_target.classes_
        else:
            # fallback - try to infer from model (not always possible)
            try:
                _class_names = [str(i) for i in range(len(probs))]
            except:
                _class_names = []
            
        
        # order top predictions
        top_idx = np.argsort(probs)[::-1][:5]
        predictions = [{"class": _class_names[i] if i < len(_class_names) else str(i), "prob": float(probs[i])} for i in top_idx]

        return {
            "predictions": predictions,
            "top_class": predictions[0]["class"],
            "top_prob": predictions[0]["prob"],
            "explanations": {
                "features": FEATURE_NAMES if FEATURE_NAMES else "unknown",
                "method": "pipeline (preprocessor + xgboost)"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción clínica: {str(e)}")

# ----------------------------------------------------------
# PREDICCIÓN POR IMAGEN (EFFICIENTNET)
# ----------------------------------------------------------
@app.post("/predict_image")
async def predict_image(request: ImageRequest):
    """
    Predict disease from image using the EfficientNet model
    """
    try:
        image_data = request.image.split(",")[1] if "," in request.image else request.image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_np = np.array(image)

        img_tensor = transform_image(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = image_model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0]

            visible_probs = probs[visible_indices]
            confidence, predicted_idx = torch.max(visible_probs, 0)
            
            pred_class_idx = visible_indices[predicted_idx.item()]
            
         # 🧠 Grad-CAM
        heatmap = generate_gradcam(image_model, img_tensor, pred_class_idx)
        heatmap_resized = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
        overlay = cv2.addWeighted(original_np, 0.5, heatmap_resized, 0.5, 0)

        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        gradcam_b64 = base64.b64encode(buffer).decode('utf-8')

        predictions = [
            {"class": classes_visible[i], "prob": float(visible_probs[i])}
            for i in np.argsort(visible_probs.cpu().numpy())[::-1]
        ][:5]

        return {
            "predictions": predictions,
            "top_class": classes_visible[predicted_idx.item()],
            "top_prob": float(confidence.item()),
            "gradcam_url": f"data:image/png;base64,{gradcam_b64}",
            "category": "Desease Detected" if classes_visible[predicted_idx.item()] != "Healthy" else "Healthy"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción por imagen: {str(e)}")

# ----------------------------------------------------------
# EJECUCIÓN LOCAL
# ----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
