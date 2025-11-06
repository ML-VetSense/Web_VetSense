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
MODEL_TABULAR_PATH = "../models/xgboost_model.pkl"
SCALER_PATH = "../models/scaler.pkl"
ENCODERS_PATH = "../models/label_encoders.pkl"
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
try:
    tabular_model = joblib.load(MODEL_TABULAR_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    print("✅ Modelo tabular y preprocesadores cargados correctamente.")
except Exception as e:
    print(f"⚠️ Error al cargar modelo tabular o preprocesadores: {e}")
    tabular_model, scaler, label_encoders = None, None, None

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

TABULAR_CLASSES = [
    "Parvovirus in Dog", "Worm Infection", "Distemper",
    "Gastroenteritis", "Food Poisoning"
]

# ----------------------------------------------------------
# ESTRUCTURAS DE ENTRADA
# ----------------------------------------------------------
class AnimalData(BaseModel):
    species: str
    age: float
    weight: float

class Vitals(BaseModel):
    temperature: Optional[float] = None
    heart_rate: Optional[float] = None
    resp_rate: Optional[float] = None

class ClinicalRequest(BaseModel):
    animal: AnimalData
    symptoms: List[str]
    vitals: Vitals

class ImageRequest(BaseModel):
    image: str  # base64 encoded image
    

# ----------------------------------------------------------
# FUNCIONES AUXILIARES 🧩
# ----------------------------------------------------------
def preprocess_input(data: ClinicalRequest):
    """
    Preprocesa los datos clínicos con los encoders y scaler reales
    """
    # Ejemplo: según tu dataset real, ajusta el orden de features
    input_dict = {
        "Species": data.animal.species,
        "Age": data.animal.age,
        "Weight": data.animal.weight,
        "Temperature": data.vitals.temperature or 38.0,
        "Heart_Rate": data.vitals.heart_rate or 80,
        "Resp_Rate": data.vitals.resp_rate or 25,
        "Num_Symptoms": len(data.symptoms)
    }

    # Codificar variables categóricas
    for col, encoder in label_encoders.items():
        if col in input_dict:
            try:
                input_dict[col] = encoder.transform([input_dict[col]])[0]
            except:
                input_dict[col] = 0  # valor desconocido → categoría base

    # Convertir a array numpy y escalar
    X = np.array([list(input_dict.values())])
    X_scaled = scaler.transform(X)
    return X_scaled


# ----------------------------------------------------------
# FUNCIÓN GRAD-CAM
# ----------------------------------------------------------
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
        X_scaled = preprocess_input(request)
        
        print("Shape de X:", X_scaled.shape)
        print("Columnas:", X_scaled.columns)

        
        # 🔮 Predicción
        probs = tabular_model.predict_proba(X_scaled)[0]

        predictions = [
            {"class": TABULAR_CLASSES[i], "prob": float(probs[i])}
            for i in np.argsort(probs)[::-1]
        ][:5]

        return {
            "predictions": predictions,
            "top_class": predictions[0]["class"],
            "top_prob": predictions[0]["prob"],
            "explanations": {
                "features": ["Age", "Weight", "Temperature", "Heart Rate", "Resp Rate", "Symptoms"],
                "method": "feature importance (pending)"
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
            #"gradcam_url": None,  # Will be generated when model is implemented
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
