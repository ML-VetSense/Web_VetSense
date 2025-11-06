# VetAI Python Server

Este servidor FastAPI carga tus modelos PyTorch y expone endpoints para predicciones.

## Instalación

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Configuración de Modelos

1. Asegúrate de que tus modelos estén en la carpeta `models/`:
   - `models/best_model_tabular.pth` (modelo tabular)
   - `models/best_efficientnet_model_b.pth` (modelo de imagen)

2. Edita `main.py` y descomenta las líneas de carga de modelos:
   ```python
   tabular_model = torch.load(MODEL_TABULAR_PATH)
   image_model = torch.load(MODEL_IMAGE_PATH)
   ```

3. Implementa la lógica de inferencia en cada endpoint:
   - `predict_clinical()`: Procesa datos clínicos
   - `predict_image()`: Procesa imágenes

## Ejecutar el Servidor

```bash
# Desde la carpeta python_server
uvicorn main:app --reload --port 8000
```

El servidor estará disponible en: `http://localhost:8000`

## Endpoints

### GET /
Health check del servidor

### POST /predict_clinical
Predicción basada en datos clínicos

**Request:**
```json
{
  "animal": {
    "species": "dog",
    "age": 3,
    "weight": 10.5
  },
  "symptoms": ["vomiting", "lethargy"],
  "vitals": {
    "temperature": 38.2,
    "heart_rate": 95,
    "resp_rate": 20
  }
}
```

### POST /predict_image
Predicción basada en imagen

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

## Desarrollo

Para implementar tus modelos:

1. Carga los modelos en el inicio de la aplicación
2. Define las transformaciones necesarias para las imágenes
3. Implementa la lógica de preprocesamiento de datos clínicos
4. Actualiza las listas `TABULAR_CLASSES` e `IMAGE_CLASSES` con tus clases reales
5. Genera Grad-CAM para explicabilidad visual (opcional)
6. Calcula SHAP values para el modelo tabular (opcional)

## Testing

Puedes probar los endpoints con:

```bash
# Test health check
curl http://localhost:8000/

# Test clinical prediction
curl -X POST http://localhost:8000/predict_clinical \
  -H "Content-Type: application/json" \
  -d '{"animal":{"species":"dog","age":3,"weight":10},"symptoms":["vomiting"],"vitals":{"temperature":38}}'
```
