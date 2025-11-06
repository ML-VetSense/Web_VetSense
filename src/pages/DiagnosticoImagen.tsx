import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { ArrowLeft, Loader2, Upload, X } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useDropzone } from "react-dropzone";
import ResultadoImagen from "@/components/ResultadoImagen";

const DiagnosticoImagen = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [resultado, setResultado] = useState<any>(null);
  const [imagenFile, setImagenFile] = useState<File | null>(null);
  const [imagenPreview, setImagenPreview] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setImagenFile(file);
      
      const reader = new FileReader();
      reader.onload = () => {
        setImagenPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp']
    },
    maxFiles: 1
  });

  const limpiarImagen = () => {
    setImagenFile(null);
    setImagenPreview(null);
    setResultado(null);
  };

  const analizar = async () => {
    if (!imagenFile) {
      toast({
        title: "No hay imagen",
        description: "Por favor sube una imagen primero",
        variant: "destructive"
      });
      return;
    }

    setLoading(true);
    try {
      // Convert image to base64
      const reader = new FileReader();
      reader.onload = async () => {
        const base64 = reader.result as string;
        
        // Call local Python server directly for local development
        const response = await fetch('http://localhost:8000/predict_image', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ image: base64 })
        });

        if (!response.ok) {
          throw new Error('Error al analizar la imagen');
        }

        const data = await response.json();

        // Reclasificar como "Healthy" si la suma de enfermedades es < 5%
        const nonHealthyPredictions = data.predictions.filter(
          (p: any) => p.class.toLowerCase() !== "healthy" && p.prob > 0
        );
        const sumNonHealthy = nonHealthyPredictions.reduce(
          (acc: number, p: any) => acc + p.prob, 0
        );

        let finalResult = data;
        if (sumNonHealthy < 0.05) {
          const healthyProb = 1 - sumNonHealthy;
          finalResult = {
            ...data,
            top_class: "Healthy",
            top_prob: healthyProb,
            predictions: [
              { class: "Healthy", prob: healthyProb },
              ...data.predictions.filter((p: any) => p.prob >= 0.01) // Solo mostrar >= 1%
            ],
            category: "healthy",
            isReclassified: true,
            originalSum: sumNonHealthy,
            originalPredictions: nonHealthyPredictions // Guardar las predicciones originales
          };
        }

        setResultado(finalResult);

        toast({
          title: "Análisis completado",
          description: "Los resultados están listos"
        });
        setLoading(false);
      };
      reader.readAsDataURL(imagenFile);
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "No se pudo conectar con el servidor Python. Asegúrate de que esté corriendo en http://localhost:8000",
        variant: "destructive"
      });
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5 p-6">
      <div className="max-w-4xl mx-auto">
        <Button 
          variant="ghost" 
          onClick={() => navigate("/")}
          className="mb-6"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Volver
        </Button>

        <h1 className="text-4xl font-bold mb-2 text-foreground">Diagnóstico por Imagen</h1>
        <p className="text-muted-foreground mb-8">Análisis dermatológico basado en imágenes del animal</p>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Cargar Imagen</CardTitle>
              <CardDescription>Sube una foto clara de la zona afectada</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {!imagenPreview ? (
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
                    isDragActive ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50'
                  }`}
                >
                  <input {...getInputProps()} />
                  <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-lg font-medium mb-2">
                    {isDragActive ? 'Suelta la imagen aquí' : 'Arrastra una imagen o haz clic para seleccionar'}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Formatos: PNG, JPG, JPEG, WEBP
                  </p>
                </div>
              ) : (
                <div className="relative">
                  <img 
                    src={imagenPreview} 
                    alt="Preview" 
                    className="w-full h-auto rounded-lg"
                  />
                  <Button
                    variant="destructive"
                    size="icon"
                    className="absolute top-2 right-2"
                    onClick={limpiarImagen}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          <Button 
            onClick={analizar} 
            disabled={loading || !imagenFile}
            className="w-full"
            size="lg"
          >
            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Analizar Imagen
          </Button>

          {resultado && imagenPreview && (
            <ResultadoImagen 
              resultado={resultado} 
              imagenOriginal={imagenPreview}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default DiagnosticoImagen;
