import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { ArrowLeft, Loader2, Upload, X, CheckCircle, Camera, Info } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useDropzone } from "react-dropzone";
import ResultadoImagen from "@/components/ResultadoImagen";
import { Alert, AlertDescription } from "@/components/ui/alert";

const DiagnosticoImagen = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [paso, setPaso] = useState<'validacion' | 'diagnostico'>('validacion');
  const [loading, setLoading] = useState(false);
  const [validando, setValidando] = useState(false);
  const [resultado, setResultado] = useState<any>(null);
  const [imagenFile, setImagenFile] = useState<File | null>(null);
  const [imagenPreview, setImagenPreview] = useState<string | null>(null);
  const [animalDetectado, setAnimalDetectado] = useState<'dog' | 'cat' | null>(null);

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

  const [imagenMascotaValidada, setImagenMascotaValidada] = useState<string | null>(null);

  const limpiarImagen = () => {
    setImagenFile(null);
    setImagenPreview(null);
    setResultado(null);
  };
  
  const volverAValidacion = () => {
    setImagenFile(null);
    setImagenPreview(null);
    setResultado(null);
    setPaso('validacion');
    setAnimalDetectado(null);
    setImagenMascotaValidada(null);
  };

  const validarMascota = async () => {
    if (!imagenFile) return;

    setValidando(true);
    try {
      const reader = new FileReader();
      reader.onload = async () => {
        const base64 = reader.result as string;
        
        const response = await fetch('http://48.221.120.179:8000//validate_pet', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ image: base64 })
        });

        if (!response.ok) {
          throw new Error('Error al validar la imagen');
        }

        const data = await response.json();
        
        if (data.is_valid) {
          setAnimalDetectado(data.animal_type);
          setImagenMascotaValidada(imagenPreview);
          setImagenFile(null);
          setImagenPreview(null);
          setPaso('diagnostico');
          toast({
            title: "¡Mascota detectada!",
            description: `Se ha detectado un ${data.animal_type === 'dog' ? 'perro' : 'gato'}. Ahora puedes proceder con la detección.`
          });
        } else {
          toast({
            title: "Imagen no válida",
            description: "Por favor, sube una foto clara de un perro o gato.",
            variant: "destructive"
          });
          setImagenFile(null);
          setImagenPreview(null);
        }
        setValidando(false);
      };
      reader.readAsDataURL(imagenFile);
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "No se pudo validar la imagen. Asegúrate de que el servidor esté corriendo.",
        variant: "destructive"
      });
      setValidando(false);
    }
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
        const response = await fetch('http://48.221.120.179:8000//predict_image', {
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
        
        // Reclasificar como "Healthy" si la suma de enfermedades es < 10%
        const nonHealthyPredictions = data.predictions.filter(
          (p: any) => p.class.toLowerCase() !== "healthy" && p.prob > 0
        );
        const sumNonHealthy = nonHealthyPredictions.reduce(
          (acc: number, p: any) => acc + p.prob, 0
        );
        
        let finalResult = data;
        if (sumNonHealthy < 0.10) {
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

        {paso === 'validacion' ? (
          <>
            <div className="text-center mb-8">
              <h1 className="text-4xl font-bold mb-2 text-foreground">¡Hora de Registrar tu Mascota! 🐾</h1>
              <p className="text-muted-foreground text-lg">Primero, verifiquemos que sea un perro o gato</p>
            </div>

            <Card className="border-primary/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Camera className="h-5 w-5" />
                  Sube una Foto de tu Mascota
                </CardTitle>
                <CardDescription>
                  Necesitamos verificar que sea un perro o gato antes de continuar con la detección
                </CardDescription>
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
                      Solo se aceptan fotos de perros y gatos
                    </p>
                  </div>
                ) : (
                  <>
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
                    <Button 
                      onClick={validarMascota} 
                      disabled={validando}
                      className="w-full"
                      size="lg"
                    >
                      {validando && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                      Validar Mascota
                    </Button>
                  </>
                )}
              </CardContent>
            </Card>

            <Alert className="mt-6">
              <Info className="h-4 w-4" />
              <AlertDescription>
                Este paso asegura que la foto sea de un perro o gato para poder realizar una detección precisa.
              </AlertDescription>
            </Alert>
          </>
        ) : (
          <>
            <div className="text-center mb-8">
              <h1 className="text-4xl font-bold mb-2 text-primary">Detección Dermatológica</h1>
              <p className="text-muted-foreground">Análisis de imagen de tu {animalDetectado === 'dog' ? 'perro' : 'gato'}</p>
            </div>

            {imagenMascotaValidada && (
              <Card className="mb-6 border-primary/20">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <CheckCircle className="h-5 w-5 text-primary" />
                    Mascota Validada: {animalDetectado === 'dog' ? 'Perro 🐕' : 'Gato 🐈'}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex justify-center">
                    <img 
                      src={imagenMascotaValidada} 
                      alt="Mascota validada" 
                      className="w-32 h-32 object-cover rounded-lg border-2 border-primary/20"
                    />
                  </div>
                </CardContent>
              </Card>
            )}

            <Card className="mb-6 border-primary/20">
              <CardHeader>
                <CardTitle className="text-primary">Imagen de la Zona Afectada</CardTitle>
                <CardDescription>Sube una foto clara de la zona dermatológica que necesitas analizar</CardDescription>
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
                    <Upload className="h-12 w-12 mx-auto mb-4 text-primary" />
                    <p className="text-lg font-medium mb-2">
                      {isDragActive ? 'Suelta la imagen aquí' : 'Arrastra una imagen o haz clic para seleccionar'}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Foto de la zona dermatológica a analizar
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="relative">
                      <img 
                        src={imagenPreview} 
                        alt="Zona a analizar" 
                        className="w-full h-auto rounded-lg"
                      />
                      <Button
                        variant="outline"
                        size="icon"
                        className="absolute top-2 right-2"
                        onClick={limpiarImagen}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>

            <div className="flex gap-4">
              <Button 
                variant="outline"
                onClick={volverAValidacion}
                className="flex-1"
                size="lg"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Cambiar Mascota
              </Button>
              <Button 
                onClick={analizar} 
                disabled={loading || !imagenFile}
                className="flex-1"
                size="lg"
              >
                {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Analizar Imagen
              </Button>
            </div>

            {resultado && imagenPreview && (
              <ResultadoImagen 
                resultado={resultado} 
                imagenOriginal={imagenPreview}
                animalType={animalDetectado}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default DiagnosticoImagen;
