import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useNavigate } from "react-router-dom";
import { FileText, Image } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      <div className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          <div className="space-y-4">
            <h1 className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/60">
              VetAI Diagnostic Demo
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Asistente de diagnóstico inteligente para veterinarios
            </p>
            <p className="text-sm text-muted-foreground">
              Utiliza IA avanzada para analizar síntomas clínicos e imágenes dermatológicas
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6 mt-12">
            <Card 
              className="cursor-pointer transition-all hover:shadow-lg hover:scale-105 border-2 hover:border-primary/50"
              onClick={() => navigate("/diagnostico/texto")}
            >
              <CardHeader className="text-center">
                <div className="mx-auto mb-4 h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center">
                  <FileText className="h-8 w-8 text-primary" />
                </div>
                <CardTitle className="text-2xl">Diagnóstico por Texto</CardTitle>
                <CardDescription className="text-base">
                  Análisis basado en síntomas y datos clínicos del animal
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button className="w-full" size="lg">
                  Iniciar Análisis Tabular
                </Button>
              </CardContent>
            </Card>

            <Card 
              className="cursor-pointer transition-all hover:shadow-lg hover:scale-105 border-2 hover:border-primary/50"
              onClick={() => navigate("/diagnostico/imagen")}
            >
              <CardHeader className="text-center">
                <div className="mx-auto mb-4 h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center">
                  <Image className="h-8 w-8 text-primary" />
                </div>
                <CardTitle className="text-2xl">Diagnóstico por Imagen</CardTitle>
                <CardDescription className="text-base">
                  Análisis dermatológico mediante reconocimiento de imágenes
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button className="w-full" size="lg">
                  Subir Imagen
                </Button>
              </CardContent>
            </Card>
          </div>

          <div className="mt-12 p-6 bg-muted/50 rounded-lg">
            <h3 className="font-semibold text-lg mb-2">⚠️ Aviso Importante</h3>
            <p className="text-sm text-muted-foreground">
              Esta es una herramienta de asistencia diagnóstica. Los resultados son orientativos 
              y no sustituyen el criterio clínico profesional ni una consulta veterinaria completa.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
