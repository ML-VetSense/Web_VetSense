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
            <h1 className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary via-secondary to-primary">
              VetSense
            </h1>
            <p className="text-xl text-foreground max-w-2xl mx-auto font-medium">
              Sistema Inteligente de Detección Veterinaria
            </p>
            <p className="text-base text-muted-foreground">
              Tecnología de IA avanzada para detección de enfermedades dermatológicas en perros y gatos
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6 mt-12">
            <Card 
              className="cursor-pointer transition-all hover:shadow-xl hover:scale-[1.02] border-2 hover:border-primary bg-card/50 backdrop-blur"
              onClick={() => navigate("/diagnostico/texto")}
            >
              <CardHeader className="text-center">
                <div className="mx-auto mb-4 h-16 w-16 rounded-full bg-gradient-to-br from-secondary to-secondary/70 flex items-center justify-center shadow-lg">
                  <FileText className="h-8 w-8 text-secondary-foreground" />
                </div>
                <CardTitle className="text-2xl text-primary">Detección por Datos Clínicos</CardTitle>
                <CardDescription className="text-base">
                  Análisis basado en síntomas y parámetros clínicos del paciente
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button className="w-full bg-secondary hover:bg-secondary/90" size="lg">
                  Iniciar Análisis
                </Button>
              </CardContent>
            </Card>

            <Card 
              className="cursor-pointer transition-all hover:shadow-xl hover:scale-[1.02] border-2 hover:border-primary bg-card/50 backdrop-blur"
              onClick={() => navigate("/diagnostico/imagen")}
            >
              <CardHeader className="text-center">
                <div className="mx-auto mb-4 h-16 w-16 rounded-full bg-gradient-to-br from-primary to-primary/70 flex items-center justify-center shadow-lg">
                  <Image className="h-8 w-8 text-primary-foreground" />
                </div>
                <CardTitle className="text-2xl text-primary">Detección por Imagen</CardTitle>
                <CardDescription className="text-base">
                  Análisis dermatológico mediante inteligencia artificial
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button className="w-full" size="lg">
                  Subir Imagen
                </Button>
              </CardContent>
            </Card>
          </div>

          <div className="mt-12 p-6 bg-warning/10 border-2 border-warning/30 rounded-lg">
            <h3 className="font-semibold text-lg mb-2 text-warning-foreground">⚠️ Aviso Importante</h3>
            <p className="text-sm text-foreground">
              VetSense es una herramienta de asistencia en la detección de enfermedades. Los resultados son orientativos 
              y no sustituyen el criterio clínico profesional ni un examen veterinario completo.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;