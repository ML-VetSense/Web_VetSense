import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Info, HelpCircle } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface ResultadoImagenProps {
  resultado: {
    predictions: Array<{
      class: string;
      prob: number;
    }>;
    top_class: string;
    top_prob: number;
    gradcam_url?: string;
    category?: string;
    isReclassified?: boolean;
    originalSum?: number;
    originalPredictions?: Array<{
      class: string;
      prob: number;
    }>;
  };
  imagenOriginal: string;
}

const categoriaColors: Record<string, string> = {
  dermatologicas: "bg-blue-500",
  gastrointestinales: "bg-cyan-500",
  externas_no_dermatologicas: "bg-pink-500",
  internas_no_gastrointestinales: "bg-red-500",
  Desease_Detected : "bg-yellow-500",
  healthy: "bg-green-500"
};

const ResultadoImagen = ({ resultado, imagenOriginal }: ResultadoImagenProps) => {
  const categoriaColor = resultado.category ? categoriaColors[resultado.category] : "bg-gray-500";

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription>
          Este resultado es orientativo y no sustituye una consulta veterinaria profesional.
        </AlertDescription>
      </Alert>

      {resultado.isReclassified && (
        <Alert className="border-green-500 bg-green-50 dark:bg-green-950/20">
          <Info className="h-4 w-4 text-green-600 dark:text-green-400" />
          <AlertDescription className="text-green-800 dark:text-green-300 flex items-center gap-2">
            <span>
              El sistema detecta muy baja probabilidad de enfermedad (suma: {((resultado.originalSum || 0) * 100).toFixed(2)}%). 
              Clasificado como <strong>Healthy</strong> (estimado {(resultado.top_prob * 100).toFixed(1)}%).
            </span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-4 w-4 cursor-help flex-shrink-0 text-green-600 dark:text-green-400" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs p-4">
                  <div className="space-y-2">
                    <p className="font-semibold text-sm mb-2">Desglose de probabilidades detectadas:</p>
                    {resultado.originalPredictions && resultado.originalPredictions.length > 0 ? (
                      <div className="space-y-1">
                        {resultado.originalPredictions
                          .sort((a, b) => b.prob - a.prob)
                          .map((pred, idx) => (
                            <div key={idx} className="flex justify-between text-xs">
                              <span>{pred.class}</span>
                              <span className="font-mono">{(pred.prob * 100).toFixed(2)}%</span>
                            </div>
                          ))}
                      </div>
                    ) : (
                      <p className="text-xs text-muted-foreground">No hay enfermedades detectadas</p>
                    )}
                  </div>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </AlertDescription>
        </Alert>
      )}

      <Card className="border-primary/20">
        <CardHeader>
          <div className="flex justify-between items-start">
            <div>
              <CardTitle className="text-2xl">Diagnóstico Principal</CardTitle>
              <CardDescription>Predicción más probable según el modelo</CardDescription>
            </div>
            {resultado.category && (
              <Badge className={`${categoriaColor} text-white capitalize`}>
                {resultado.category.replace(/_/g, ' ')}
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <h3 className="text-xl font-semibold text-primary">{resultado.top_class}</h3>
              <span className="text-2xl font-bold">{(resultado.top_prob * 100).toFixed(1)}%</span>
            </div>
            <Progress value={resultado.top_prob * 100} className="h-3" />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Top 5 Diagnósticos Posibles</CardTitle>
          <CardDescription>Probabilidades de otras condiciones consideradas</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {resultado.predictions.slice(0, 5).map((pred, idx) => (
              <div key={idx} className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="font-medium">{pred.class}</span>
                  <span className="text-muted-foreground">{(pred.prob * 100).toFixed(1)}%</span>
                </div>
                <Progress value={pred.prob * 100} className="h-2" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Análisis Visual</CardTitle>
          <CardDescription>Imagen original y mapa de atención (Grad-CAM)</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm font-medium mb-2">Imagen Original</p>
              <img 
                src={imagenOriginal} 
                alt="Original" 
                className="w-full h-auto rounded-lg border"
              />
            </div>
            <div>
              <p className="text-sm font-medium mb-2">Grad-CAM (Zonas de Interés)</p>
              {resultado.gradcam_url ? (
                <img 
                  src={resultado.gradcam_url} 
                  alt="Grad-CAM" 
                  className="w-full h-auto rounded-lg border"
                />
              ) : (
                <div className="w-full aspect-square bg-muted rounded-lg flex items-center justify-center">
                  <p className="text-muted-foreground text-sm">Mapa de atención no disponible</p>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ResultadoImagen;
