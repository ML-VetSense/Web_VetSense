import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Info, HelpCircle, ChevronDown } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useState } from "react";

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
  animalType?: 'dog' | 'cat' | null;
}

const categoriaColors: Record<string, string> = {
  dermatologicas: "bg-blue-500",
  gastrointestinales: "bg-cyan-500",
  externas_no_dermatologicas: "bg-pink-500",
  internas_no_gastrointestinales: "bg-red-500",
  healthy: "bg-green-500"
};

const ResultadoImagen = ({ resultado, imagenOriginal, animalType }: ResultadoImagenProps) => {
  const categoriaColor = resultado.category ? categoriaColors[resultado.category] : "bg-gray-500";
  const [detallesAbiertos, setDetallesAbiertos] = useState(false);

  const getRecomendacion = () => {
    if (resultado.isReclassified) {
      return "Tu mascota parece estar saludable. Continúa con los cuidados habituales y mantén las visitas regulares al veterinario.";
    }
    if (resultado.top_prob > 0.7) {
      return "Se recomienda consultar con un veterinario lo antes posible para confirmar el diagnóstico y recibir tratamiento adecuado.";
    }
    if (resultado.top_prob > 0.4) {
      return "Se detectaron algunos signos que podrían requerir atención. Considera programar una visita veterinaria.";
    }
    return "Los resultados no son concluyentes. Si notas síntomas persistentes, consulta con un veterinario.";
  };

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription>
          Este resultado es orientativo y no sustituye una consulta veterinaria profesional.
        </AlertDescription>
      </Alert>

      {resultado.isReclassified && (
        <Alert className="border-success bg-success/10">
          <Info className="h-4 w-4 text-success" />
          <AlertDescription className="text-success-foreground flex items-center gap-2">
            <span>
              El sistema detecta muy baja probabilidad de enfermedad (suma: {((resultado.originalSum || 0) * 100).toFixed(2)}%). 
              Clasificado como <strong>Healthy</strong> (estimado {(resultado.top_prob * 100).toFixed(1)}%).
            </span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-4 w-4 cursor-help flex-shrink-0 text-success" />
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

      <Card className={`${resultado.isReclassified ? 'border-success' : 'border-primary/20'}`}>
        <CardHeader className="bg-primary/5">
          <div className="flex justify-between items-start">
            <div>
              <CardTitle className="text-3xl text-primary">
                {resultado.isReclassified ? '✅ Tu mascota está saludable' : `⚠️ ${resultado.top_class}`}
              </CardTitle>
              <CardDescription className="text-base mt-2">
                {animalType && `Análisis para ${animalType === 'dog' ? 'perro' : 'gato'}`}
              </CardDescription>
            </div>
            {resultado.category && !resultado.isReclassified && (
              <Badge className={`${categoriaColor} text-white capitalize`}>
                {resultado.category.replace(/_/g, ' ')}
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-3">
            <div className="flex justify-between items-baseline">
              <h3 className="text-xl font-semibold text-muted-foreground">Nivel de confianza</h3>
              <span className="text-3xl font-bold text-primary">
                {(resultado.top_prob * 100).toFixed(1)}%
              </span>
            </div>
            <Progress value={resultado.top_prob * 100} className="h-4" />
          </div>

          <Alert className={resultado.isReclassified ? 'bg-success/10 border-success' : 'bg-accent border-primary'}>
            <Info className={`h-4 w-4 ${resultado.isReclassified ? 'text-success' : 'text-primary'}`} />
            <AlertDescription className={resultado.isReclassified ? 'text-success-foreground' : 'text-accent-foreground'}>
              <strong>Recomendación:</strong> {getRecomendacion()}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>

      <Collapsible open={detallesAbiertos} onOpenChange={setDetallesAbiertos}>
        <Card>
          <CollapsibleTrigger asChild>
            <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Detalles Técnicos Avanzados</CardTitle>
                  <CardDescription>Métricas y predicciones detalladas del modelo</CardDescription>
                </div>
                <ChevronDown className={`h-5 w-5 transition-transform ${detallesAbiertos ? 'rotate-180' : ''}`} />
              </div>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent className="space-y-6 pt-0">
              <div>
                <h4 className="font-semibold mb-3">Top 5 Predicciones</h4>
                <div className="space-y-4">
                  {resultado.predictions.slice(0, 5).map((pred, idx) => (
                    <div key={idx} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="font-medium">{pred.class}</span>
                        <span className="text-muted-foreground font-mono">{(pred.prob * 100).toFixed(2)}%</span>
                      </div>
                      <Progress value={pred.prob * 100} className="h-2" />
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="font-semibold mb-3">Análisis Visual</h4>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground mb-2">Imagen Original</p>
                    <img 
                      src={imagenOriginal} 
                      alt="Original" 
                      className="w-full h-auto rounded-lg border"
                    />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground mb-2">Mapa de Atención (Grad-CAM)</p>
                    {resultado.gradcam_url ? (
                      <img 
                        src={resultado.gradcam_url} 
                        alt="Grad-CAM" 
                        className="w-full h-auto rounded-lg border"
                      />
                    ) : (
                      <div className="w-full aspect-square bg-muted rounded-lg flex items-center justify-center border">
                        <p className="text-muted-foreground text-sm text-center px-4">Mapa de atención no disponible</p>
                      </div>
                    )}
                  </div>
                </div>
                <p className="text-xs text-muted-foreground mt-3">
                  Las zonas más cálidas (rojas/amarillas) indican las áreas donde el modelo centró su atención para el diagnóstico.
                </p>
              </div>

              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription className="text-xs">
                  <strong>Nota técnica:</strong> Los valores mostrados representan probabilidades calculadas por el modelo de deep learning. 
                  La confianza del diagnóstico depende de la calidad de la imagen y la claridad de los síntomas visibles.
                </AlertDescription>
              </Alert>
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>
    </div>
  );
};

export default ResultadoImagen;