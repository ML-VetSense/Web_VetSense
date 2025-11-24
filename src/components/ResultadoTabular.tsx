import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Info, HelpCircle } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface ResultadoTabularProps {
  resultado: {
    predictions: Array<{
      class: string;
      prob: number;
    }>;
    top_class: string;
    top_prob: number;
    explanations?: {
      features?: string[];
      method?: string;
    };
    isReclassified?: boolean;
    originalSum?: number;
    originalPredictions?: Array<{
      class: string;
      prob: number;
    }>;
  };
}

const ResultadoTabular = ({ resultado }: ResultadoTabularProps) => {
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

      <Card className="border-primary/20">
        <CardHeader className="bg-primary/5">
          <CardTitle className="text-2xl text-primary">Condición Identificada</CardTitle>
          <CardDescription>Predicción más probable según los datos clínicos</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <h3 className="text-xl font-semibold text-primary">{resultado.top_class}</h3>
              <span className="text-2xl font-bold text-primary">{(resultado.top_prob * 100).toFixed(1)}%</span>
            </div>
            <Progress value={resultado.top_prob * 100} className="h-3" />
          </div>
        </CardContent>
      </Card>

      <Card className="border-primary/20">
        <CardHeader className="bg-primary/5">
          <CardTitle className="text-primary">Top 5 Condiciones Posibles</CardTitle>
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

      <Card className="border-primary/20">
        <CardHeader className="bg-primary/5">
          <CardTitle className="text-primary">Explicación del Análisis</CardTitle>
          <CardDescription>Variables clínicas más relevantes en la detección</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {resultado.explanations && (
              <>
                <p className="text-sm text-muted-foreground mb-3">
                  Método de explicación: <span className="font-semibold uppercase">{resultado.explanations.method}</span>
                </p>
                <div className="flex flex-wrap gap-2">
                  {resultado.explanations.features?.map((feature, idx) => (
                    <div
                      key={idx}
                      className="px-3 py-1.5 bg-primary/10 text-primary rounded-full text-sm font-medium"
                    >
                      {feature}
                    </div>
                  ))}
                </div>
              </>
            )}
            {!resultado.explanations && (
              <p className="text-sm text-muted-foreground">
                No hay explicaciones disponibles para este diagnóstico.
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ResultadoTabular;