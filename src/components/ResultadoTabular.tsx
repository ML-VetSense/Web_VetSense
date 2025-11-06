import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Info } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface ResultadoTabularProps {
  resultado: {
    predictions: Array<{
      class: string;
      prob: number;
    }>;
    top_class: string;
    top_prob: number;
    explanations: {
      features: string[];
      method: string;
    };
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

      <Card className="border-primary/20">
        <CardHeader>
          <CardTitle className="text-2xl">Diagnóstico Principal</CardTitle>
          <CardDescription>Predicción más probable según el modelo</CardDescription>
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
          <CardDescription>Probabilidades de otras enfermedades consideradas</CardDescription>
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
          <CardTitle>Explicación del Análisis</CardTitle>
          <CardDescription>Variables más relevantes en el diagnóstico</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground mb-3">
              Método de explicación: <span className="font-semibold uppercase">{resultado.explanations.method}</span>
            </p>
            <div className="flex flex-wrap gap-2">
              {resultado.explanations.features.map((feature, idx) => (
                <div
                  key={idx}
                  className="px-3 py-1.5 bg-primary/10 text-primary rounded-full text-sm font-medium"
                >
                  {feature}
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ResultadoTabular;
