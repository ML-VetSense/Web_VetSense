import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { ArrowLeft, Loader2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import ResultadoTabular from "@/components/ResultadoTabular";

const sintomas = [
  "letargo",
  "vómitos", 
  "diarrea",
  "fiebre",
  "pérdida de apetito",
  "tos",
  "secreción ocular",
  "picazón",
  "dolor",
  "dificultad respiratoria"
];

const DiagnosticoTexto = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [resultado, setResultado] = useState<any>(null);
  
  const [nombre, setNombre] = useState("");
  const [especie, setEspecie] = useState("");
  const [edad, setEdad] = useState("");
  const [peso, setPeso] = useState("");
  const [sintomasSeleccionados, setSintomasSeleccionados] = useState<string[]>([]);
  const [temperatura, setTemperatura] = useState("");
  const [pulso, setPulso] = useState("");
  const [frecResp, setFrecResp] = useState("");

  const handleSintomaChange = (sintoma: string, checked: boolean) => {
    if (checked) {
      setSintomasSeleccionados([...sintomasSeleccionados, sintoma]);
    } else {
      setSintomasSeleccionados(sintomasSeleccionados.filter(s => s !== sintoma));
    }
  };

  const limpiar = () => {
    setNombre("");
    setEspecie("");
    setEdad("");
    setPeso("");
    setSintomasSeleccionados([]);
    setTemperatura("");
    setPulso("");
    setFrecResp("");
    setResultado(null);
  };

  const analizar = async () => {
    if (!especie || !edad || !peso || sintomasSeleccionados.length === 0) {
      toast({
        title: "Campos incompletos",
        description: "Por favor completa al menos: especie, edad, peso y un síntoma",
        variant: "destructive"
      });
      return;
    }

    setLoading(true);
    try {
      // Call local Python server directly for local development
      const response = await fetch('http://localhost:8000/predict_clinical', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          animal: {
            species: especie,
            age: parseFloat(edad),
            weight: parseFloat(peso)
          },
          symptoms: sintomasSeleccionados,
          vitals: {
            temperature: temperatura ? parseFloat(temperatura) : null,
            heart_rate: pulso ? parseFloat(pulso) : null,
            resp_rate: frecResp ? parseFloat(frecResp) : null
          }
        })
      });

      if (!response.ok) {
        throw new Error('Error al analizar los datos clínicos');
      }

      const data = await response.json();
      setResultado(data);
      
      toast({
        title: "Análisis completado",
        description: "Los resultados están listos"
      });
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "No se pudo conectar con el servidor Python. Asegúrate de que esté corriendo en http://localhost:8000",
        variant: "destructive"
      });
    } finally {
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

        <h1 className="text-4xl font-bold mb-2 text-foreground">Diagnóstico por Texto</h1>
        <p className="text-muted-foreground mb-8">Análisis clínico basado en síntomas y datos del animal</p>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Datos del Animal</CardTitle>
              <CardDescription>Información básica del paciente</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="nombre">Nombre</Label>
                  <Input 
                    id="nombre" 
                    value={nombre} 
                    onChange={(e) => setNombre(e.target.value)}
                    placeholder="Nombre del animal"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="especie">Especie *</Label>
                  <Select value={especie} onValueChange={setEspecie}>
                    <SelectTrigger>
                      <SelectValue placeholder="Selecciona especie" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Dog">Perro</SelectItem>
                      <SelectItem value="Cat">Gato</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="edad">Edad (años) *</Label>
                  <Input 
                    id="edad" 
                    type="number" 
                    value={edad} 
                    onChange={(e) => setEdad(e.target.value)}
                    placeholder="Ej: 3"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="peso">Peso (kg) *</Label>
                  <Input 
                    id="peso" 
                    type="number" 
                    value={peso} 
                    onChange={(e) => setPeso(e.target.value)}
                    placeholder="Ej: 10.5"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Signos Clínicos y Síntomas</CardTitle>
              <CardDescription>Selecciona todos los síntomas observados</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                {sintomas.map((sintoma) => (
                  <div key={sintoma} className="flex items-center space-x-2">
                    <Checkbox 
                      id={sintoma}
                      checked={sintomasSeleccionados.includes(sintoma)}
                      onCheckedChange={(checked) => handleSintomaChange(sintoma, checked as boolean)}
                    />
                    <Label htmlFor={sintoma} className="cursor-pointer capitalize">
                      {sintoma}
                    </Label>
                  </div>
                ))}
              </div>
              <div className="pt-4 border-t">
                <h4 className="font-medium mb-4">Signos Vitales (Opcional)</h4>
                <div className="grid grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="temp">Temperatura (°C)</Label>
                    <Input 
                      id="temp" 
                      type="number" 
                      value={temperatura}
                      onChange={(e) => setTemperatura(e.target.value)}
                      placeholder="38.2"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="pulso">Pulso (lpm)</Label>
                    <Input 
                      id="pulso" 
                      type="number" 
                      value={pulso}
                      onChange={(e) => setPulso(e.target.value)}
                      placeholder="95"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="freq">Frec. Resp. (rpm)</Label>
                    <Input 
                      id="freq" 
                      type="number" 
                      value={frecResp}
                      onChange={(e) => setFrecResp(e.target.value)}
                      placeholder="20"
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="flex gap-4">
            <Button 
              variant="outline" 
              onClick={limpiar}
              className="flex-1"
            >
              Limpiar
            </Button>
            <Button 
              onClick={analizar} 
              disabled={loading}
              className="flex-1"
            >
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Analizar con Modelo Tabular
            </Button>
          </div>

          {resultado && <ResultadoTabular resultado={resultado} />}
        </div>
      </div>
    </div>
  );
};

export default DiagnosticoTexto;
