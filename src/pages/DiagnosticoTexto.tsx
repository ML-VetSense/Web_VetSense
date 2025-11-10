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

// Estos campos coinciden exactamente con las columnas del modelo
interface ClinicalData {
  Animal_Type: string;
  Breed: string;
  Age: number;
  Gender: string;
  Weight: number;
  Duration: string;
  Appetite_Loss: number;
  Vomiting: number;
  Diarrhea: number;
  Lethargy: number;
  Coughing: number;
  Sneezing: number;
  Skin_Lesion_Present: number;
  Skin_Lesion_Type: string;
  Skin_Lesion_Location: string;
  Hair_Loss: number;
  Itching_Scratching: number;
  Lesion_Color: string;
  Nasal_Discharge: number;
  Eye_Discharge: number;
  Eye_Redness: number;
  Ear_Discharge: number;
  Head_Shaking: number;
  Bad_Breath: number;
  Drooling: number;
  Difficulty_Eating: number;
  Frequent_Urination: number;
  Blood_in_Urine: number;
  Straining_to_Urinate: number;
  Increased_Thirst: number;
  Weakness: number;
  Body_Temperature: number | null;
  Heart_Rate: number | null;
  Respiratory_Rate: number | null;
}

const DiagnosticoTexto = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [resultado, setResultado] = useState<any>(null);
  
  // Datos básicos
  const [animalType, setAnimalType] = useState("");
  const [breed, setBreed] = useState("");
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [weight, setWeight] = useState("");
  const [duration, setDuration] = useState("");
  
  // Síntomas generales (0 o 1)
  const [appetiteLoss, setAppetiteLoss] = useState(false);
  const [vomiting, setVomiting] = useState(false);
  const [diarrhea, setDiarrhea] = useState(false);
  const [lethargy, setLethargy] = useState(false);
  const [coughing, setCoughing] = useState(false);
  const [sneezing, setSneezing] = useState(false);
  const [weakness, setWeakness] = useState(false);
  
  // Síntomas de piel
  const [skinLesionPresent, setSkinLesionPresent] = useState(false);
  const [skinLesionType, setSkinLesionType] = useState("");
  const [skinLesionLocation, setSkinLesionLocation] = useState("");
  const [hairLoss, setHairLoss] = useState(false);
  const [itchingScratching, setItchingScratching] = useState(false);
  const [lesionColor, setLesionColor] = useState("");
  
  // Síntomas específicos
  const [nasalDischarge, setNasalDischarge] = useState(false);
  const [eyeDischarge, setEyeDischarge] = useState(false);
  const [eyeRedness, setEyeRedness] = useState(false);
  const [earDischarge, setEarDischarge] = useState(false);
  const [headShaking, setHeadShaking] = useState(false);
  const [badBreath, setBadBreath] = useState(false);
  const [drooling, setDrooling] = useState(false);
  const [difficultyEating, setDifficultyEating] = useState(false);
  
  // Síntomas urinarios
  const [frequentUrination, setFrequentUrination] = useState(false);
  const [bloodInUrine, setBloodInUrine] = useState(false);
  const [strainingToUrinate, setStrainingToUrinate] = useState(false);
  const [increasedThirst, setIncreasedThirst] = useState(false);
  
  // Signos vitales
  const [bodyTemperature, setBodyTemperature] = useState("");
  const [heartRate, setHeartRate] = useState("");
  const [respiratoryRate, setRespiratoryRate] = useState("");

  const limpiar = () => {
    setAnimalType("");
    setBreed("");
    setAge("");
    setGender("");
    setWeight("");
    setDuration("");
    setAppetiteLoss(false);
    setVomiting(false);
    setDiarrhea(false);
    setLethargy(false);
    setCoughing(false);
    setSneezing(false);
    setWeakness(false);
    setSkinLesionPresent(false);
    setSkinLesionType("");
    setSkinLesionLocation("");
    setHairLoss(false);
    setItchingScratching(false);
    setLesionColor("");
    setNasalDischarge(false);
    setEyeDischarge(false);
    setEyeRedness(false);
    setEarDischarge(false);
    setHeadShaking(false);
    setBadBreath(false);
    setDrooling(false);
    setDifficultyEating(false);
    setFrequentUrination(false);
    setBloodInUrine(false);
    setStrainingToUrinate(false);
    setIncreasedThirst(false);
    setBodyTemperature("");
    setHeartRate("");
    setRespiratoryRate("");
    setResultado(null);
  };

  const analizar = async () => {
    if (!animalType || !age || !weight) {
      toast({
        title: "Campos incompletos",
        description: "Por favor completa al menos: tipo de animal, edad y peso",
        variant: "destructive"
      });
      return;
    }

    setLoading(true);
    try {
      // Construir el objeto con las columnas exactas del modelo
      const clinicalData: ClinicalData = {
        Animal_Type: animalType,
        Breed: breed || "Unknown",
        Age: parseFloat(age),
        Gender: gender || "Unknown",
        Weight: parseFloat(weight),
        Duration: duration || "Unknown",
        Appetite_Loss: appetiteLoss ? 1 : 0,
        Vomiting: vomiting ? 1 : 0,
        Diarrhea: diarrhea ? 1 : 0,
        Lethargy: lethargy ? 1 : 0,
        Coughing: coughing ? 1 : 0,
        Sneezing: sneezing ? 1 : 0,
        Skin_Lesion_Present: skinLesionPresent ? 1 : 0,
        Skin_Lesion_Type: skinLesionType || "None",
        Skin_Lesion_Location: skinLesionLocation || "None",
        Hair_Loss: hairLoss ? 1 : 0,
        Itching_Scratching: itchingScratching ? 1 : 0,
        Lesion_Color: lesionColor || "None",
        Nasal_Discharge: nasalDischarge ? 1 : 0,
        Eye_Discharge: eyeDischarge ? 1 : 0,
        Eye_Redness: eyeRedness ? 1 : 0,
        Ear_Discharge: earDischarge ? 1 : 0,
        Head_Shaking: headShaking ? 1 : 0,
        Bad_Breath: badBreath ? 1 : 0,
        Drooling: drooling ? 1 : 0,
        Difficulty_Eating: difficultyEating ? 1 : 0,
        Frequent_Urination: frequentUrination ? 1 : 0,
        Blood_in_Urine: bloodInUrine ? 1 : 0,
        Straining_to_Urinate: strainingToUrinate ? 1 : 0,
        Increased_Thirst: increasedThirst ? 1 : 0,
        Weakness: weakness ? 1 : 0,
        Body_Temperature: bodyTemperature ? parseFloat(bodyTemperature) : null,
        Heart_Rate: heartRate ? parseFloat(heartRate) : null,
        Respiratory_Rate: respiratoryRate ? parseFloat(respiratoryRate) : null
      };

      const response = await fetch('http://localhost:8000/predict_clinical', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(clinicalData)
      });

      if (!response.ok) {
        throw new Error('Error al analizar los datos clínicos');
      }

      const data = await response.json();
      
      // Aplicar la misma lógica de "Healthy" que en imágenes
      const nonHealthyPredictions = data.predictions.filter((p: any) => 
        !p.class.toLowerCase().includes('healthy') && !p.class.toLowerCase().includes('normal')
      );
      
      const sumNonHealthy = nonHealthyPredictions.reduce((sum: number, p: any) => sum + p.prob, 0);
      
      if (sumNonHealthy < 0.10) {
        const healthyProb = 1 - sumNonHealthy;
        setResultado({
          ...data,
          predictions: [
            { class: "Healthy", prob: healthyProb },
            ...nonHealthyPredictions.filter((p: any) => p.prob >= 0.01)
          ],
          top_class: "Healthy",
          top_prob: healthyProb,
          isReclassified: true,
          originalSum: sumNonHealthy,
          originalPredictions: nonHealthyPredictions
        });
      } else {
        setResultado(data);
      }
      
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
                  <Label htmlFor="animalType">Tipo de Animal *</Label>
                  <Select value={animalType} onValueChange={setAnimalType}>
                    <SelectTrigger>
                      <SelectValue placeholder="Selecciona tipo" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Dog">Perro</SelectItem>
                      <SelectItem value="Cat">Gato</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="breed">Raza</Label>
                  <Input 
                    id="breed" 
                    value={breed} 
                    onChange={(e) => setBreed(e.target.value)}
                    placeholder="Ej: Labrador"
                  />
                </div>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="age">Edad (años) *</Label>
                  <Input 
                    id="age" 
                    type="number" 
                    value={age} 
                    onChange={(e) => setAge(e.target.value)}
                    placeholder="Ej: 3"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="gender">Género</Label>
                  <Select value={gender} onValueChange={setGender}>
                    <SelectTrigger>
                      <SelectValue placeholder="Selecciona" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Male">Macho</SelectItem>
                      <SelectItem value="Female">Hembra</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="weight">Peso (kg) *</Label>
                  <Input 
                    id="weight" 
                    type="number" 
                    value={weight} 
                    onChange={(e) => setWeight(e.target.value)}
                    placeholder="Ej: 10.5"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="duration">Duración de síntomas (días)</Label>
                <Input 
                  id="duration"  
                  value={duration} 
                  onChange={(e) => setDuration(e.target.value)}
                  placeholder="Ej: 3d"
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Síntomas Generales</CardTitle>
              <CardDescription>Selecciona todos los síntomas que presenta</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div className="flex items-center space-x-2">
                  <Checkbox id="appetiteLoss" checked={appetiteLoss} onCheckedChange={(c) => setAppetiteLoss(c as boolean)} />
                  <Label htmlFor="appetiteLoss" className="cursor-pointer">Pérdida de apetito</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="vomiting" checked={vomiting} onCheckedChange={(c) => setVomiting(c as boolean)} />
                  <Label htmlFor="vomiting" className="cursor-pointer">Vómitos</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="diarrhea" checked={diarrhea} onCheckedChange={(c) => setDiarrhea(c as boolean)} />
                  <Label htmlFor="diarrhea" className="cursor-pointer">Diarrea</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="lethargy" checked={lethargy} onCheckedChange={(c) => setLethargy(c as boolean)} />
                  <Label htmlFor="lethargy" className="cursor-pointer">Letargo</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="coughing" checked={coughing} onCheckedChange={(c) => setCoughing(c as boolean)} />
                  <Label htmlFor="coughing" className="cursor-pointer">Tos</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="sneezing" checked={sneezing} onCheckedChange={(c) => setSneezing(c as boolean)} />
                  <Label htmlFor="sneezing" className="cursor-pointer">Estornudos</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="weakness" checked={weakness} onCheckedChange={(c) => setWeakness(c as boolean)} />
                  <Label htmlFor="weakness" className="cursor-pointer">Debilidad</Label>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Síntomas de Piel</CardTitle>
              <CardDescription>Información sobre lesiones cutáneas</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center space-x-2">
                  <Checkbox id="skinLesion" checked={skinLesionPresent} onCheckedChange={(c) => setSkinLesionPresent(c as boolean)} />
                  <Label htmlFor="skinLesion" className="cursor-pointer">Lesión en piel presente</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="hairLoss" checked={hairLoss} onCheckedChange={(c) => setHairLoss(c as boolean)} />
                  <Label htmlFor="hairLoss" className="cursor-pointer">Pérdida de pelo</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="itching" checked={itchingScratching} onCheckedChange={(c) => setItchingScratching(c as boolean)} />
                  <Label htmlFor="itching" className="cursor-pointer">Picazón/Rascado</Label>
                </div>
              </div>
              {skinLesionPresent && (
                <div className="grid grid-cols-3 gap-4 pt-4 border-t">
                  <div className="space-y-2">
                    <Label htmlFor="lesionType">Tipo de lesión</Label>
                    <Input 
                      id="lesionType" 
                      value={skinLesionType} 
                      onChange={(e) => setSkinLesionType(e.target.value)}
                      placeholder="Ej: Pápula, Úlcera"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lesionLocation">Ubicación</Label>
                    <Input 
                      id="lesionLocation" 
                      value={skinLesionLocation} 
                      onChange={(e) => setSkinLesionLocation(e.target.value)}
                      placeholder="Ej: Abdomen, Patas"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lesionColor">Color</Label>
                    <Input 
                      id="lesionColor" 
                      value={lesionColor} 
                      onChange={(e) => setLesionColor(e.target.value)}
                      placeholder="Ej: Rojo, Marrón"
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Otros Síntomas</CardTitle>
              <CardDescription>Síntomas adicionales específicos</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div className="flex items-center space-x-2">
                  <Checkbox id="nasalDischarge" checked={nasalDischarge} onCheckedChange={(c) => setNasalDischarge(c as boolean)} />
                  <Label htmlFor="nasalDischarge" className="cursor-pointer">Secreción nasal</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="eyeDischarge" checked={eyeDischarge} onCheckedChange={(c) => setEyeDischarge(c as boolean)} />
                  <Label htmlFor="eyeDischarge" className="cursor-pointer">Secreción ocular</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="eyeRedness" checked={eyeRedness} onCheckedChange={(c) => setEyeRedness(c as boolean)} />
                  <Label htmlFor="eyeRedness" className="cursor-pointer">Enrojecimiento ocular</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="earDischarge" checked={earDischarge} onCheckedChange={(c) => setEarDischarge(c as boolean)} />
                  <Label htmlFor="earDischarge" className="cursor-pointer">Secreción del oído</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="headShaking" checked={headShaking} onCheckedChange={(c) => setHeadShaking(c as boolean)} />
                  <Label htmlFor="headShaking" className="cursor-pointer">Sacudida de cabeza</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="badBreath" checked={badBreath} onCheckedChange={(c) => setBadBreath(c as boolean)} />
                  <Label htmlFor="badBreath" className="cursor-pointer">Mal aliento</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="drooling" checked={drooling} onCheckedChange={(c) => setDrooling(c as boolean)} />
                  <Label htmlFor="drooling" className="cursor-pointer">Babeo excesivo</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="difficultyEating" checked={difficultyEating} onCheckedChange={(c) => setDifficultyEating(c as boolean)} />
                  <Label htmlFor="difficultyEating" className="cursor-pointer">Dificultad al comer</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="frequentUrination" checked={frequentUrination} onCheckedChange={(c) => setFrequentUrination(c as boolean)} />
                  <Label htmlFor="frequentUrination" className="cursor-pointer">Micción frecuente</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="bloodInUrine" checked={bloodInUrine} onCheckedChange={(c) => setBloodInUrine(c as boolean)} />
                  <Label htmlFor="bloodInUrine" className="cursor-pointer">Sangre en orina</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="strainingToUrinate" checked={strainingToUrinate} onCheckedChange={(c) => setStrainingToUrinate(c as boolean)} />
                  <Label htmlFor="strainingToUrinate" className="cursor-pointer">Esfuerzo al orinar</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="increasedThirst" checked={increasedThirst} onCheckedChange={(c) => setIncreasedThirst(c as boolean)} />
                  <Label htmlFor="increasedThirst" className="cursor-pointer">Sed aumentada</Label>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Signos Vitales</CardTitle>
              <CardDescription>Opcional - Mediciones clínicas</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="bodyTemp">Temperatura (°C)</Label>
                  <Input 
                    id="bodyTemp" 
                    type="number" 
                    value={bodyTemperature}
                    onChange={(e) => setBodyTemperature(e.target.value)}
                    placeholder="38.2"
                    step="0.1"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="heartRate">Frecuencia cardíaca (lpm)</Label>
                  <Input 
                    id="heartRate" 
                    type="number" 
                    value={heartRate}
                    onChange={(e) => setHeartRate(e.target.value)}
                    placeholder="95"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="respRate">Frec. Respiratoria (rpm)</Label>
                  <Input 
                    id="respRate" 
                    type="number" 
                    value={respiratoryRate}
                    onChange={(e) => setRespiratoryRate(e.target.value)}
                    placeholder="20"
                  />
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