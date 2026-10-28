"""

Meilenstein 2:
a)

Datenaufbereitung

-> Daten durch 255 Teilen um Skala 0-1 zu bekommen (einmalig)

-> Daten-Mischung (Daten randomisiert für jede Epoche)

Trainingsprozess (Framework):
	
	Trainingsart:
	-> Adam oder Rprop
    
    Initialisierung:
    -> Schrittzähler t wird einmal mit 0 initialisiert und zählt 
       selbstständig hoch über die Vollbatches / Minibatches 
       und Epochen
    -> Schwellwerte werden in Gewichtsmatrix gespeichert (Zeile 0)
    -> Maße der Matrizen für m, v, w und g mit den Maßen M+1*N
    -> Mit M = Anzahl der Neuronen von Schicht A
        -> +1 wegen extra Zeile der festen Eingabe -1
        -> Diese extra Zeile enthält die Schwellwerte θ der 
           N Neuronen aus Schicht A+1
    -> Mit N = Anzahl der Neuronen von Schicht A+1
    -> Anzahl der Matrizen ist Anzahl der Übergänge (i) zwischen Schichten
       im Netz
    -> Initialisierung der Gewichtsmatrix
        -> Gewichte mit Standardabweichung √(2/M) (Zeilen 1-M) 
        -> Eine Zeile mit Schwellwerten θ mit Nullen
    -> Initialisierung von m und v als Null Matrizen
    -> Keine Initialisierung für g vor dem Training da in jedem Batch 
       Backpropagation es neu berechnet
            -> g muss für jeden Mini-Batch als Null Matrix bereitgestellt
               werden (Nur für die Schleifenversion)
    -> Festlegung der Aktivierungsfunktionen pro Schicht

	Hyperparameter: 
	-> Anzahl der Epochen
	-> Batch-Size (abhängig von Trainingsart)
    -> Anzahl der Schichten (flexibel anpassbar)
	-> Anzahl der Neuronen je Schicht (flexibel anpassbar)
    
    Gewichte ändern (z.B. Adam):
    -> Bekommt t (Schrittzähler) übergeben, sowie die aktuellen
       m und v
    -> Aus Ergebnis von Backpropagation mit Ableitungen (Gradienten)
       aus Backpropagation
    -> Trainingsart z.B. Adam berechnet die neuen Gewichte
    -> Trainingsart z.B. Adam passt die Gewichte an
    -> Berechnet neue m und v und gibt diese zurück

Netz (Architektur):
	Input-Layer
	-> Input: Aufbereitete Daten
	-> Anzahl Input-Neuronen

	Hidden-Layer
    Forwardpropagation:
    -> Die -1 an die Eingabe (in Zeile 0 der jeweiligen vorherigen Eingabe) 
    -> Berechne Übertragungsfunktion netj = SUMME(wij * oi) i ∈ prec(j ) 
       mit Gewicht w0j = θj mit konstantem Eingang o0=-1
    -> Anwendung der Aktivierungsfunktion (Sigmoid, ReLu, Identität) 
       pro Neuronenübergang
    -> Ausgabe der Neuronen an nächste Schicht weitergeben
    -> Gewichte
	-> Bias
    -> Weiterverarbeitung der Ausgaben aus vorheriger Schicht
    -> Berechnungen zur Aufbereitung der Informationen sodass 
    Output Schicht eine Entscheidung treffen kann 
    -> merkt sich: net pro Neuron und ihre erweiterte Eingabe (mit -1)

    Backpropagation:
    -> Bekommt herein: die Schuld ihrer Ausgaben, von der Schicht danach.
    -> Berechnet ihr δ: Schuld mal Ableitung der Aktivierung.
    -> Berechnet ihre Gradienten G: erweiterte Eingabe mal δ.
    -> Gibt weiter: die Schuld für die Schicht davor, mit den Zeilen 1 bis M ihrer Gewichtsmatrix, ohne Zeile 0.
    -> Die Backpropagation hört mit dem ersten Hidden Layer nach der Input Schicht auf -> die Input Schicht berechnet nichts
    -> Für jedes Gewicht und jeden Bias wird mithilfe
       der Kettenregel berechnet, wie stark es zum 
       Gesamtfehler beiträgt
    -> Nach vollständigem Ablauf der Backpropagation, kann
       durch Übergabe der Gradienten mit Adam weitergearbeitet 
       werden

	Output-Layer
    Forwardpropagation:
    -> Rechnet wie Hidden Layer
    -> Trifft Entscheidungen auf Basis der Berechnungen
    -> Aufbereitung durch Aktivierungsfunktion über gesamte Ausgabeschicht (z.B. Softmax)
    -> Vorhersage als Output
    -> merkt sich: Vorhersage (o) und Ihre erweiterte Eingabe (mit -1)
    
    Lossfunktion:
    -> Erhält vom Framework die Zielwerte t
    -> Berechnung von Loss (CCE)
    -> Aus der gemeinsamen Ableitung aus Softmax und CCE wird das Fehlersignal δ = o − t 
       für jedes Output Neuron berechnet
            -> Ist der Startwert für die Backpropagation

    Backpropagation:
    -> Berechnet ihre Gradienten G: erweiterte Eingabe mal δ.
    -> Gibt weiter: die Schuld für die Schicht davor, mit den Zeilen 1 bis M ihrer Gewichtsmatrix, ohne Zeile 0.
    -> Für jedes Gewicht und jeden Bias wird mithilfe
       der Kettenregel berechnet, wie stark es zum 
       Gesamtfehler beiträgt
    -> Nach vollständigem Ablauf der Backpropagation, kann
       durch Übergabe der Gradienten mit Adam weitergearbeitet 
       werden

Prozess (Ablaufkette):

Komponenten:
1. Datenaufbereitung
2. Trainingsprozess (Framework)
3. Netz
4. Trainingsart (Rprop- oder Adam)
5. Input Layer
6. Hidden Layer
7. Ouput Layer
8. Aktivierungsfunktion


"""