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
    -> Schwellwerte werden in Gewichtsmatrix gespeichert
    -> Maße der Matrizen für m, v, w und g mit den Maßen M+1*N
    -> Mit M = Anzahl der Neuronen von Schicht A
        -> +1 wegen extra Zeile der Schwellwerte
    -> Mit N = Anzahl der Neuronen von Schicht A+1
    -> Anzahl der Matrizen ist Anzahl der Übergänge (i) zwischen Schichten
       im Netz
    -> Initialisierung der Gewichtsmatrix
        -> Gewichte mit Standardabweichung √(2/M) (Zeilen M) 
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
    
    Loss-Funktion:
	-> Bestimmt den Fehler
    -> Verlustfunktion (z.B. CCE)
    
    Backpropagation:
    -> δ = o − t für jedes Output Neuron berechnen
    -> Berechnet die Ableitungen (Gradienten)
    -> Für jedes Gewicht und jeden Bias wird mithilfe
       der Kettenregel berechnet, wie stark es zum 
       Gesamtfehler beiträgt
    -> Nach vollständigem Ablauf der Backpropagation, kann
       durch Übergabe der Gradienten mit Adam weitergearbeitet 
       werden
    
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
	-> Berechne Übertragungsfunktion netj = SUMME(wij * oi) i ∈ prec(j ) 
       mit Gewicht w0j = θj
	-> Anwendung der Aktivierungsfunktion (Sigmoid, ReLu, Identität) 
       pro Neuronenübergang
    -> Gewichte
	-> Bias
    -> Weiterverarbeitung der Ausgaben aus vorheriger Schicht
    -> Berechnungen zur Aufbereitung der Informationen sodass 
    Output Schicht eine Entscheidung treffen kann 

	Output-Layer
	-> Berechne Übertragungsfunktion netj = SUMME(wij * oi) i ∈ prec(j ) 
       mit Gewicht w0j = θj
	-> Anwendung der Aktivierungsfunktion (Softmax) 
       pro Outputneuron
    -> Gewichte
	-> Bias
    -> Rechnet wie Hidden Layer
    -> Trifft Entscheidungen auf Basis der Berechnungen
    -> Aufbereitung durch Aktivierungsfunktion (z.B. Softmax)
    -> Vorhersage als Output
    
Prozess (Ablaufkette):

"""