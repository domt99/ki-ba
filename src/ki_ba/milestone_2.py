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
    -> Start Gewichtsmatrizen erzeugen 
    -> Eine Gewichtsmatrix pro Übergang zwischen zwei Schichten
    -> Maße der Gewichtsmatrix jeweils m * n
    -> Mit m = Anzahl der Neuronen von Schicht A
    -> Mit n = Anzahl der Neuronen von Schicht A+1

	Hyperparameter:
	-> Anzahl der Epochen
	-> Batch-Size (abhängig von Trainingsart)
    -> Anzahl der Schichten (flexibel anpassbar)
	-> Anzahl der Neuronen je Schicht (flexibel anpassbar)
    
    Loss-Funktion:
	-> Bestimmt den Fehler
    -> Verlustfunktion (z.B. CCE)
    
    Backpropagation:
    -> Speicherung von t (Schrittzähler), m, v, 
    -> Ableitungen berechnen
    -> Weitergabe von t (Schrittzähler), m, v an z.B. Adam
    -> Adam gibt neues m & v zurück
    
    Gewichte ändern:
    -> Aus Ergebnis von Backpropagation
    -> Trainingsart z.B. Adam berechnet die neuen Gewichte
    -> Trainingsprozess (Framework) passt Gewichtsmatrizen an

Netz (Architektur):

	Input-Layer
	-> Input: Aufbereitete Daten
	-> Anzahl Input-Neuronen
	-> Gewichte
    -> Bias

	Hidden-Layer
	-> Übertragungsfunktion
	-> Aktivierungsfunktion
    -> Gewichte
	-> Bias
    -> Weiterverarbeitung der Ausgaben aus vorheriger Schicht
    -> Berechnungen zur Aufbereitung der Informationen sodass 
    Output Schicht eine Entscheidung treffen kann 

	Output-Layer
    -> Übertragungsfunktion
	-> Aktivierungsfunktion (z.B. Softmax)
    -> Gewichte
	-> Bias
    -> Rechnet wie Hidden Layer
    -> Trifft Entscheidungen auf Basis der Berechnungen
    -> Aufbereitung durch Aktivierungsfunktion (z.B. Softmax)
    -> Vorhersage als Output
    
Prozess (Ablaufkette):

"""