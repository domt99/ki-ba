"""

Meilenstein 2:
a)

Datenaufbereitung

-> Daten durch 255 Teilen um Skala 0-1 zu bekommen (einmalig)

-> Daten-Mischung (Daten randomisiert für jede Epoche)

Trainingsprozess (Framework):
	
	Trainingsart:
	-> Adam oder Rprop

	Hyperparameter:
	-> Anzahl der Epochen
	-> Batch-Size (abhängig von Trainingsart)

	Loss-Funktion:
	-> 

Netz (Architektur):

	Input-Layer
	-> Input: Aufbereitete Daten 
	-> Anzahl Input-Neuronen
	-> Gewichte

	Hidden-Layer
	-> Anzahl der Schichten (flexibel anpassbar)
	-> Anzahl der Neuronen je Schicht (flexibel anpassbar)
	-> Gewichte
	-> Bias
	-> Übertragungsfunktion
	-> Aktivierungsfunktion

	Output-Layer
	-> Vorhersage als Output
	-> Aktivierungsfunktion (z.B. Softmax, CCP)

"""