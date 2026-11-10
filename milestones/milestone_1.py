import numpy as np

def adam_step(t, g, m, v, w_old, alpha=0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
    
    """

    Adam-Schritt

    Im Adam-Verfahren wird ueber die Epochen und die jeweiligen Mini-Batches in den Epochen
    iteriert. Diese Iteration ruft fuer jeden Schritt t die Funktion adam_step() auf,
    um jeweils die Gewichte im numpy.Array w_new zu berechnen. Um den naechsten Schritt im Verfahren
    berechnen zu koennen, benoetigt das Modell die neuen Werte fuer m und v, da fuer jedes Gewicht
    jeweils ein eigenes m und v berechnet und gespeichert
    (adam_step gibt die Werte an das Modell zurueck) wird.

    Die Startwerte fuer m und v sind 0, da es fuer t = 1 keine vergangenen Gradienten gibt.

    Fuer t gilt: 1 <= t <= Anzahl der Mini-Batches pro Epoche * Anzahl der Epochen
    Der Startwert fuer t ist 0, wird aber vor jedem Schritt hochgezaehlt.
    t wird global gespeichert/hochgezaehlt

    Hyperparameter:
        alpha (Standardwert: 0.001) beschreibt die Lernrate
        beta_1 (Standardwert: 0.9) beschreibt, wie viel von dem alten m-Wert behalten wird
        beta_2 (Standardwert: 0.999) beschreibt, wie viel von dem alten v-Wert behalten wird
        epsilon (Standardwert: 1e-8) verhindert eine Division durch 0.

    Parameter:
        t: int - Schrittnummer
        g: numpy.array mit float Werten - Gradient: Summe der Ableitungen ueber den Batch geteilt
            durch die Batchgroesse
        m: numpy.array mit float Werten - Array mit den gewichteten Mittelwerten der vergangenen
            Gradienten
        v: numpy.array mit float Werten - Array mit den gewichteten Mittelwerten der vergangenen
            Gradienten^2
        w_old: numpy.array mit float Werten - Array mit Gewichten aus dem vorherigen Schritt
        --> Die Arrays m, v, g und w_old muessen dieselbe Form haben

    Rueckgabewerte:
        w_new: numpy.array mit float Werten - Array mit den neuen Gewichten
        m_new: numpy.array mit float Werten - Array mit den gewichteten Werten aus m und g
        v_new: numpy.array mit float Werten - Array mit den gewichteten Werten aus v und g**2

    """

    m_new = beta_1 * m + (1 - beta_1) * g
    v_new = beta_2 * v + (1 - beta_2) * g**2

    m_corrected = m_new / (1 - beta_1**t)
    v_corrected = v_new / (1 - beta_2**t)

    w_new = w_old - alpha * m_corrected / (np.sqrt(v_corrected) + epsilon)

    return w_new, m_new, v_new