# Steam Reviews Analysis

HIER KOMMT NOCH TEXT AUS DER PRÄSI ZUSAMMENGEFASST

## Projektstruktur

hier kommt noch text aus der PRÄSI

## Dateien und Ordner

### Notebooks

- **0_Data_Preparation.ipynb**:

  - **Zweck**: Datenvorbereitung
  - **Inhalt**:
    - Laden der Rohdaten aus `data/source_data/`
    - Bereinigung und Filterung der Daten
    - Speicherung der verarbeiteten Daten in `data/processed_data/`
- **0_First_EDA.ipynb**:

  - **Zweck**: Erste explorative Datenanalyse (EDA)
  - **Inhalt**:
    - Untersuchung der Verteilung der Bewertungen
    - Analyse der häufigsten Wörter und Phrasen
    - Visualisierung der Daten
- **1_Emotion_Model.ipynb**:

  - **Zweck**: Modellierung und Analyse von Emotionen
  - **Inhalt**:
    - Verwendung von vortrainierten Modellen zur Emotionserkennung
    - Anwendung des Modells auf die gefilterten Daten
    - Speicherung der Ergebnisse in `data/emotions_results/`
- **1_Sarcasm_Models.ipynb**:

  - **Zweck**: Modellierung und Analyse von Sarkasmus
  - **Inhalt**:
    - Verwendung verschiedener Modelle zur Sarkasmuserkennung (z.B. T5, Helinivan, SDCR)
    - Anwendung der Modelle auf die gefilterten Daten
    - Speicherung der Ergebnisse in `data/sarcasm_results/`
- **2_DistilBERT_Model.ipynb**:

  - **Zweck**: Verwendung des DistilBERT-Modells für die Analyse
  - **Inhalt**:
    - Feinabstimmung des DistilBERT-Modells auf die Steam Reviews
    - Bewertung der Modellleistung
    - Anwendung des Modells auf die gefilterten Daten
    - Experimente mit verschiedenen Datengrundlagen
    - Speicherung der Ergebnisse in `data/sentiment_results/`
    - Speicherung des Modells für zukünftige Verwendung unter `models/`

### Datenordner

- **data/emotions_results/**: Enthält die Ergebnisse der Emotionsanalyse.

  - `steam_reviews_filtered_emotions.parquet`: Parquet-Datei mit den gefilterten Emotionsergebnissen.
- **data/processed_data/**: Enthält die verarbeiteten Daten.

  - `steam_reviews_filtered.parquet`: Parquet-Datei mit den gefilterten Steam Reviews.
- **data/sarcasm_results/**: Enthält die Ergebnisse der Sarkasmusanalyse.

  - `sarcasm_predictions_t5.parquet`: Parquet-Datei mit den Sarkasmusvorhersagen des T5-Modells.
  - `sarcasm_scores_helinivan.parquet`: Parquet-Datei mit den Sarkasmusscores des Helinivan-Modells.
  - `sarcasm_scores_sdcr.parquet`: Parquet-Datei mit den Sarkasmusscores des SDCR-Modells.
- **data/source_data/**: Enthält die Rohdaten.

  - `game_info.parquet`: Parquet-Datei mit Informationen zu den Spielen. (API Daten)
  - `steam_reviews_2020.parquet`: Parquet-Datei mit Steam Reviews aus dem Jahr 2020. (Gesäuberte Daten)
  - `steam_reviews.parquet`: Parquet-Datei mit allen Steam Reviews. (Orginaldaten aus Kaggle)

## Anforderungen

- CUDA-fähige GPU (optional)
- Python 3.x
- PyTorch
- Transformers
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Nutzung

1. Die Notebooks sind in der Reihenfolge ihrer Nummerierung auszuführen, beginnend mit `0_First_EDA.ipynb` und endend mit `2_DistilBERT_Model.ipynb`.
2. Dabei ist die Reihenfolge der Notebooks wichtig, da die Ergebnisse der Nummerierungen hierarchisch aufeinander aufbauen, wenn es um die Datenverarbeitung, -analyse und modellierung geht.
3. Die Ergebnisse werden in den entsprechenden Ordnern im `data`-Verzeichnis gespeichert.
4. Modelle werden im `models`-Verzeichnis gespeichert.
5. Die Ergebnisse können dann in den Notebooks oder externen Tools weiter analysiert und visualisiert werden.

## Autoren

- Markus Grau, Daniel Kosma, Mikail Yildiz
