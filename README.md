# Steam Reviews Analysis

Dieses Projekt analysiert Steam-Bewertungen und wendet dabei verschiedene NLP-Modelle an, um Emotionen, Sarkasmus und Empfehlungen zu identifizieren. Ziel ist es, anhand von Benutzerbewertungen Rückschlüsse auf die Spielerfahrung und Spielqualität zu ziehen.

## Projektstruktur

Das Projekt ist in verschiedene Bereiche aufgeteilt, darunter Datenvorbereitung, Modellierung von Emotionen, Sarkasmus und Vorhersagemodelle.

## Dateien und Ordner

### Notebooks

- **00_First_EDA.ipynb**:
  - **Zweck**: Erste explorative Datenanalyse (EDA)
  - **Inhalt**:
    - Untersuchung der Verteilung der Bewertungen
    - Analyse der häufigsten Wörter und Phrasen
    - Visualisierung der Daten

- **01_Data_Preparation.ipynb**:
  - **Zweck**: Datenvorbereitung
  - **Inhalt**:
    - Laden der Rohdaten aus `data/source_data/`
    - Bereinigung und Filterung der Daten
    - Speicherung der verarbeiteten Daten in `data/processed_data/`

- **10_Emotion_Model.ipynb**:
  - **Zweck**: Modellierung und Analyse von Emotionen
  - **Inhalt**:
    - Verwendung von vortrainierten Modellen zur Emotionserkennung
    - Anwendung des Modells auf die gefilterten Daten
    - Speicherung der Ergebnisse in `data/emotions_results/`

- **11_Sarcasm_Models.ipynb**:
  - **Zweck**: Modellierung und Analyse von Sarkasmus
  - **Inhalt**:
    - Verwendung verschiedener Modelle zur Sarkasmuserkennung (z.B. T5, Helinivan, SDCR)
    - Anwendung der Modelle auf die gefilterten Daten
    - Speicherung der Ergebnisse in `data/sarcasm_results/`

- **20_DistilBERT_Model.ipynb**:
  - **Zweck**: Verwendung des DistilBERT-Modells für die Analyse
  - **Inhalt**:
    - Feinabstimmung des DistilBERT-Modells auf die Steam Reviews
    - Bewertung der Modellleistung
    - Anwendung des Modells auf die gefilterten Daten
    - Experimente mit verschiedenen Datengrundlagen
    - Speicherung der Ergebnisse in `data/sentiment_results/`
    - Speicherung des Modells für zukünftige Verwendung unter `models/`

- **21_Supervised_Models.ipynb**:
  - **Zweck**: Supervised Learning Modelle
  - **Inhalt**:
    - Training von Modellen auf den Steam-Daten
    - Vergleich der Modellleistung

### Datenordner

- **data/emotions_results/**: Enthält die Ergebnisse der Emotionsanalyse.
  - `steam_reviews_filtered_emotions.parquet`: Parquet-Datei mit den gefilterten Emotionsergebnissen.

- **data/processed_data/**: Enthält die verarbeiteten Daten.
  - `steam_reviews_filtered.parquet`: Parquet-Datei mit den gefilterten Steam Reviews.
  - `steam_reviews_with_emotions_full_predictions.parquet`: Datei mit vollständigen Emotionsergebnissen.

- **data/sarcasm_results/**: Enthält die Ergebnisse der Sarkasmusanalyse.
  - `sarcasm_predictions_t5.parquet`: Parquet-Datei mit den Sarkasmusvorhersagen des T5-Modells.
  - `sarcasm_scores_helinivan.parquet`: Parquet-Datei mit den Sarkasmusscores des Helinivan-Modells.
  - `sarcasm_scores_sdcr.parquet`: Parquet-Datei mit den Sarkasmusscores des SDCR-Modells.

- **data/source_data/**: Enthält die Rohdaten.
  - `game_info.parquet`: Parquet-Datei mit Informationen zu den Spielen.
  - `steam_reviews_2020.parquet`: Parquet-Datei mit Steam Reviews aus dem Jahr 2020.
  - `steam_reviews.parquet`: Parquet-Datei mit allen Steam Reviews.

- **data/word2vec_data/**: Enthält vorverarbeitete Word2Vec-Daten.
  - `word2vec_review_vector_CBOW_300.parquet`: Word2Vec Review Vektoren (CBOW, 300 Dimensionen).
  - `word2vec_review_vector_SG_100.parquet`: Word2Vec Review Vektoren (Skip-Gram, 100 Dimensionen).
  - `word2vec_review_vector_SG_300.parquet`: Word2Vec Review Vektoren (Skip-Gram, 300 Dimensionen).

## Anforderungen

- CUDA-fähige GPU (optional - Die Version muss hier bei Bedarf angepasst werden!)
- Python 3.x (Das Projekt wurde mit Python 3.11.8 erstellt - diese Version ist empfohlen.)
- PyTorch
- Transformers
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

Alle benötigten libraries können mit `pip install -r requirements.txt` installiert werden.

## Nutzung

1. Die Notebooks sind in der Reihenfolge ihrer Nummerierung auszuführen, beginnend mit `00_First_EDA.ipynb` und endend mit `20_DistilBERT_Model.ipynb` für die Transformer bzw. `21_Supervised_Models.ipynb` für die Supervised Methoden.
2. Dabei ist die Reihenfolge der Notebooks wichtig, da die Ergebnisse der Nummerierungen hierarchisch aufeinander aufbauen, wenn es um die Datenverarbeitung, -analyse und Modellierung geht.
3. Die Ergebnisse werden in den entsprechenden Ordnern im `data`-Verzeichnis gespeichert.
4. Modelle werden im `models`-Verzeichnis gespeichert.
5. Die Ergebnisse können dann in den Notebooks oder externen Tools weiter analysiert und visualisiert werden.

## Autoren

- Markus Grau, Daniel Kosma, Mikail Yildiz