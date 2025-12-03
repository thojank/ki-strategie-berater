# KI-Strategie-Berater

Der **KI-Strategie-Berater** ist eine kleine RAG-Anwendung (Retrieval-Augmented Generation) auf Basis von:

- 🧠 einer Wissensdatenbank in **Postgres / Supabase**  
- 📚 Vektor-Suche mit **pgvector**  
- 💬 einer **Streamlit**-UI als Frontend  
- 🤖 einem LLM (z. B. OpenAI) für Antworten auf natürliche Sprache

Ziel:  
Interne Strategiepapiere, Folien, Notizen und Architekturbeschreibungen so aufzubereiten, dass du sie
über eine Chat-Oberfläche gezielt abfragen kannst (z. B. „Wie ist unsere 3-Schicht-Architektur aufgebaut?“,
„Welche Usecases haben wir im Aftersales priorisiert?“).

---

## 1. Projektstruktur

> **Hinweis:** Die genauen Dateinamen können bei dir leicht abweichen. Falls du etwas umbenennst, bitte hier anpassen.

Typische Struktur:

```text
ki-strategie-berater/
├─ app.py                # Streamlit-App (UI + Query-Logik)
├─ db.py                 # DB-Verbindung + Helper-Funktionen (optional)
├─ ingest.py             # Ingestion-Script für Dokumente (optional)
├─ requirements.txt      # Python-Abhängigkeiten
├─ .env                  # Lokale Umgebungsvariablen (NICHT ins Git!)
└─ README.md             # (diese Datei)
```

Wenn du weitere Module hast (`utils/`, `models/`, `config/` etc.), bitte bei Bedarf ergänzen.

---

## 2. Architekturüberblick

### 2.1 Komponenten

- **Postgres / Supabase**
  - Speichert:
    - Roh-Dokumente (Metadaten)
    - Text-Chunks mit Embeddings
    - optional: Graph-Struktur (Nodes/Edges) für Themen-Beziehungen

- **Embedding-Service**
  - z. B. OpenAI-Embeddings (`text-embedding-3-small` o. ä.)
  - wird beim Ingest verwendet, um Vektoren zu berechnen

- **LLM**
  - z. B. OpenAI Chat-Modell (`gpt-4.1` / `gpt-4o` / `gpt-5.1-mini`)
  - bekommt:
    - User-Frage
    - relevante Chunks (aus dem Vektor-Search)
    - Prompt (z. B. „Antworte knapp, in Deutsch, mit Quellenangabe“)

- **Streamlit-UI (`app.py`)**
  - Eingabefeld für Fragen
  - Anzeige der Antwort
  - Anzeige der genutzten Dokument-Snippets (Quellen)
  - evtl. Filter (z. B. Themenbereich, Jahr, Dokumenttyp)

---

## 3. Datenbankschema (Supabase / Postgres)

Das Schema ist anpassbar. So ist es typischerweise gedacht:

### 3.1 Tabelle `kb_documents`

Metadaten pro Dokument.

```sql
CREATE TABLE IF NOT EXISTS public.kb_documents (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  title          text,
  source_path    text,         -- Pfad/Dateiname
  doc_type       text,         -- z.B. 'pptx', 'pdf', 'md'
  tags           text[],       -- Themen-Tags
  created_at     timestamptz DEFAULT now()
);
```

### 3.2 Tabelle `kb_chunks`

Text-Chunks + Embeddings.

```sql
CREATE TABLE IF NOT EXISTS public.kb_chunks (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id        uuid REFERENCES kb_documents(id) ON DELETE CASCADE,
  chunk_index        int,          -- Reihenfolge im Dokument
  chunk_text         text,
  embedding_openai   vector,       -- pgvector-Spalte
  created_at         timestamptz DEFAULT now()
);
```

Index für Vektor-Suche:

```sql
CREATE INDEX IF NOT EXISTS kb_chunks_embedding_idx
ON public.kb_chunks
USING ivfflat (embedding_openai vector_cosine_ops)
WITH (lists = 100);
```

### 3.3 (Optional) Tabellen für Wissensgraph

Falls du ein Themen-/Begriffsnetz modellierst:

```sql
CREATE TABLE IF NOT EXISTS public.ki_strat_nodes (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name        text,
  node_type   text,         -- z.B. 'UseCase', 'Capability', 'System'
  description text
);

CREATE TABLE IF NOT EXISTS public.ki_strat_edges (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_id   uuid REFERENCES ki_strat_nodes(id) ON DELETE CASCADE,
  target_id   uuid REFERENCES ki_strat_nodes(id) ON DELETE CASCADE,
  edge_type   text           -- z.B. 'depends_on', 'implements', 'related_to'
);
```

Die Graph-Struktur kann in der UI z. B. über Filter oder Visualisierungen genutzt werden.

---

## 4. Installation & Setup

### 4.1 Voraussetzungen

- Python ≥ 3.10
- Zugriff auf eine **Supabase**-Instanz (oder eigenes Postgres mit pgvector)
- OpenAI-Account (oder anderer LLM/Embedding-Anbieter)
- `git` (optional)

### 4.2 Projekt clonen / lokal bereitstellen

```bash
git clone <DEIN-REPO-URL>
cd ki-strategie-berater
```

Oder den Projektordner einfach lokal hinlegen.

### 4.3 Virtuelle Umgebung & Abhängigkeiten

```bash
# venv erstellen (optional, aber empfohlen)
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# Dependencies installieren
pip install -r requirements.txt
```

Stelle sicher, dass `requirements.txt` mindestens enthält:

```txt
streamlit
openai
psycopg2-binary
python-dotenv
pgvector            # falls direkt genutzt
```

(Bei Bedarf anpassen.)

---

## 5. Konfiguration (.env)

Lege im Projektverzeichnis eine Datei `.env` an (NICHT in Git einchecken):

Beispiel:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Supabase / Postgres
DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<db_name>
# oder z.B.
SUPABASE_DB_USER=...
SUPABASE_DB_PASSWORD=...
SUPABASE_DB_HOST=...
SUPABASE_DB_PORT=5432
SUPABASE_DB_NAME=...

# Sonstige Einstellungen
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4.1-mini
```

In `app.py` / `db.py` werden diese Variablen mit `python-dotenv` geladen.

---

## 6. Dokumente einlesen (Ingestion)

> Falls du ein eigenes `ingest.py` oder ähnliche Scripts hast, bitte hier konkretisieren.  
> Unten steht ein typischer Ablauf, an dem du dich orientieren kannst.

### 6.1 Typischer Ingest-Ablauf

1. **Dokumentquelle definieren**  
   - Ordner mit PDFs, PPTX, Markdown, …  
2. **Dokumente extrahieren**  
   - Text pro Dokument lesen  
3. **Chunking**  
   - Text in Abschnitte (z. B. 500–1000 Token) zerlegen  
4. **Embeddings berechnen**  
   - `embedding = openai_client.embeddings.create(...)`  
5. **In `kb_documents` & `kb_chunks` schreiben**  

Pseudo-Beispiel (verkürzt):

```python
from openai import OpenAI
import psycopg2
import textwrap

client = OpenAI()

def ingest_document(path: str, conn):
    text = open(path).read()
    chunks = textwrap.wrap(text, 1000)  # sehr simplifiziert

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO kb_documents (title, source_path) VALUES (%s, %s) RETURNING id",
            ("Strategie-Dokument XY", path)
        )
        doc_id = cur.fetchone()[0]

        for idx, chunk in enumerate(chunks):
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            ).data[0].embedding

            cur.execute(
                \"\"\"
                INSERT INTO kb_chunks (document_id, chunk_index, chunk_text, embedding_openai)
                VALUES (%s, %s, %s, %s)
                \"\"\",
                (doc_id, idx, chunk, emb)
            )
    conn.commit()
```

---

## 7. Streamlit-App starten

Sobald:

- `.env` korrekt ist  
- DB-Schema existiert  
- und mindestens ein paar Dokumente als Chunks in der DB liegen,

kannst du die App starten:

```bash
streamlit run app.py
```

Standardmäßig öffnet sich die UI im Browser (z. B. `http://localhost:8501`).

---

## 8. Funktionsweise der UI (app.py)

> Hier die typische Logik – bitte auf deinen konkreten Code mappen.

1. **User gibt eine Frage ein**  
   z. B. „Welche Ebenen hat unser Agentic-State-Modell?“

2. **App erzeugt Embedding der Frage**
   ```python
   q_emb = client.embeddings.create(
       model=EMBEDDING_MODEL,
       input=user_question
   ).data[0].embedding
   ```

3. **Vektor-Suche in `kb_chunks`**
   ```sql
   SELECT
     c.chunk_text,
     d.title,
     d.source_path,
     1 - (c.embedding_openai <=> %s::vector) AS similarity
   FROM kb_chunks c
   JOIN kb_documents d ON d.id = c.document_id
   ORDER BY c.embedding_openai <=> %s::vector
   LIMIT 5;
   ```

4. **Antwort mit Kontext generieren**
   - Die Top-N Chunks werden als Kontext in den Prompt gepackt.
   - LLM generiert eine Antwort (Deutsch, Quellen am Ende).

5. **Anzeige in Streamlit**
   - Antwort im Hauptbereich  
   - Darunter Liste der verwendeten Quellen (Titel + ggf. Pfad)  
   - Optional: Debug-Infos, Ähnlichkeitswerte

---

## 9. Entwicklung & Erweiterung

Mögliche Erweiterungen:

- 🔎 **Filter** nach:
  - Thema (Tag), Jahr, Dokumenttyp
- 🌐 **Mehrere Wissensräume**
  - z. B. `domain`-Spalte (Aftersales, Agentic AI, Compliance, …)
- 🌲 **Graph-Integration**
  - Anzeige von `ki_strat_nodes` und ihren Verknüpfungen zu Dokumenten
- 👥 **User-Sessions**
  - Chatverlauf pro User speichern (z. B. in eigener Tabelle)

---

## 10. Sicherheit & Betrieb

- `.env` **niemals** ins Git committen  
- Bei Supabase:
  - RLS-Regeln (Row-Level Security) prüfen
  - ggf. Service-Role-Key nur im Backend verwenden, nicht im Frontend
- Rate-Limits & Kosten der LLM-API im Blick behalten

---

## 11. Known Issues / ToDo (Beispiele)

- [ ] Edge-Cases beim Chunking (Tabellen, Bullet-Listen, Code-Blöcke)  
- [ ] Bessere Prompt-Templates für Antworten (konkrete Struktur, Tonfall, Sprache)  
- [ ] UI-Verbesserungen (History, Dark-Mode, Export als Markdown/PDF)  

Passe diese Liste einfach an deinen tatsächlichen Stand an.

---

## 12. Lizenz / Nutzung

> Falls du das Projekt veröffentlichen möchtest, hier eine Lizenz angeben (z. B. MIT).  
> Wenn intern: kurze Notiz, wer es nutzen darf und wer Maintainer ist.

```text
© <Jahr> <Dein Name>. Nur für internen Gebrauch.
```
