# app.py
# Streamlit RAG (VERSION 31.5: Branding-Update)
from __future__ import annotations

import os, re, json
from typing import Any, Dict, List, Tuple
from decimal import Decimal
import contextlib 

import streamlit as st
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit_antd_components as sac 

# OpenAI SDK (>=1.x/2.x)
try:
    from openai import OpenAI
    _OPENAI_MODE = "sdk_v1"
except Exception:
    import openai
    _OPENAI_MODE = "legacy"

# ------------------------------------------------------------------------------
# Konfiguration 
# ------------------------------------------------------------------------------
load_dotenv(override=True)

KB_TABLE   = os.getenv("KB_TABLE", "public.kb_chunks") 
EMBED_COL  = os.getenv("EMBED_COL", "embedding_openai")
TEXT_COLS  = [c.strip() for c in os.getenv("TEXT_COLS", "chunk_text, source_filename").split(",")]
CHUNK_TEXT_COL = "chunk_text"
FILENAME_COL = "source_filename"
CHUNK_ID_COL = "id" # Wichtig für Graph-Verknüpfung

DEFAULT_CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

SUPABASE_PG_CONN = os.getenv("SUPABASE_PG_CONN") or os.getenv("DATABASE_URL")
if not SUPABASE_PG_CONN:
    raise RuntimeError("Fehlt: SUPABASE_PG_CONN oder DATABASE_URL (.env)")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Fehlt: OPENAI_API_KEY (.env)")

if _OPENAI_MODE == "sdk_v1":
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai.api_key = OPENAI_API_KEY
    client = openai

# System Prompt für den Chat
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """
DEINE WICHTIGSTE REGEL: Du darfst unter KEINEN UMSTÄNDEN Wissen außerhalb des bereitgestellten KONTEXTS verwenden.
- Wenn der KONTEXT leer ist oder die Frage nicht beantworten kann, MUSST du mit exakt diesem Satz antworten: "Ich konnte diese Information nicht in meiner Wissensdatenbank finden." Es gibt keine Ausnahmen.
- Wenn KONTEXT vorhanden ist: Beantworte die Frage des Nutzers präzise und ausschließlich auf Basis dieses KONTEXTS.
- Antworte auf Deutsch.
- Zitiere am Ende deiner Antwort die Quellen (filename) aus dem Kontext, falls der Kontext genutzt wurde.
- Bei allgemeinen Fragen (z.B. 'Hallo') antworte höflich (dies ist die EINZIGE Ausnahme, bei der du ohne Kontext antworten darfst).
- ERINNERUNG: Wenn die Frage spezifisch ist und der KONTEXT leer ist, lautet die ANTWORT IMMER: "Ich konnte diese Information nicht in meiner Wissensdatenbank finden."
""")

# System-Prompt für den Berater (ANGEPASST FÜR AKQUISE)
CONFIGURATOR_SYSTEM_PROMPT = """
Du bist ein Experte für KI-Strategieberatung, spezialisiert auf KMUs und Mittelstand.
Du führst ein Beratungsgespräch.
Deine Aufgabe ist es, auf Basis des bereitgestellten KONTEXTS (der Best Practices, Methoden und Roadmaps enthält) eine maßgeschneiderte Empfehlung zu generieren.
Auch bei Folgefragen ("Erzähl mir mehr zu...") nutzt du den gesamten Gesprächsverlauf UND den NEU gefundenen KONTEXT.

- ES IST ABSOLUT VERBOTEN, WISSEN AUSSERHALB DES KONTEXTS ZU VERWENDEN ODER ZU HALLUZINIEREN.
- Halte dich strikt an den KONTEXT. Wenn der KONTEXT leer ist oder die Frage nicht beantworten kann, MUSST du mit exakt diesem Satz antworten: "Ich konnte diese Information nicht in meiner Wissensdatenbank finden."
- Wenn du eine erste Empfehlung (basierend auf dem Formular) erstellst UND KONTEXT vorhanden ist, strukturiere sie so:
    1.  **Zusammenfassung:** Wiederhole kurz die Situation des Nutzers (Branche, Größe, Ziele).
    2.  **Empfohlene Handlungsfelder:** Leite basierend auf Zielen und Kontext die wichtigsten Handlungsfelder ab.
    3.  **Best Practices & Methoden:** Nenne konkrete Tipps, Tricks und Methoden aus dem KONTEXT, die zur Situation (insb. KMU) passen.
    4.  **Implementierungs-Roadmap (Entwurf):** Skizziere die nächsten Schritte für die Implementierung.

- SEI HILFREICH, aber fasse dich kurz. Deine Antwort ist eine *automatisierte Erstanalyse*. Der Nutzer weiß, dass der nächste Schritt ein persönliches Gespräch ist.
"""

# --- NEU: System-Prompt für Entitäts-Extraktion ---
ENTITY_EXTRACTOR_PROMPT = """
Du bist ein Experte für semantische Suche. Extrahiere die 2-3 wichtigsten Substantive oder Konzepte (Entitäten) aus der folgenden Nutzerfrage.
- Gib NUR eine JSON-Liste von Strings zurück.
- Beispiel: "Wie erstelle ich eine KI-Roadmap für mein KMU?"
- Ausgabe: ["KI-Roadmap", "KMU"]

Nutzerfrage:
{prompt}

JSON-Liste:
"""


# ------------------------------------------------------------------------------
# Streamlit UI-Setup
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="KI-Strategie Berater",
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("KI-Strategie Berater") 
# HINWEIS: Logo wird jetzt in der Sidebar gesetzt

# --- CSS-HACK (Layout-Änderungen) ---
st.markdown(f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.10.5/font/bootstrap-icons.min.css">
    
    <style>
    /* 1. Main Content Area (hellgrau & abgesetzt) */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }}
    div[data-testid="stVerticalBlock"] > div:not([data-testid="stSidebar"]) > div.block-container {{
         background-color: #FAFAFA; /* Hellgrau */
         border-left: 1px solid #DDDDDD; /* Trennlinie */
    }}
    
    
    /* 2. Formular-Styling (unverändert) */
    [data-testid="stForm"] {{
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }}

    /* 3. Input-Felder Styling (unverändert) */
    [data-testid="stSelectbox"] > div {{
        border: 1px solid #DDDDDD !important; 
        border-radius: 0.25rem;
    }}
    [data-testid="stMultiSelect"] {{
        border: 1px solid #DDDDDD !important;
        border-radius: 0.25rem;
    }}
    [data-testid="stTextInput"] > div > div > input,
    [data-testid="stTextArea"] > div > textarea {{
        border: 1px solid #DDDDDD !important;
        border-radius: 0.25rem !important;
        padding-left: 0.5rem; 
    }}
    
    /* 4. Tab-Styling (JETZT MIT NEUER SCHRIFTART) */
    .sac-tabs-bar {{
        border-bottom: 2px solid #DDDDDD !important; 
    }}
    .sac-tabs-item {{
        font-family: 'Lexend Exa', sans-serif !important; /* NEUE SCHRIFTART */
        font-weight: 700 !important;
        background-color: #EAEAEA !important; 
        border-radius: 0.25rem 0.25rem 0 0 !important; 
        margin-bottom: -2px !important; 
        border: 1px solid #DDDDDD !important; 
        border-bottom: none !important;
    }}
    
    /* 5. Aktiver Tab (Farbe wird jetzt über Python gesteuert) */
    .sac-tabs-item-active {{
        font-family: 'Lexend Exa', sans-serif !important; /* NEUE SCHRIFTART */
        font-weight: 700 !important;
        color: #FFFFFF !important; /* Weiße Schrift (Kontrast zu Rot) */
        border: 2px solid #ea3323 !important;  
        border-bottom: 2px solid #FAFAFA !important; /* Schneidet Linie mit BG-Farbe */
    }}

    /* 6. Suchschlitz-Rand (KORRIGIERT FÜR FOKUS-FORM) */
    div[data-testid="stChatInput"] {{
        border: 1px solid #ea3323 !important;
        border-radius: 0.5rem !important; /* Runde Ecken */
        background-color: #FFFFFF;
    }}
    div[data-testid="stChatInput"] div[data-baseweb="input"] {{
         border: none !important;
         box-shadow: none !important;
         background-color: transparent !important;
    }}
    div[data-testid="stChatInput"]:focus-within {{
        border-color: #ea3323 !important;
        box-shadow: 0 0 0 2px #ea332333 !important; /* Heller roter Schatten */
    }}
    div[data-testid="stChatInput"] div[data-baseweb="input"]:focus-within {{
        box-shadow: none !important;
    }}
    </style>
    """, unsafe_allow_html=True)
# --- ENDE CSS-HACK ---


# ------------------------------------------------------------------------------
# Hilfsfunktionen (DB-Connect, Embedding)
# ------------------------------------------------------------------------------
@contextlib.contextmanager
def connect_db():
    """ 
    Stellt eine Standardverbindung her. Wird jetzt für ALLES verwendet.
    """
    conn = None 
    try:
        conn = psycopg2.connect(SUPABASE_PG_CONN)
        yield conn 
    except Exception as e:
        st.error(f"DB-Verbindungsfehler: {e}")
        if conn:
            conn.close() 
        st.stop()
    finally:
        if conn:
            conn.close() 

@st.cache_data(show_spinner="[OpenAI] Erstelle Vektor-Embedding für Suchanfrage...")
def get_embedding(text_to_embed: str, model: str = DEFAULT_EMBED_MODEL) -> List[float]:
    text_to_embed = text_to_embed.strip()
    if not text_to_embed: return []
    try:
        if _OPENAI_MODE == "sdk_v1":
            res = client.embeddings.create(model=model, input=[text_to_embed])
            return res.data[0].embedding
        else:
            res = client.Embedding.create(model=model, input=[text_to_embed])
            return res["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Fehler bei OpenAI Embedding API: {e}")
        st.stop()

# --- NEU: Hilfsfunktion für Entitäts-Extraktion (Unverändert) ---
@st.cache_data(show_spinner="[OpenAI] Extrahiere Entitäten...")
def extract_entities_from_prompt(prompt: str, model: str) -> List[str]:
    messages = [
        {"role": "system", "content": ENTITY_EXTRACTOR_PROMPT.format(prompt=prompt)}
    ]
    try:
        if _OPENAI_MODE == "sdk_v1":
            res = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
            data = res.choices[0].message.content
        else:
            res = client.ChatCompletion.create(model=model, messages=messages, temperature=0.0)
            data = res['choices'][0]['message']['content']
        
        entities = json.loads(data)
        
        with st.sidebar.expander("Entitäts-Extraktion (Graph-Debug)", expanded=True):
            st.write("**Original-Frage:**", prompt)
            st.write("**Extrahierte Entitäten:**", entities)
            
        return entities
    except Exception as e:
        st.warning(f"Fehler bei Entitäts-Extraktion: {e}. Nutze leere Liste.")
        return []

# --- KORREKTUR: Graph-Abfrage (sucht jetzt Source UND Target) ---
@st.cache_data(show_spinner="[Graph] Suche relevante Chunks...")
def query_graph_for_chunks(entities: List[str]) -> List[str]:
    """ 
    Frägt die SQL-Graph-Tabellen nach Chunks ab.
    Gibt eine Liste von Chunk-IDs zurück.
    """
    if not entities:
        return []
        
    chunk_ids = set()
    
    query = f"""
    SELECT DISTINCT e.chunk_id
    FROM public.ki_strat_edges AS e
    JOIN public.ki_strat_nodes AS n ON (e.source_node_id = n.id OR e.target_node_id = n.id)
    WHERE n.name = ANY(%s);
    """

    try:
        with connect_db() as conn: 
            with conn.cursor() as cur:
                cur.execute(query, (entities,))
                results = cur.fetchall()
                for row in results:
                    chunk_ids.add(str(row[0])) 
                    
        with st.sidebar.expander("Graph-Suchtreffer (Debug)", expanded=False):
            st.write(f"Graph lieferte {len(chunk_ids)} Chunk-IDs:", chunk_ids)
            
        return list(chunk_ids)
        
    except Exception as e:
        st.warning(f"Fehler bei SQL-Graph-Abfrage: {e}")
        return []

# --- NEU: Funktion zum Finden verwandter Themen ---
@st.cache_data(show_spinner="[Graph] Finde verwandte Themen...")
def get_related_topics_from_graph(entities: List[str]) -> List[str]:
    """
    Findet Knoten, die mit den extrahierten Entitäten verbunden sind,
    um "Best Practices" oder "Verwandte Themen" vorzuschlagen.
    """
    if not entities:
        return []
    
    query = """
    SELECT DISTINCT n2.name
    FROM public.ki_strat_nodes AS n1
    JOIN public.ki_strat_edges AS e ON (n1.id = e.source_node_id OR n1.id = e.target_node_id)
    JOIN public.ki_strat_nodes AS n2 ON (n2.id = e.target_node_id OR n2.id = e.source_node_id)
    WHERE 
        n1.name = ANY(%s) 
        AND n2.name != n1.name
        AND n2.name NOT IN (SELECT unnest(%s))
    LIMIT 5;
    """
    
    try:
        with connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (entities, entities)) # Liste wird 2x übergeben
                results = cur.fetchall()
                related_topics = [row[0] for row in results]
                return related_topics
    except Exception as e:
        st.warning(f"Fehler bei Graph-Themen-Suche: {e}")
        return []


def show_hits(hits: List[Dict]):
    with st.expander("⬇️ Gefundene Treffer (Debug-Ansicht)", expanded=False):
        if not hits:
            st.info("Keine Treffer gefunden.")
            return
        data = []
        for i, hit in enumerate(hits):
            data.append({
                "Nr": i + 1,
                "Score": f"{hit.get('rank_score', 0.0):.4f}", 
                "Filename": hit.get('filename', 'N/A'),
                "Text": hit.get('text_content', hit.get('content', 'N/A'))[:500] + "..."
            })
        st.dataframe(data, use_container_width=True)


def parse_query(q: str) -> Tuple[str, List[str], str | None]:
    q_clean = q
    must_term = None
    match = re.search(r"muss:(\w+)", q_clean, re.IGNORECASE)
    if match:
        must_term = match.group(1).strip()
        q_clean = q_clean.replace(match.group(0), "").strip()
    terms = re.split(r'\s+', q_clean)
    terms = [re.sub(r'\W+', '', t).lower() for t in terms]
    stopwords = [
        "a", "ab", "aber", "als", "am", "an", "auch", "auf", "aus", "bei", "bin", 
        "bis", "bist", "da", "dadurch", "daher", "darum", "das", "dass", "dein", 
        "deine", "dem", "den", "denn", "der", "des", "die", "dir", "dich", "doch", 
        "dort", "du", "durch", "ein", "eine", "einem", "einen", "einer", "eines", 
        "er", "es", "euer", "eure", "für", "hat", "hatte", "hatten", "hier", "hin", 
        "ich", "ihr", "ihre", "im", "in", "ist", "ja", "jede", "jedem", "jeden", 
        "jeder", "jedes", "jener", "jenes", "jetzt", "kann", "kannst", "können", 
        "könnt", "machen", "mein", "meine", "mit", "muss", "müssen", "müsst", 
        "nach", "nicht", "nun", "nur", "ob", "oder", "ohne", "seid", "sein", 
        "seine", "sich", "sie", "sind", "soll", "sollen", "sollst", "sollt", 
        "sonst", "um", "und", "uns", "unser", "unsere", "unter", "vom", "von", 
        "vor", "wann", "war", "waren", "warst", "warum", "was", "weg", "weil", 
        "weiter", "welche", "welchem", "welchen", "welcher", "welches", "wenn", 
        "wer", "wie", "wieder", "wir", "wird", "wirst", "wo", "zu", "zum", "zur"
    ]
    terms = [t for t in terms if t and len(t) > 2 and t not in stopwords]
    return q_clean, terms, must_term


# ------------------------------------------------------------------------------
# Kern-Suchfunktionen (FÜR CHUNKS)
# ------------------------------------------------------------------------------

def match_documents(query_embedding: List[float], match_threshold: float, match_count: int) -> List[Dict]:
    sql = f"""
    SELECT 
        {CHUNK_ID_COL}::text as id, {FILENAME_COL} as filename, {CHUNK_TEXT_COL} as text_content,
        1 - ({EMBED_COL} <=> %s::vector) AS rank_score
    FROM {KB_TABLE}
    WHERE 1 - ({EMBED_COL} <=> %s::vector) > %s
    ORDER BY rank_score DESC LIMIT %s;
    """
    with connect_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (query_embedding, query_embedding, match_threshold, match_count))
            return cur.fetchall()


def keyword_search(query_terms: List[str], k: int = 10, require_term: str | None = None) -> List[Dict]:
    if not query_terms: return []
    cols_to_search = " || ' ' || ".join(TEXT_COLS)
    tsvector_col = f"to_tsvector('german', {cols_to_search})"
    
    tsquery = " & ".join(query_terms)
    
    if require_term:
        tsquery = f"({tsquery}) & {require_term}" 

    sql = f"""
    SELECT 
        {CHUNK_ID_COL}::text as id, {FILENAME_COL} as filename, {CHUNK_TEXT_COL} as text_content,
        ts_rank({tsvector_col}, to_tsquery('german', %s)) AS rank_score
    FROM {KB_TABLE}
    WHERE {tsvector_col} @@ to_tsquery('german', %s)
    ORDER BY rank_score DESC LIMIT %s;
    """
    with connect_db() as conn: 
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (tsquery, tsquery, k))
            return cur.fetchall()

# --- Angepasste Ranking-Funktion (Hybrid) (Unverändert) ---
def combine_and_rank_chunks(
    vec_hits: List[Dict], 
    kw_hits: List[Dict], 
    graph_hits: List[Dict], 
    max_chunks: int
) -> List[Dict]:
    
    deduplicated: Dict[str, Dict] = {} 

    for hit in graph_hits:
        hit['rank_score'] = 100.0 
        key = hit['id']
        if key not in deduplicated:
            deduplicated[key] = hit

    all_other_hits = sorted(vec_hits + kw_hits, key=lambda x: x.get('rank_score', 0.0), reverse=True)
    
    for hit in all_other_hits:
        key = hit['id'] 
        if key not in deduplicated:
            deduplicated[key] = hit
        
        if len(deduplicated) >= max_chunks:
            break
            
    final_list = sorted(deduplicated.values(), key=lambda x: x.get('rank_score', 0.0), reverse=True)
    return final_list[:max_chunks]


def pick_context(hits: List[Dict]) -> List[Dict]:
    blocks = []
    for hit in hits:
        blocks.append({
            "filename": hit.get('filename', 'Unbekannt'),
            "score": hit.get('rank_score', 0.0),
            "content": hit.get('text_content', '')
        })
    return blocks


def ctx_to_text(blocks: List[Dict]) -> str:
    if not blocks: return ""
    ctx = ["KONTEXT-ABSCHNITTE:"]
    for i, block in enumerate(blocks):
        ctx.append(f"\n--- Abschnitt {i+1} (Quelle: {block['filename']}, Score: {block['score']:.2f}) ---")
        ctx.append(block['content'])
    return "\n".join(ctx)

# ------------------------------------------------------------------------------
# Generische RAG-Pipeline (HYBRID: SQL-GRAPH + VEKTOR + KEYWORD)
# ------------------------------------------------------------------------------
# --- ÄNDERUNG: Gibt jetzt auch 'related_topics' zurück ---
def run_rag_pipeline(prompt: str, threshold: float, k: int, max_chunks: int) -> Tuple[str, List[Dict], List[str]]:
    """
    Führt die gesamte HYBRID-RAG-Pipeline aus:
    1. Entitäts-Extraktion -> SQL-Graph-Suche (holt Chunk-IDs)
    2. Graph-Suche nach verwandten Themen
    3. Embedding -> Vektorsuche
    4. Keyword-Suche
    5. Ranking (Graph-Treffer haben Prio) -> Kontext-Erstellung.
    Gibt (kontext_string, ranked_chunks, related_topics) zurück.
    """
    
    # --- SCHRITT 1: GRAPH-SUCHE ---
    entities = extract_entities_from_prompt(prompt, model=chat_model)
    graph_chunk_ids = query_graph_for_chunks(entities)
    
    # --- NEU: Verwandte Themen holen ---
    related_topics = get_related_topics_from_graph(entities)
    
    # --- SCHRITT 2: PARSEN ---
    q_clean, terms, must = parse_query(prompt) 
    if must:
        st.info(f"Filtere Ergebnisse auf: `{must}`")

    # --- SCHRITT 3: EMBEDDING ---
    try:
        query_embedding = get_embedding(prompt, model=embed_model)
    except Exception as e:
        st.error(f"Fehler beim Erstellen des OpenAI Embeddings: {e}")
        st.stop()

    # --- SCHRITT 4: HYBRID-SUCHE ---
    
    # A) Vektor-Suche
    vec_hits = match_documents(query_embedding, match_threshold=threshold, match_count=k)
    
    # B) Keyword-Suche
    kw_hits = keyword_search(terms, k=k, require_term=must if must else None) 
    
    # C) Graph-Treffer
    graph_hits = []
    if graph_chunk_ids:
        with connect_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = f"""
                SELECT {CHUNK_ID_COL}::text as id, {FILENAME_COL} as filename, {CHUNK_TEXT_COL} as text_content
                FROM {KB_TABLE}
                WHERE {CHUNK_ID_COL}::text = ANY(%s); 
                """
                cur.execute(sql, (graph_chunk_ids,))
                graph_hits = cur.fetchall()

    # --- SCHRITT 5: RANKEN & KONTEXTEN ---
    ranked_chunks = combine_and_rank_chunks(
        vec_hits, 
        kw_hits, 
        graph_hits, 
        max_chunks=max_chunks
    )
    
    blocks = pick_context(ranked_chunks)
    ctx = ctx_to_text(blocks)
    
    # --- SCHRITT 6: ZURÜCKGEBEN (mit related_topics) ---
    return ctx, ranked_chunks, related_topics

# ------------------------------------------------------------------------------
# Streamlit Hauptlogik
# ------------------------------------------------------------------------------

# --- NEUE SIDEBAR-STRUKTUR ---
with st.sidebar:
    # 1. Bild
    st.sidebar.image("ciferecigo.png", width=67) # 1/3 von 200px
    
    # 2. Über uns Text & Button
    # KORREKTUR: st.subheader() wendet 'headingFont' an
    st.subheader("Der KI Berater") 
    st.markdown(
        """
        Dieses Tool wurde von Thorsten Jankowski / ciferecigo entwickelt.
        Wir helfen KMUs und dem Mittelstand, KI-Strategien erfolgreich umzusetzen.
        """
    )
    # KORREKTUR: type="primary" hinzugefügt, um den Button rot zu machen
    st.link_button(
        "Kostenloses Erstgespräch buchen", 
        "https://calendar.app.google/kemaHAmTcqB2k5bE9",
        use_container_width=True,
        type="primary"
    )
    st.divider()

    # 3. Modell-Einstellungen
    # st.subheader() wendet 'headingFont' an
    st.subheader("Modell-Einstellungen")
    chat_model = st.text_input("Chat-Modell", DEFAULT_CHAT_MODEL)
    embed_model = st.text_input("Embedding-Modell", DEFAULT_EMBED_MODEL)
    debug_hits = st.checkbox("Debug-Modus (Treffer anzeigen)", value=True)

    # 4. Verläufe löschen
    clicked_button = sac.buttons(
        items=[
            sac.ButtonsItem(label='Alle Verläufe löschen', icon='trash')
        ],
        format_func='title', 
        index=None,
        use_container_width=True
    )
    if clicked_button == 'Alle Verläufe löschen':
        st.session_state.messages = []
        st.session_state.berater_messages = []
        st.rerun()
    
    st.divider()

    # 5. Such-Einstellungen
    # st.subheader() wendet 'headingFont' an
    st.subheader("Such-Einstellungen")
    threshold = st.slider(
        "Ähnlichkeits-Threshold (Vektor)", 
        min_value=0.0, max_value=1.0, value=0.01, step=0.01,
        help="Mindest-Score für die Vektorsuche. 0.01 = fast alles erlauben, 0.8 = nur Top-Treffer."
    )
    k = st.slider(
        "Maximale Treffer (K)", 
        min_value=3, max_value=20, value=10, step=1,
        help="Maximale Anzahl Treffer, die Vektor- und Keyword-Suche *initial* holen."
    )
    max_chunks_to_llm = st.slider(
        "Kontext-Abschnitte (Chunks)",
        min_value=1, max_value=10, value=5, step=1,
        help="Anzahl der besten Chunks, die als Kontext an das LLM gesendet werden."
    )
# --- ENDE SIDEBAR ---


# --- Initialisierung der Chat-Verläufe (2x) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "berater_messages" not in st.session_state:
    st.session_state.berater_messages = []
# --- NEU: Initialisierung für verwandte Themen ---
if "related_topics" not in st.session_state:
    st.session_state.related_topics = []


# --- sac.tabs (Linksbündig + FARBIG) ---
selected_tab = sac.tabs([
    sac.TabsItem(label='Strategie Berater', icon='robot'),
    sac.TabsItem(label='Allgemeiner Chat', icon='chat-dots'),
], format_func='title', align='left', return_index=False, color="#ea3323") # Rote Farbe hier
# --- ENDE sac.tabs ---


# --- TAB 1: Chat ---
if selected_tab == "Allgemeiner Chat":
    st.subheader("Allgemeiner Chat mit Gedächtnis")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Stellen Sie eine Frage an die Wissensdatenbank..."):
        
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # --- NEUER AUFRUF: RAG-Pipeline (ohne related_topics) ---
        ctx, ranked_chunks, _ = run_rag_pipeline(prompt, threshold, k, max_chunks_to_llm)
        
        if debug_hits:
            show_hits(ranked_chunks) 
            with st.expander("⬇️ KONTEXT, der in die Antwort geht", expanded=False):
                st.code(ctx)
        
        if not ctx.strip():
            st.warning("Keine relevanten Kontext-Abschnitte gefunden.")
        
        messages_for_api = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages_for_api.extend(st.session_state.messages) 

        if ctx.strip():
            last_user_message = messages_for_api.pop() 
            messages_for_api.append({
                "role": "user",
                "content": f"Frage: {last_user_message['content']}\n\nKONTEXT:\n{ctx}"
            })

        try:
            with st.spinner("KI-Assistent denkt nach..."):
                if _OPENAI_MODE == "sdk_v1":
                    res = client.chat.completions.create(model=chat_model, messages=messages_for_api, temperature=0.1)
                    answer = res.choices[0].message.content
                else:
                    res = client.ChatCompletion.create(model=chat_model, messages=messages_for_api, temperature=0.1)
                    answer = res['choices'][0]['message']['content']
            
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Fehler bei der OpenAI ChatCompletion API: {e}")

# --- TAB 2: Strategie Berater ---
if selected_tab == "Strategie Berater":
    # st.subheader() wendet 'headingFont' an
    st.subheader("Ihr KI-Strategie Berater")
    
    # FALL 1: Das Gespräch hat noch nicht begonnen -> Zeige Formular
    if not st.session_state.berater_messages:
        st.markdown("Beantworten Sie die folgenden Fragen, um eine maßgeschneiderte Roadmap und Best Practices zu erhalten.")
        
        with st.form(key="configurator_form"):
            st.markdown("##### 1. Wer sind Sie?")
            
            branchen_liste = [
                "(Bitte auswählen)", "Maschinenbau / Fertigung", "Handel / E-Commerce", "Handwerk", 
                "Dienstleistung (z.B. Agentur, Beratung)", "Gesundheitswesen / Soziales", 
                "Finanzen / Versicherung", "Baugewerbe / Immobilien", "Gastronomie / Tourismus",
                "Verwaltung / Öffentlicher Dienst", 
                "Andere Branche (bitte unten angeben)"
            ]
            branche = st.selectbox("Branche", branchen_liste)
            branche_freitext = st.text_input("Falls 'Andere' oder zur Spezifizierung:", placeholder="z.B. Spezialisierter Anlagenbau für die Pharmaindustrie")

            company_size_options = [
                "1-5 Mitarbeiter", "6-10 Mitarbeiter", "11-25 Mitarbeiter", "26-50 Mitarbeiter", 
                "51-250 Mitarbeiter (KMU)", "251-1000 Mitarbeiter (Mittelstand)", "> 1000 Mitarbeiter (Konzern)"
            ]
            company_size = st.selectbox("Unternehmensgröße", company_size_options)

            st.markdown("##### 2. Welche Rahmenbedingungen gibt es?")
            
            data_sources_options = [
                "E-Mail-Postfächer (z.B. Outlook, Gmail)",
                "Textdokumente (z.B. Word, PDFs, Verträge, Angebote)",
                "Tabellen (z.B. Excel-Listen für Controlling, Planung)",
                "Lokale Buchhaltungssoftware (z.B. DATEV, Lexware)",
                "Branchen-Software (z.B. Handwerker-Software, Kanzlei-Software)",
                "Website-Analytics / CMS (z.B. Wordpress, Matomo, Google Analytics)",
                "Support-Tickets (z.B. Zendesk, Zammad, E-Mail-Support)",
                "CRM (z.B. Salesforce, Hubspot, Pipedrive, WeClapp)",
                "ERP (z.B. SAP, Navision, JTL, Sage)",
                "Produktionsdaten (z.B. IoT, Sensoren, BDE)"
            ]
            
            st.markdown("Welche Datenquellen liegen vor? (Mehrfachauswahl)")
            data_sources = st.multiselect(
                "data_sources_hidden_label", 
                data_sources_options,
                placeholder="Bitte auswählen...",
                label_visibility="collapsed"
            )
            
            data_sources_freitext = st.text_input("Andere Datenquellen oder Spezifizierung:", placeholder="z.B. Altes Warenwirtschaftssystem, diverse Access-DBs")
            
            departments_options = [
                "Geschäftsführung", "Vertrieb / Sales", "Marketing", "Kundenservice / Support", "HR / Personal",
                "Buchhaltung / Finanzen / Controlling", "Produktion / F&E / Dienstleistungserbringung", "IT / Administration", "Logistik / Einkauf / SCM"
            ]
            
            st.markdown("Welche Abteilungen sind im Fokus? (MehrfachausLahl)")
            departments = st.multiselect(
                "departments_hidden_label", 
                departments_options,
                placeholder="Bitte auswählen...",
                label_visibility="collapsed"
            )

            st.markdown("##### 3. Was ist Ihr Ziel?")
            
            goals_options = [
                "Effizienz steigern / Kosten senken (z.B. durch Automatisierung)",
                "Vertriebsprozesse automatisieren (z.B. Angebotserstellung)",
                "Kundenservice verbessern (z.B. 24/7 Support, schnellere Antworten)",
                "Marketing personalisieren / Content-Erstellung",
                "Fachkräftemangel begegnen (Mitarbeiter entlasten)",
                "Neue digitale Geschäftsmodelle entwickeln",
                "Entscheidungsfindung datenbasiert verbessern (z.B. Reports)"
            ]
            
            st.markdown("Was sind Ihre Hauptziele? (Mehrfachauswahl)")
            goals_preselected = st.multiselect(
                "goals_hidden_label", 
                goals_options,
                placeholder="Bitte auswählen...",
                label_visibility="collapsed"
            )
            
            goals_freitext = st.text_area("Weitere Ziele oder Details:", 
                                          placeholder="z.B. Automatisierung der Rechnungsprüfung, Analyse von Support-Tickets zur Produktverbesserung...")

            st.markdown("---") 
            submit_button = st.form_submit_button("Strategie-Empfehlung generieren", use_container_width=True)

        # --- Logik nach dem Absenden des Formulars ---
        if submit_button:
            if (branche == "(Bitte auswählen)" and not branche_freitext) or (not goals_preselected and not goals_freitext):
                st.error("Bitte füllen Sie zumindest 'Branche' und 'Ziele' aus, um eine Empfehlung zu erhalten.")
            else:
                with st.spinner("Suche nach passenden Strategien und generiere Empfehlung..."):
                    
                    company_details = branche
                    if branche_freitext:
                        company_details = f"{branche} (Spezialisierung: {branche_freitext})"
                    
                    all_goals = goals_preselected
                    if goals_freitext:
                        all_goals.append(f"Weitere Details: {goals_freitext}")
                    
                    all_data_sources = data_sources
                    if data_sources_freitext:
                        all_data_sources.append(f"Weitere Details: {data_sources_freitext}")

                    mega_prompt_content = f"""
                    Der Nutzer benötigt eine KI-Implementierungs-Roadmap und Best Practices.
                    Situation des Nutzers:
                    - Unternehmen / Branche: {company_details}
                    - Größe: {company_size}
                    - Relevante Abteilungen: {', '.join(departments) if departments else 'Nicht spezifiziert'}
                    - Vorhandene Datenquellen: {', '.join(all_data_sources) if all_data_sources else 'Nicht spezifiziert'}
                    - Hauptziele: {', '.join(all_goals)}
                    
                    Bitte finde die relevantesten Informationen zu Methoden, Roadmaps, Best Practices und Use Cases,
                    die auf diese spezifische Situation (insbesondere KMU, die genannte Branche und die Ziele) passen.
                    """
                    
                    user_message_display = f"Meine Situation: {company_details} (Größe: {company_size}). Meine Ziele sind: {', '.join(all_goals)}."
                    
                    # --- ÄNDERUNG: 'related_topics' wird empfangen ---
                    ctx, ranked_chunks, related_topics = run_rag_pipeline(mega_prompt_content, threshold, k, max_chunks_to_llm)

                    if debug_hits:
                        show_hits(ranked_chunks)
                        with st.expander("⬇️ KONTEXT, der in die Antwort geht", expanded=False):
                            st.code(ctx)
                    
                    if not ctx.strip():
                        st.warning("Es wurden keine spezifischen Kontext-Abschnitte für diese Konfiguration gefunden. Die Antwort wird allgemeiner ausfallen.")
                    
                    messages_for_api = [
                        {"role": "system", "content": CONFIGURATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Nutzer-Anfrage (basierend auf Formular):\n{mega_prompt_content}\n\nKONTEXT:\n{ctx}"}
                    ]
                    
                    try:
                        res = client.chat.completions.create(model=chat_model, messages=messages_for_api, temperature=0.1)
                        answer = res.choices[0].message.content
                        
                        st.session_state.berater_messages.append({"role": "user", "content": user_message_display})
                        st.session_state.berater_messages.append({"role": "assistant", "content": answer})
                        
                        # --- NEU: Verwandte Themen im Session State speichern ---
                        st.session_state.related_topics = related_topics
                        
                        st.rerun()

                    except Exception as e:
                        st.error(f"Fehler bei der OpenAI ChatCompletion API: {e}")
    
    # FALL 2: Das Gespräch läuft bereits -> Zeige Chat-Verlauf und Eingabefeld
    else:
        for i, message in enumerate(st.session_state.berater_messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and i == 1: 
                    st.divider()
                    st.markdown("**War diese Erstanalyse hilfreich?**")
                    st.markdown("Eine automatisierte Analyse kann einen persönlichen Workshop nicht ersetzen. Wenn Sie diese Roadmap und Methoden konkret umsetzen möchten, lassen Sie uns sprechen.")
                    
                    st.link_button(
                        label="📅 Kostenloses Erstgespräch buchen", 
                        url="https://calendar.app.google/kemaHAmTcqB2k5bE9",
                        use_container_width=True,
                        type="primary" 
                    )
                    st.divider()
        
        # --- NEU: Verwandte Themen anzeigen (NACH der Schleife) ---
        if st.session_state.related_topics:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("**Themen, in denen Ihre Problematik ebenfalls angesprochen wird (Best Practices):**")
                
                # Zeigt die Themen als klickbare "Tags" an
                topics_md = " ".join([f"`{t}`" for t in st.session_state.related_topics])
                st.info(f"Fragen Sie mich bei Interesse gerne nach: {topics_md}")
            
            # Themen nur einmal anzeigen
            st.session_state.related_topics = []

        # Chat-Eingabefeld für FOLGEFRAGEN
        if prompt := st.chat_input("Stellen Sie eine Folgefrage zu Ihrer Strategie..."):
            
            st.chat_message("user").markdown(prompt)
            st.session_state.berater_messages.append({"role": "user", "content": prompt})

            # --- ÄNDERUNG: 'related_topics' wird empfangen ---
            ctx, ranked_chunks, related_topics = run_rag_pipeline(prompt, threshold, k, max_chunks_to_llm)
            
            if debug_hits:
                show_hits(ranked_chunks) 
                with st.expander("⬇️ KONTEXT, der in die Antwort geht", expanded=False):
                    st.code(ctx)
            
            if not ctx.strip():
                st.warning("Keine relevanten Kontext-Abschnitte für diese Folgefrage gefunden.")
            
            messages_for_api = [{"role": "system", "content": CONFIGURATOR_SYSTEM_PROMPT}]
            messages_for_api.extend(st.session_state.berater_messages) 

            if ctx.strip():
                last_user_message = messages_for_api.pop() 
                messages_for_api.append({
                    "role": "user",
                    "content": f"Frage: {last_user_message['content']}\n\nKONTEXT:\n{ctx}"
                })
            
            try:
                with st.spinner("KI-Assistent denkt nach..."):
                    if _OPENAI_MODE == "sdk_v1":
                        res = client.chat.completions.create(model=chat_model, messages=messages_for_api, temperature=0.1)
                        answer = res.choices[0].message.content
                    else:
                        res = client.ChatCompletion.create(model=chat_model, messages=messages_for_api, temperature=0.1)
                        answer = res['choices'][0]['message']['content']
                
                st.chat_message("assistant").markdown(answer)
                st.session_state.berater_messages.append({"role": "assistant", "content": answer})
                
                # --- NEU: Verwandte Themen im Session State speichern ---
                st.session_state.related_topics = related_topics
                st.rerun() 

            except Exception as e:
                st.error(f"Fehler bei der OpenAI ChatCompletion API: {e}")