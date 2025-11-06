# test_db.py
import os
import psycopg2
from dotenv import load_dotenv

# 1. Lade die .env-Datei (genau wie in app.py)
print("Lade .env-Datei...")
load_dotenv(override=True)
CONNECTION_STRING = os.getenv("SUPABASE_PG_CONN")

if not CONNECTION_STRING:
    print("FEHLER: 'SUPABASE_PG_CONN' wurde nicht in der .env-Datei gefunden.")
    exit()

print(f"Versuche, Verbindung herzustellen mit String: {CONNECTION_STRING[:80]}...") 
# (Zeigt nur die ersten 80 Zeichen, um das Passwort zu verbergen)

try:
    # 2. Baue die Verbindung auf (genau wie in app.py)
    conn = psycopg2.connect(CONNECTION_STRING)
    
    print("\n-------------------------------------------")
    print("✅ ERFOLG! Verbindung zur Datenbank hergestellt.")
    print("-------------------------------------------\n")
    
    # 3. Mache eine Test-Abfrage
    print("Führe eine Test-Abfrage aus (zähle Chunks)...")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM public.kb_chunks;")
    
    # 4. Zeige das Ergebnis
    count = cur.fetchone()[0]
    print(f"Ergebnis: {count} Chunks in 'public.kb_chunks' gefunden.")
    
    # 5. Schließe die Verbindung
    cur.close()
    conn.close()
    print("Verbindung erfolgreich geschlossen.")

except Exception as e:
    print("\n-------------------------------------------")
    print("❌ FEHLER BEIM VERBINDUNGSAUFBAU:")
    print(e)
    print("-------------------------------------------\n")
    print("Bitte prüfen Sie die 'SUPABASE_PG_CONN'-Zeile in Ihrer .env-Datei.")