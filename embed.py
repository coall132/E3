#!/usr/bin/env python3
import os
import re
import time
import math
from typing import List, Optional, Tuple

import numpy as np
import psycopg2
import psycopg2.extras as pgx
import psycopg2
import psycopg2.extras as pgx
import psycopg2.errors as pgerr

# ---------------------------
#  Paramètres via ENV
# ---------------------------
PG_HOST = "localhost"
PG_PORT = "5433"
PG_DB   = 'mydb'
PG_USER = 'admin'
PG_PASS = 'admin123'

ETAB_SCHEMA = os.getenv("ETAB_SCHEMA", "public")
ETAB_TABLE  = os.getenv("ETAB_TABLE", "etab")

REVIEW_SCHEMA = os.getenv("REVIEW_SCHEMA", "public")
REVIEW_TABLE  = os.getenv("REVIEW_TABLE", "reviews")

EMBED_SCHEMA  = os.getenv("EMBED_SCHEMA", "public")
EMBED_TABLE   = os.getenv("EMBED_TABLE", "etab_embedding")

# colonnes texte des établissements
COL_EDITORIAL = os.getenv("EDITORIAL_COL", "editorialSummary_text")
COL_DESC      = os.getenv("DESCRIPTION_COL", "description")

# colonnes de reviews
COL_REVIEW_ID   = os.getenv("REVIEW_ID_COL", "id_review")
COL_REVIEW_ETAB = os.getenv("REVIEW_ETAB_COL", "id_etab")
COL_REVIEW_TEXT = os.getenv("REVIEW_TEXT_COL", "original_text")

# modèle d'embedding
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
EMBED_DEVICE     = os.getenv("EMBED_DEVICE", "cpu") 

# limites
BATCH_SIZE   = int(os.getenv("BATCH_ETABS", "200"))
EMBED_MAX_REV = int(os.getenv("EMBED_MAX_REV", "5"))

# recomputation
RECOMPUTE = os.getenv("EMBED_RECOMPUTE", "0") == "1"
DEBUG_REV = os.getenv("DEBUG_REV", "1") == "1"

def log(msg: str):
    print(f"[embed] {msg}", flush=True)

def _safe_ident(s: str) -> str:
    """Très simple validation d'identifiant SQL (nom de table/colonne)."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s):
        raise ValueError(f"Identifiant SQL invalide: {s}")
    return s

def connect():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )

def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS "{_safe_ident(EMBED_SCHEMA)}"."{_safe_ident(EMBED_TABLE)}" (
                id_etab    INTEGER PRIMARY KEY
                           REFERENCES "{_safe_ident(ETAB_SCHEMA)}"."{_safe_ident(ETAB_TABLE)}"(id_etab)
                           ON DELETE CASCADE,
                desc_embed JSONB,
                rev_embeds JSONB,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
    conn.commit()

def fetch_total_etabs(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM "{_safe_ident(ETAB_SCHEMA)}"."{_safe_ident(ETAB_TABLE)}"')
        return cur.fetchone()[0]

def fetch_existing_ids(conn) -> set:
    with conn.cursor() as cur:
        cur.execute(f'SELECT id_etab FROM "{_safe_ident(EMBED_SCHEMA)}"."{_safe_ident(EMBED_TABLE)}"')
        rows = cur.fetchall()
    return {r[0] for r in rows}

def fetch_etab_batch(conn, offset: int, limit: int) -> List[Tuple[int, Optional[str], Optional[str]]]:
    with conn.cursor(cursor_factory=pgx.DictCursor) as cur:
        cur.execute(f'''
            SELECT id_etab,
                   COALESCE("{_safe_ident(COL_EDITORIAL)}",'') AS editorial,
                   COALESCE("{_safe_ident(COL_DESC)}",'')       AS descr
            FROM "{_safe_ident(ETAB_SCHEMA)}"."{_safe_ident(ETAB_TABLE)}"
            ORDER BY id_etab
            OFFSET %s LIMIT %s
        ''', (offset, limit))
        rows = cur.fetchall()
    return [(int(r["id_etab"]), r["editorial"] or "", r["descr"] or "") for r in rows]

def fetch_reviews(conn, id_etab: int, limit: int) -> List[str]:
    try:
        with conn.cursor() as cur:
            cur.execute(f'''
                SELECT {_safe_ident(COL_REVIEW_TEXT)}
                FROM "{_safe_ident(REVIEW_SCHEMA)}"."{_safe_ident(REVIEW_TABLE)}"
                WHERE {_safe_ident(COL_REVIEW_ETAB)} = %s
                ORDER BY {_safe_ident(COL_REVIEW_ID)} DESC
                LIMIT %s
            ''', (id_etab, limit))
            rows = cur.fetchall()
    except (pgerr.UndefinedTable, pgerr.UndefinedColumn) as e:
        conn.rollback()
        if DEBUG_REV:
            print(f"[embed][DEBUG] reviews indispo pour etab {id_etab}: {type(e).__name__}")
        return []
    except Exception as e:
        conn.rollback()
        if DEBUG_REV:
            print(f"[embed][DEBUG] erreur SQL reviews etab {id_etab}: {e}")
        return []

    texts = []
    for (t,) in rows:
        if t:
            t = str(t).strip()
            if t:
                texts.append(t)
    if DEBUG_REV:
        print(f"[embed][DEBUG] etab {id_etab}: {len(texts)} reviews non vides")
    return texts
def encode_texts(model, texts: List[str]) -> List[np.ndarray]:
    if not texts:
        return []
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, device=EMBED_DEVICE)
    arr = np.asarray(emb, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return [arr[i] for i in range(arr.shape[0])]

def upsert_embeddings(conn, rows: List[Tuple[int, Optional[list], Optional[list]]]):
    if not rows:
        return
    try:
        with conn.cursor() as cur:
            pgx.execute_values(
                cur,
                f'''
                INSERT INTO "{_safe_ident(EMBED_SCHEMA)}"."{_safe_ident(EMBED_TABLE)}"
                    (id_etab, desc_embed, rev_embeds)
                VALUES %s
                ON CONFLICT (id_etab) DO UPDATE SET
                    desc_embed = EXCLUDED.desc_embed,
                    rev_embeds = EXCLUDED.rev_embeds,
                    updated_at = now()
                ''',
                [(i, pgx.Json(d), pgx.Json(r)) for (i, d, r) in rows],
                template="(%s, %s, %s)",
                page_size=100
            )
        conn.commit()
    except Exception as e:
        # Affiche le code/texte d’erreur Postgres pour comprendre si c’est une FK, etc.
        conn.rollback()
        errcode = getattr(e, 'pgcode', None)
        errmsg  = getattr(e, 'pgerror', str(e))
        print(f"[embed][UPSERT-ERROR] code={errcode} msg={errmsg}")
        # Tu peux re-raise si tu préfères stopper net :
        # raise

def assert_etab_columns(conn):
    need = {COL_EDITORIAL, COL_DESC}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s
        """, (ETAB_SCHEMA, ETAB_TABLE))
        cols = {r[0] for r in cur.fetchall()}
    missing = [c for c in need if c not in cols]
    if missing:
        log(f"ATTENTION: colonnes manquantes dans {ETAB_SCHEMA}.{ETAB_TABLE}: {missing} (je les considérerai comme vides)")


def main():
    # 1) DB + table
    conn = connect()
    ensure_table(conn)
    assert_etab_columns(conn)

    # 2) Modèle embeddings
    from sentence_transformers import SentenceTransformer
    log(f"Loading model {EMBED_MODEL_NAME} on {EMBED_DEVICE}")
    st = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)

    total = fetch_total_etabs(conn)
    if total == 0:
        log("Aucun établissement trouvé.")
        return

    skip_ids = set()
    if not RECOMPUTE:
        skip_ids = fetch_existing_ids(conn)
        log(f"{len(skip_ids)} id_etab déjà présents — ils seront ignorés.")

    pages = math.ceil(total / BATCH_SIZE)
    t0 = time.time()
    processed = 0

    for p in range(pages):
        offset = p * BATCH_SIZE
        batch = fetch_etab_batch(conn, offset, BATCH_SIZE)
        if not batch:
            break

        # prépare les upserts
        upserts: List[Tuple[int, Optional[list], Optional[list]]] = []

        for id_etab, editorial, descr in batch:
            if (not RECOMPUTE) and (id_etab in skip_ids):
                continue

            # Texte de base pour "desc_embed"
            base_texts = [t for t in [editorial, descr] if t]
            desc_vec_list: Optional[list] = None
            if base_texts:
                concat = ". ".join(base_texts)
                v = encode_texts(st, [concat])[0]
                desc_vec_list = v.tolist()

            # Reviews -> liste de vecteurs
            rev_vecs_list: Optional[list] = None
            try:
                rev_texts = fetch_reviews(conn, id_etab, EMBED_MAX_REV)
                if rev_texts:
                    rev_vecs_list = [v.tolist() for v in encode_texts(st, rev_texts)]
            except Exception as e:
                # si la table de reviews n'existe pas, on ignore
                rev_vecs_list = None

            upserts.append((id_etab, desc_vec_list, rev_vecs_list))
            processed += 1

        # upsert DB
        upsert_embeddings(conn, upserts)
        log(f"Page {p+1}/{pages} -> upserts: {len(upserts)} (cumul: {processed}/{total})")

    log(f"Terminé en {time.time()-t0:.1f}s. Éléments traités: {processed}/{total}")
    conn.close()

if __name__ == "__main__":
    main()
