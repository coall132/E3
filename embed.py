#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import math
from typing import List, Tuple, Optional

import numpy as np
import psycopg2
import psycopg2.extras as pgx
import psycopg2.errors as pgerr
from sentence_transformers import SentenceTransformer

# =========================
# Configuration (ENV)
# =========================
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5433")
PG_DB   = os.getenv("PG_DB",   "mydb")
PG_USER = os.getenv("PG_USER", "admin")
PG_PASS = os.getenv("PG_PASS", "admin123")

ETAB_SCHEMA = os.getenv("ETAB_SCHEMA", "public")
ETAB_TABLE  = os.getenv("ETAB_TABLE",  "etab")

REVIEW_SCHEMA = os.getenv("REVIEW_SCHEMA", "public")
REVIEW_TABLE  = os.getenv("REVIEW_TABLE",  "reviews")

EMBED_SCHEMA = os.getenv("EMBED_SCHEMA", "public")
EMBED_TABLE  = os.getenv("EMBED_TABLE",  "etab_embedding")

COL_EDITORIAL = os.getenv("EDITORIAL_COL",   "editorialSummary_text")
COL_DESC      = os.getenv("DESCRIPTION_COL", "description")

COL_REVIEW_ID   = os.getenv("REVIEW_ID_COL",   "id_review")
COL_REVIEW_ETAB = os.getenv("REVIEW_ETAB_COL", "id_etab")
COL_REVIEW_TEXT = os.getenv("REVIEW_TEXT_COL", "original_text")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
EMBED_DEVICE     = os.getenv("EMBED_DEVICE",     "cpu")

BATCH_SIZE     = int(os.getenv("BATCH_ETABS",   "200"))
EMBED_MAX_REV  = int(os.getenv("EMBED_MAX_REV", "5"))
RECOMPUTE      = os.getenv("EMBED_RECOMPUTE", "0") == "1"
DEBUG_REV      = os.getenv("DEBUG_REV",       "1") == "1"

# =========================
# Helpers
# =========================
def log(msg: str):
    print(f"[embed] {msg}", flush=True)

def _id(s: str) -> str:
    """Validation simple d'identifiants SQL (nom de schéma/table/colonne)."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s):
        raise ValueError(f"Identifiant SQL invalide: {s}")
    return s

def connect():
    return psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS)

def ensure_embedding_table(conn):
    with conn.cursor() as cur:
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS "{_id(EMBED_SCHEMA)}"."{_id(EMBED_TABLE)}" (
                id_etab    INTEGER PRIMARY KEY
                           REFERENCES "{_id(ETAB_SCHEMA)}"."{_id(ETAB_TABLE)}"(id_etab)
                           ON DELETE CASCADE,
                desc_embed JSONB,
                rev_embeds JSONB,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        ''')
    conn.commit()

def total_etabs(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM "{_id(ETAB_SCHEMA)}"."{_id(ETAB_TABLE)}"')
        return cur.fetchone()[0]

def existing_embed_ids(conn) -> set:
    with conn.cursor() as cur:
        cur.execute(f'SELECT id_etab FROM "{_id(EMBED_SCHEMA)}"."{_id(EMBED_TABLE)}"')
        return {r[0] for r in cur.fetchall()}

def etab_batch(conn, offset: int, limit: int):
    with conn.cursor(cursor_factory=pgx.DictCursor) as cur:
        cur.execute(f'''
            SELECT id_etab,
                   COALESCE("{_id(COL_EDITORIAL)}",'') AS editorial,
                   COALESCE("{_id(COL_DESC)}",'')       AS descr
            FROM "{_id(ETAB_SCHEMA)}"."{_id(ETAB_TABLE)}"
            ORDER BY id_etab
            OFFSET %s LIMIT %s
        ''', (offset, limit))
        rows = cur.fetchall()
        return [(int(r["id_etab"]), r["editorial"] or "", r["descr"] or "") for r in rows]

def recent_reviews(conn, id_etab: int, limit: int):
    try:
        with conn.cursor() as cur:
            cur.execute(f'''
                SELECT {_id(COL_REVIEW_TEXT)}
                FROM "{_id(REVIEW_SCHEMA)}"."{_id(REVIEW_TABLE)}"
                WHERE {_id(COL_REVIEW_ETAB)} = %s
                ORDER BY {_id(COL_REVIEW_ID)} DESC
                LIMIT %s
            ''', (id_etab, limit))
            rows = cur.fetchall()
    except (pgerr.UndefinedTable, pgerr.UndefinedColumn):
        conn.rollback()
        if DEBUG_REV:
            log(f"[DEBUG] table/colonne reviews absente pour etab {id_etab}")
        return []
    except Exception as e:
        conn.rollback()
        if DEBUG_REV:
            log(f"[DEBUG] erreur SQL reviews etab {id_etab}: {e}")
        return []

    texts = [str(t[0]).strip() for t in rows if t and t[0]]
    if DEBUG_REV:
        log(f"[DEBUG] etab {id_etab}: {len(texts)} reviews non vides")
    return texts

def encode_many(model: SentenceTransformer, texts: List[str]):
    if not texts:
        return []
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, device=EMBED_DEVICE)
    arr = np.asarray(emb, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return [arr[i] for i in range(arr.shape[0])]

def upsert_embed_rows(conn, rows: List[Tuple[int, Optional[list], Optional[list]]]):
    if not rows:
        return
    try:
        with conn.cursor() as cur:
            pgx.execute_values(
                cur,
                f'''
                INSERT INTO "{_id(EMBED_SCHEMA)}"."{_id(EMBED_TABLE)}"
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
        conn.rollback()
        log(f"[UPSERT-ERROR] {getattr(e, 'pgcode', None)} {getattr(e, 'pgerror', str(e))}")

def assert_etab_cols(conn) -> None:
    needed = {COL_EDITORIAL, COL_DESC}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s
        """, (ETAB_SCHEMA, ETAB_TABLE))
        cols = {r[0] for r in cur.fetchall()}
    miss = [c for c in needed if c not in cols]
    if miss:
        log(f"ATTENTION: colonnes manquantes dans {ETAB_SCHEMA}.{ETAB_TABLE}: {miss} (traitées comme vides)")

# =========================
# Main
# =========================
def main():
    t0 = time.time()
    conn = connect()
    ensure_embedding_table(conn)
    assert_etab_cols(conn)

    log(f"Chargement du modèle {EMBED_MODEL_NAME} sur {EMBED_DEVICE}")
    model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)

    total = total_etabs(conn)
    if total == 0:
        log("Aucun établissement.")
        conn.close()
        return

    skip = existing_embed_ids(conn) if not RECOMPUTE else set()
    if skip:
        log(f"{len(skip)} embeddings déjà présents — ignorés (RECOMPUTE=0).")

    pages = math.ceil(total / BATCH_SIZE)
    processed = 0

    for p in range(pages):
        batch = etab_batch(conn, offset=p * BATCH_SIZE, limit=BATCH_SIZE)
        if not batch:
            break

        upserts: List[Tuple[int, Optional[list], Optional[list]]] = []
        for id_etab, editorial, descr in batch:
            if not RECOMPUTE and id_etab in skip:
                continue

            # Texte principal (desc_embed)
            desc_vec = None
            base = ". ".join([t for t in (editorial, descr) if t])
            if base:
                vec = encode_many(model, [base])[0]
                desc_vec = vec.tolist()

            # Reviews (rev_embeds)
            rev_vecs = None
            texts = recent_reviews(conn, id_etab, EMBED_MAX_REV)
            if texts:
                rev_vecs = [v.tolist() for v in encode_many(model, texts)]

            upserts.append((id_etab, desc_vec, rev_vecs))
            processed += 1

        upsert_embed_rows(conn, upserts)
        log(f"Page {p+1}/{pages} — upserts: {len(upserts)} (cumul {processed}/{total})")

    log(f"Terminé en {time.time()-t0:.1f}s — {processed}/{total} établissements traités.")
    conn.close()

if __name__ == "__main__":
    main()
