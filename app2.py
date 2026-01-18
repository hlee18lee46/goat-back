import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import snowflake.connector
import os
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import snowflake.connector
from gradient import (
    GradientChatInput,
    gradient_chat,
    GradientConfigError,
    GradientRequestError,
)
from mongo_api import router as mongo_router




import json
import base58
from pydantic import BaseModel, Field
from fastapi import HTTPException

from solana.rpc.api import Client
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solders.hash import Hash

from solana.rpc.commitment import Finalized

from solders.signature import Signature
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
# Add this line right here:
SOLANA_CLIENT = Client(os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com"))

app = FastAPI(title="Snowflake Events API")
app.include_router(mongo_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_private_key_bytes():
    path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
    if not path:
        raise RuntimeError("Missing SNOWFLAKE_PRIVATE_KEY_PATH")

    passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE") or None
    if passphrase:
        passphrase = passphrase.encode()

    with open(path, "rb") as f:
        p_key = serialization.load_pem_private_key(
            f.read(),
            password=passphrase,
            backend=default_backend()
        )

    # Snowflake connector expects DER bytes
    return p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

def get_conn():
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        private_key=get_private_key_bytes(),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE"),
    )


class EventIn(BaseModel):
    id: str
    title: str
    starts_at: str  # 'YYYY-MM-DD HH:MM:SS'
    venue: Optional[str] = None
    city: Optional[str] = None

class EventOut(BaseModel):
    id: str
    title: str
    starts_at: str
    venue: Optional[str] = None
    city: Optional[str] = None
    created_at: str

@app.get("/health")
def health():
    try:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return {"ok": True}
        finally:
            conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/events", response_model=dict)
def insert_event(e: EventIn):
    sql = """
        INSERT INTO EVENTS (id, title, starts_at, venue, city)
        VALUES (%s, %s, %s, %s, %s)
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (e.id, e.title, e.starts_at, e.venue, e.city))
        conn.commit()
        return {"ok": True, "inserted_id": e.id}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
    finally:
        conn.close()

@app.post("/events/bulk", response_model=dict)
def insert_events_bulk(events: List[EventIn]):
    if not events:
        raise HTTPException(status_code=400, detail="events is empty")

    sql = """
        INSERT INTO EVENTS (id, title, starts_at, venue, city)
        VALUES (%s, %s, %s, %s, %s)
    """
    rows = [(e.id, e.title, e.starts_at, e.venue, e.city) for e in events]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()
        return {"ok": True, "inserted": len(events)}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
    finally:
        conn.close()

@app.get("/events", response_model=dict)
def read_events(limit: int = Query(50, ge=1, le=1000)):
    sql = """
        SELECT id, title, TO_VARCHAR(starts_at), venue, city, TO_VARCHAR(created_at)
        FROM EVENTS
        ORDER BY created_at DESC
        LIMIT %s
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()

        # rows are tuples in the same order as SELECT
        data = [
            {
                "id": r[0],
                "title": r[1],
                "starts_at": r[2],
                "venue": r[3],
                "city": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]
        return {"ok": True, "events": data}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
    finally:
        conn.close()

@app.post("/ai/chat", response_model=dict)
async def ai_chat(payload: GradientChatInput):
    try:
        text = await gradient_chat(
            prompt=payload.prompt,
            system=payload.system,
            model=payload.model,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
        )
        return {"ok": True, "text": text}
    except GradientConfigError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except GradientRequestError as e:
        raise HTTPException(
            status_code=e.status_code or 502,
            detail={"message": str(e), "body": e.body},
        )


MEMO_PROGRAM_ID = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")


def get_solana_client() -> Client:
    rpc = os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com")
    return Client(rpc)


def get_solana_keypair() -> Keypair:
    b58 = os.getenv("SOLANA_PRIVATE_KEY_BASE58")
    if not b58:
        raise RuntimeError("Missing SOLANA_PRIVATE_KEY_BASE58 in .env")
    secret = base58.b58decode(b58)

    # solders Keypair.from_bytes expects 64-byte secret key (ed25519)
    # Many exports are 64 bytes; if you have 32 bytes, it’s a seed and needs conversion.
    if len(secret) != 64:
        raise RuntimeError(f"SOLANA_PRIVATE_KEY_BASE58 decoded length must be 64 bytes, got {len(secret)}")
    return Keypair.from_bytes(secret)


class SongMetadataIn(BaseModel):
    name: str = Field(..., examples=["Midnight Lofi #12"])
    artist: str = Field(..., examples=["Han"])
    bpm: int = Field(92, ge=40, le=220)
    key: str = Field("C minor")
    audio_url: str = Field(..., examples=["https://yourcdn.com/song.wav"])
    cover_url: Optional[str] = Field(None, examples=["https://yourcdn.com/cover.png"])

    # optional: hash of the audio bytes for integrity
    audio_sha256: Optional[str] = None


@app.post("/solana/devnet/publish-metadata", response_model=dict)
def publish_metadata_to_solana_devnet(payload: SongMetadataIn):
    """
    Stores metadata on Solana devnet via the Memo program.
    Returns a tx signature you can use to verify on a block explorer.
    """
    try:
        client = get_solana_client()
        kp = get_solana_keypair()
        payer = kp.pubkey()

        # Keep memo small (Solana tx size limits). Use essential fields only.
        memo_obj = {
            "type": "song_metadata_v1",
            "name": payload.name,
            "artist": payload.artist,
            "bpm": payload.bpm,
            "key": payload.key,
            "audio_url": payload.audio_url,
            "cover_url": payload.cover_url,
            "audio_sha256": payload.audio_sha256,
        }
        memo_str = json.dumps(memo_obj, separators=(",", ":"), ensure_ascii=False)

        # Memo instruction: program id = Memo, data = memo bytes, no accounts required
        ix = Instruction(
            program_id=MEMO_PROGRAM_ID,
            accounts=[],
            data=memo_str.encode("utf-8"),
        )

        # Build & send tx
        latest = client.get_latest_blockhash()
        if not latest.value:
            raise RuntimeError("Failed to fetch latest blockhash")

        # blockhash is ALREADY a solders.hash.Hash
        bh = latest.value.blockhash

        msg = MessageV0.try_compile(
            payer=payer,
            instructions=[ix],
            address_lookup_table_accounts=[],
            recent_blockhash=bh,
        )

        tx = VersionedTransaction(msg, [kp])

        res = client.send_transaction(tx)
        if not res.value:
            raise RuntimeError(f"send_transaction failed: {res}")

        sig_str = str(res.value)  # <-- convert Signature -> string

        return {
            "ok": True,
            "network": "devnet",
            "payer": str(payer),
            "signature": sig_str,
            "memo_preview": memo_obj,
        }


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



from solders.signature import Signature
import json
from fastapi import HTTPException

@app.get("/solana/devnet/memo/{signature}")
def get_memo(signature: str):
    try:
        print(f"DEBUG: Processing signature {signature}")
        sig = Signature.from_string(signature)
        
        # We use jsonParsed to get the logs in a readable format
        resp = SOLANA_CLIENT.get_transaction(
            sig, 
            encoding="jsonParsed", 
            max_supported_transaction_version=0
        )
        
        if not resp.value:
            raise HTTPException(status_code=404, detail="TX not found")

        # CRITICAL: Solders objects can't be indexed like dicts. 
        # We must navigate the object attributes.
        meta = resp.value.transaction.meta
        if not meta:
            raise HTTPException(status_code=404, detail="Transaction metadata missing")

        logs = meta.log_messages
        memos = []

        for line in logs:
            if "Program log: Memo" in line:
                # Logs usually look like: Program log: Memo (len 174): "{...}"
                # We split by the colon and remove the surrounding quotes
                parts = line.split("): ")
                if len(parts) > 1:
                    raw_content = parts[1].strip().strip('"')
                    # Unescape the string if it contains \"
                    clean_content = raw_content.replace('\\"', '"')
                    try:
                        memos.append(json.loads(clean_content))
                    except:
                        memos.append(clean_content)

        return {
            "ok": True, 
            "signature": signature, 
            "data": memos[0] if memos else None,
            "all_memos": memos
        }

    except Exception as e:
        # This will print the error to your TERMINAL window
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))

def get_memo_from_signature(signature: str):
    """
    Fetch a devnet transaction by signature and extract Memo text from log messages.
    Version-proof: we convert the RPC response to a JSON-able dict and search for logMessages.
    """
    try:
        client = get_solana_client()
        sig = Signature.from_string(signature)

        resp = client.get_transaction(
            sig,
            encoding="json",  # keep generic; shape varies by version
            max_supported_transaction_version=0,
        )

        if not resp.value:
            raise HTTPException(status_code=404, detail="Transaction not found (or not available yet).")

        # Convert response object -> json-friendly dict (handles solders types)
        obj = json.loads(json.dumps(resp, default=str))

        # The logs can appear under:
        # resp["result"]["meta"]["logMessages"] (classic)
        # or other nesting depending on encoding wrapper.
        # We'll search for logMessages anywhere in the structure.

        def find_logs(x):
            if isinstance(x, dict):
                # common key in Solana JSON RPC
                if "logMessages" in x and isinstance(x["logMessages"], list):
                    return x["logMessages"]
                # sometimes snake_case
                if "log_messages" in x and isinstance(x["log_messages"], list):
                    return x["log_messages"]
                for v in x.values():
                    got = find_logs(v)
                    if got:
                        return got
            elif isinstance(x, list):
                for v in x:
                    got = find_logs(v)
                    if got:
                        return got
            return None

        logs = find_logs(obj) or []

        memos = []
        for line in logs:
            marker = "Program log: Memo"
            if marker in line and "): " in line:
                memo_text = line.split("): ", 1)[1].strip()

                # logs often wrap in quotes
                if memo_text.startswith('"') and memo_text.endswith('"'):
                    memo_text = memo_text[1:-1]

                # unescape \" etc.
                try:
                    memo_text = memo_text.encode("utf-8").decode("unicode_escape")
                except Exception:
                    pass

                parsed = None
                try:
                    parsed = json.loads(memo_text)
                except Exception:
                    parsed = None

                memos.append({"memo": memo_text, "json": parsed})

        return {
            "ok": True,
            "network": "devnet",
            "signature": signature,
            "memos": memos,
            "log_lines_found": len(logs),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/solana/devnet/wallet", response_model=dict)
def devnet_wallet():
    kp = get_solana_keypair()
    return {"ok": True, "pubkey": str(kp.pubkey())}

import json
import traceback
from fastapi import HTTPException

@app.get("/solana/devnet/balance", response_model=dict)
def devnet_balance():
    try:
        client = get_solana_client()
        kp = get_solana_keypair()

        resp = client.get_balance(kp.pubkey())

        # ---- Version-proof extraction ----
        lamports = None

        # Newer solana-py style: resp.value is an int
        if hasattr(resp, "value"):
            lamports = resp.value

        # Some wrappers: resp["result"]["value"]
        if lamports is None:
            try:
                obj = json.loads(json.dumps(resp, default=str))
                lamports = (obj.get("result") or {}).get("value")
            except Exception:
                lamports = None

        if lamports is None:
            raise RuntimeError(f"Could not parse get_balance response: {resp}")

        return {
            "ok": True,
            "pubkey": str(kp.pubkey()),
            "lamports": int(lamports),
            "sol": float(lamports) / 1_000_000_000,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=repr(e))


# --- SONG METADATA (Snowflake) ------------------------------------------------
from datetime import datetime
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field

# If you want to control table via env:
SONG_METADATA_TABLE = os.getenv("SONG_METADATA_TABLE", "SONG_METADATA")

def _qualified_song_table() -> str:
    """
    Uses database/schema from env if present; otherwise assumes your connection already
    sets them. Fully qualifying prevents 'current database is null' surprises.
    """
    db = os.getenv("SNOWFLAKE_DATABASE") or "MY_DB"
    schema = os.getenv("SNOWFLAKE_SCHEMA") or "PUBLIC"
    return f'{db}.{schema}.{SONG_METADATA_TABLE}'


class SongMetadataRowIn(BaseModel):
    # Everything optional to avoid errors when fields missing
    id: Optional[str] = None  # allow client-supplied id; else Snowflake default UUID_STRING()
    name: Optional[str] = None
    artist: Optional[str] = None
    vibe: Optional[str] = None

    bpm: Optional[int] = Field(default=None, ge=0)
    key: Optional[str] = None
    mode: Optional[str] = None
    bars: Optional[int] = Field(default=None, ge=0)
    energy: Optional[float] = None
    seed: Optional[int] = None

    audio_url: Optional[str] = None
    midi_url: Optional[str] = None
    video_url: Optional[str] = None
    cover_url: Optional[str] = None

    solana_wallet: Optional[str] = None
    solana_signature: Optional[str] = None

    audio_sha256: Optional[str] = None
    ai_prompt: Optional[str] = None


class SongMetadataRowOut(BaseModel):
    id: str
    name: Optional[str] = None
    artist: Optional[str] = None
    vibe: Optional[str] = None
    bpm: Optional[int] = None
    key: Optional[str] = None
    mode: Optional[str] = None
    bars: Optional[int] = None
    energy: Optional[float] = None
    seed: Optional[int] = None
    audio_url: Optional[str] = None
    midi_url: Optional[str] = None
    video_url: Optional[str] = None
    cover_url: Optional[str] = None
    solana_wallet: Optional[str] = None
    solana_signature: Optional[str] = None
    audio_sha256: Optional[str] = None
    ai_prompt: Optional[str] = None
    created_at: Optional[str] = None


@app.post("/snowflake/song-metadata", response_model=dict)
def insert_song_metadata(payload: SongMetadataRowIn):
    """
    Insert into SONG_METADATA with all fields optional.
    If payload.id is None, Snowflake table default (UUID_STRING()) will generate it.
    Returns inserted id.
    """
    
    table = _qualified_song_table()
    conn = get_conn()

    try:
        # Build dynamic insert using only fields that were provided (not None).
        data = payload.model_dump(exclude_none=True)

        # If client didn't send id, don't include it, so Snowflake default runs.
        # If client did send id, include it.
        cols = list(data.keys())
        vals = [data[c] for c in cols]

        if not cols:
            # Insert an "empty" row to get defaults (id, created_at)
            sql = f"INSERT INTO {table} DEFAULT VALUES"
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()

            # Fetch the last inserted row id (best effort)
            with conn.cursor() as cur:
                cur.execute(f"SELECT id FROM {table} ORDER BY created_at DESC LIMIT 1")
                row = cur.fetchone()
            return {"ok": True, "id": row[0] if row else None}

        placeholders = ", ".join(["%s"] * len(cols))
        col_sql = ", ".join(cols)

        sql = f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})"
        with conn.cursor() as cur:
            cur.execute(sql, vals)
        conn.commit()

        # If client supplied id, return it. Otherwise fetch newest row id as best effort.
        if "id" in data and data["id"]:
            return {"ok": True, "id": data["id"]}

        with conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {table} ORDER BY created_at DESC LIMIT 1")
            row = cur.fetchone()

        return {"ok": True, "id": row[0] if row else None}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.get("/snowflake/song-metadata", response_model=dict)
def list_song_metadata(
    wallet_address: Optional[str] = Query(default=None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """
    List metadata rows. Optionally filter by solana_wallet.
    """
    table = _qualified_song_table()
    conn = get_conn()
    try:
        where = ""
        params: List[Any] = []

        if wallet_address:
            where = "WHERE solana_wallet = %s"
            params.append(wallet_address)

        sql = f"""
            SELECT
              id, name, artist, vibe, bpm, key, mode, bars, energy, seed,
              audio_url, midi_url, video_url, cover_url,
              solana_wallet, solana_signature,
              audio_sha256, ai_prompt,
              TO_VARCHAR(created_at)
            FROM {table}
            {where}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        items = []
        for r in rows:
            items.append({
                "id": r[0],
                "name": r[1],
                "artist": r[2],
                "vibe": r[3],
                "bpm": r[4],
                "key": r[5],
                "mode": r[6],
                "bars": r[7],
                "energy": r[8],
                "seed": r[9],
                "audio_url": r[10],
                "midi_url": r[11],
                "video_url": r[12],
                "cover_url": r[13],
                "solana_wallet": r[14],
                "solana_signature": r[15],
                "audio_sha256": r[16],
                "ai_prompt": r[17],
                "created_at": r[18],
            })

        return {"ok": True, "items": items, "limit": limit, "offset": offset}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.get("/snowflake/song-metadata/{row_id}", response_model=dict)
def get_song_metadata(row_id: str):
    """
    Fetch one row by id.
    """
    table = _qualified_song_table()
    conn = get_conn()
    try:
        sql = f"""
            SELECT
              id, name, artist, vibe, bpm, key, mode, bars, energy, seed,
              audio_url, midi_url, video_url, cover_url,
              solana_wallet, solana_signature,
              audio_sha256, ai_prompt,
              TO_VARCHAR(created_at)
            FROM {table}
            WHERE id = %s
            LIMIT 1
        """
        with conn.cursor() as cur:
            cur.execute(sql, (row_id,))
            r = cur.fetchone()

        if not r:
            raise HTTPException(status_code=404, detail="Not found")

        item = {
            "id": r[0],
            "name": r[1],
            "artist": r[2],
            "vibe": r[3],
            "bpm": r[4],
            "key": r[5],
            "mode": r[6],
            "bars": r[7],
            "energy": r[8],
            "seed": r[9],
            "audio_url": r[10],
            "midi_url": r[11],
            "video_url": r[12],
            "cover_url": r[13],
            "solana_wallet": r[14],
            "solana_signature": r[15],
            "audio_sha256": r[16],
            "ai_prompt": r[17],
            "created_at": r[18],
        }
        return {"ok": True, "item": item}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.get("/snowflake/song-metadata/by-wallet/{wallet}", response_model=dict)
def list_song_metadata_by_wallet(wallet: str, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    """
    Convenience: list for a wallet without query params.
    """
    return list_song_metadata(wallet_address=wallet, limit=limit, offset=offset)


SAVE_PRICE_LAMPORTS = int(float(os.getenv("SAVE_PRICE_SOL", "0.01")) * 1_000_000_000)
TREASURY = Pubkey.from_string(os.getenv("SOLANA_TREASURY_PUBKEY"))

def verify_payment(signature: str, payer: str):
    sig = Signature.from_string(signature)

    resp = SOLANA_CLIENT.get_transaction(
        sig,
        encoding="jsonParsed",
        commitment=Finalized,
        max_supported_transaction_version=0,
    )

    if not resp.value:
        raise HTTPException(status_code=400, detail="Transaction not found")

    meta = resp.value.transaction.meta
    if not meta:
        raise HTTPException(status_code=400, detail="Missing transaction meta")

    accounts = resp.value.transaction.transaction.message.account_keys

    try:
        payer_idx = accounts.index(Pubkey.from_string(payer))
        treasury_idx = accounts.index(TREASURY)
    except ValueError:
        raise HTTPException(status_code=400, detail="Wallet mismatch")

    received = meta.post_balances[treasury_idx] - meta.pre_balances[treasury_idx]

    if received < SAVE_PRICE_LAMPORTS:
        raise HTTPException(status_code=402, detail="Insufficient SOL payment")

    return True

from fastapi import HTTPException
from pydantic import BaseModel, Field
from fastapi import HTTPException
from solders.system_program import transfer, TransferParams
from solana.rpc.commitment import Confirmed

PAYOUT_SOL = float(os.getenv("PAYOUT_SOL", "0.01"))

class PayoutRequest(BaseModel):
    recipient_wallet: str = Field(..., description="User wallet to receive SOL")
    amount_sol: float = Field(default=PAYOUT_SOL, ge=0.000001, le=1.0)

@app.post("/solana/devnet/payout", response_model=dict)
def payout_user(req: PayoutRequest):
    """
    Sends SOL from your server wallet (SOLANA_PRIVATE_KEY_BASE58) to the user's wallet.
    """
    try:
        client = get_solana_client()
        kp = get_solana_keypair()
        from_pubkey = kp.pubkey()

        to_pubkey = Pubkey.from_string(req.recipient_wallet)

        lamports = int(req.amount_sol * 1_000_000_000)
        if lamports <= 0:
            raise HTTPException(status_code=400, detail="amount too small")

        # build transfer instruction
        ix = transfer(
            TransferParams(
                from_pubkey=from_pubkey,
                to_pubkey=to_pubkey,
                lamports=lamports,
            )
        )

        latest = client.get_latest_blockhash()
        if not latest.value:
            raise RuntimeError("Failed to fetch latest blockhash")
        bh = latest.value.blockhash

        msg = MessageV0.try_compile(
            payer=from_pubkey,
            instructions=[ix],
            address_lookup_table_accounts=[],
            recent_blockhash=bh,
        )
        tx = VersionedTransaction(msg, [kp])

        res = client.send_transaction(tx)
        if not res.value:
            raise RuntimeError(f"send_transaction failed: {res}")

        sig_str = str(res.value)

        # optional confirm
        client.confirm_transaction(Signature.from_string(sig_str), commitment=Confirmed)

        return {
            "ok": True,
            "from": str(from_pubkey),
            "to": str(to_pubkey),
            "amount_sol": req.amount_sol,
            "signature": sig_str,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel
from fastapi import HTTPException
import base64

@app.get("/solana/devnet/latest-blockhash", response_model=dict)
def latest_blockhash():
    try:
        client = get_solana_client()
        latest = client.get_latest_blockhash()
        if not latest.value:
            raise RuntimeError("No latest blockhash")
        return {"ok": True, "blockhash": str(latest.value.blockhash)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SendRawTxIn(BaseModel):
    tx_base64: str


@app.post("/solana/devnet/send-raw", response_model=dict)
def send_raw_tx(payload: SendRawTxIn):
    """
    Takes a Phantom-signed transaction (base64), broadcasts it, and confirms.
    """
    try:
        client = get_solana_client()
        raw = base64.b64decode(payload.tx_base64)

        # solana-py supports send_raw_transaction(bytes)
        res = client.send_raw_transaction(raw)
        if not res.value:
            raise RuntimeError(f"send_raw_transaction failed: {res}")

        sig_str = str(res.value)

        # confirm (best-effort)
        try:
            sig = Signature.from_string(sig_str)
            client.confirm_transaction(sig, commitment="confirmed")
        except Exception:
            pass

        return {"ok": True, "signature": sig_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import os

    # This pulls from environment variables, or defaults to 8005
    port = int(os.getenv("PORT", 8005))
    
    uvicorn.run(
        "app2:app",    # Changed from "app:app" to "app2:app"
        host="0.0.0.0",
        port=port,
        reload=True
    )