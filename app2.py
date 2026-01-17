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

load_dotenv()

app = FastAPI(title="Snowflake Events API")
app.include_router(mongo_router)


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

@app.get("/solana/devnet/memo/{signature}", response_model=dict)
def get_memo_from_signature(signature: str):
    """
    Fetch a devnet transaction by signature and extract Memo text.
    Works across solana-py/solders versions by trying multiple object paths.
    """
    try:
        client = get_solana_client()
        sig = Signature.from_string(signature)

        resp = client.get_transaction(
            sig,
            encoding="jsonParsed",
            max_supported_transaction_version=0,
        )

        if not resp.value:
            raise HTTPException(status_code=404, detail="Transaction not found (or not available yet).")

        v = resp.value

        # --- Try multiple known shapes to get log messages ---
        logs = None
        tried = []

        def try_get_logs(obj, path_name: str):
            nonlocal logs
            tried.append(path_name)
            if obj is None:
                return
            # solders uses snake_case: log_messages
            lm = getattr(obj, "log_messages", None)
            if isinstance(lm, list):
                logs = lm

        # Shape A: value.transaction.meta.log_messages
        tx = getattr(v, "transaction", None)
        try_get_logs(getattr(tx, "meta", None), "value.transaction.meta.log_messages")

        # Shape B: value.meta.log_messages
        if logs is None:
            try_get_logs(getattr(v, "meta", None), "value.meta.log_messages")

        # Shape C: value.transaction_status_meta.log_messages (some wrappers)
        if logs is None:
            try_get_logs(getattr(v, "transaction_status_meta", None), "value.transaction_status_meta.log_messages")

        # If still none, use JSON text if available (many solders objects expose to_json())
        logs_found = 0
        memos = []

        if logs is None:
            # Try to_json() on value (best chance to preserve structure)
            to_json = getattr(v, "to_json", None)
            if callable(to_json):
                import json as _json
                obj = _json.loads(to_json())
                # common JSON-RPC key
                logs = (((obj.get("meta") or {}).get("logMessages")) or [])
                tried.append("value.to_json()->meta.logMessages")
            else:
                logs = []
                tried.append("no logs path matched + no to_json()")

        logs_found = len(logs) if isinstance(logs, list) else 0

        if isinstance(logs, list):
            for line in logs:
                marker = "Program log: Memo"
                if marker in line and "): " in line:
                    memo_text = line.split("): ", 1)[1].strip()

                    # logs usually wrap memo in quotes
                    if memo_text.startswith('"') and memo_text.endswith('"'):
                        memo_text = memo_text[1:-1]

                    # unescape \" etc
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
            "log_lines_found": logs_found,
            "paths_tried": tried,
        }

    except HTTPException:
        raise
    except Exception as e:
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