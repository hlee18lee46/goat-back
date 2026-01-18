#solana.py
import os, json, time, math
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator
# ✅ Load .env for standalone tests
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)

_SOLANA_ENABLED = False
try:
    from base58 import b58decode
    from solana.rpc.api import Client
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solders.system_program import transfer, TransferParams
    from solders.transaction import Transaction
    from solders.hash import Hash
    from solders.message import Message

    _SOLANA_ENABLED = True
    print("✅ Solana SDK imports successful")
except Exception as e:
    print("❌ Solana SDK import failed:", e)
    _SOLANA_ENABLED = False

# ---- Storage ----
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
OBSTACLES_JSONL = DATA_DIR / "obstacles.jsonl"
REWARDS_INDEX   = DATA_DIR / "rewards_index.json"

# ---- Config ----
REQUIRE_LOCATION       = os.getenv("SOLANA_REQUIRE_LOCATION", "true").lower() != "false"
MIN_DISTANCE_METERS    = float(os.getenv("SOLANA_MIN_DISTANCE_METERS", "50"))
COOLDOWN_SECONDS       = int(os.getenv("SOLANA_COOLDOWN_SECONDS", "600"))
MAX_REWARDS_PER_DAY    = int(os.getenv("SOLANA_MAX_REWARDS_PER_DAY", "100"))
# Optional tool-level GPS fallback (belt & suspenders)
USE_FALLBACK_GPS       = os.getenv("SOLANA_USE_FALLBACK_GPS", "false").lower() == "true"
DEF_LAT                = float(os.getenv("DEFAULT_LATITUDE", "33.7756"))
DEF_LON                = float(os.getenv("DEFAULT_LONGITUDE", "-84.3963"))
DEF_ACC                = float(os.getenv("DEFAULT_ACCURACY_M", "10"))

def _now() -> int: return int(time.time())
def _append_jsonl(p: Path, obj: dict): p.open("a", encoding="utf-8").write(json.dumps(obj, ensure_ascii=False)+"\n")
def _load_index() -> dict:
    try: return json.loads(REWARDS_INDEX.read_text(encoding="utf-8"))
    except Exception: return {}
def _save_index(idx: dict): REWARDS_INDEX.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")

def _haversine_m(lat1, lon1, lat2, lon2):
    R=6371000.0; from math import radians,sin,cos,atan2,sqrt
    dlat=radians(lat2-lat1); dlon=radians(lon2-lon1)
    a=sin(dlat/2)**2+cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1-a))

def _midnight_epoch(ts:int)->int: return ts - (ts % 86400)
def _valid_location(lat: Optional[float], lng: Optional[float]) -> bool: return lat is not None and lng is not None

class ObstacleReport(BaseModel):
    description: str = Field(..., description="Short obstacle description")
    user_id: Optional[str] = Field(None, description="User identifier/email")
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    accuracy_m: Optional[float] = Field(None, ge=0, description="GPS accuracy (m)")
    image_data_url: Optional[str] = None
    image_url: Optional[str] = None
    solana_recipient: Optional[str] = Field(None, description="Recipient Solana address (base58)")

    @field_validator("solana_recipient")
    @classmethod
    def _soft_check_b58(cls, v):
        if not v: return v
        try:
            import base58 as _b58
            _b58.b58decode(v)
        except Exception:
            # Soft validation only; runtime path will fail if invalid.
            return v
        return v

def _try_solana_reward(recipient: str) -> str | None:
    try:
        rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com")
        secret_b58 = os.getenv("SOLANA_MERCHANT_SECRET_KEY")
        lamports = int(os.getenv("SOLANA_REWARD_LAMPORTS", "5000"))

        if not (secret_b58 and recipient):
            print("⚠️  Missing key or recipient")
            return None

        client = Client(rpc_url)
        payer = Keypair.from_bytes(b58decode(secret_b58))
        to_pub = Pubkey.from_string(recipient)

        # --- Build transfer instruction ---
        ix = transfer(TransferParams(from_pubkey=payer.pubkey(),
                                     to_pubkey=to_pub,
                                     lamports=lamports))

        # --- Fetch a recent blockhash ---
        bh = client.get_latest_blockhash().value.blockhash

        # --- Build and sign transaction correctly ---
        msg = Message([ix], payer.pubkey())
        tx = Transaction([payer], msg, bh)

        # --- Send transaction ---
        sig = client.send_transaction(tx).value

        print(f"✅ Solana reward sent! https://explorer.solana.com/tx/{sig}?cluster=devnet")
        return sig
    except Exception as e:
        print(f"❌ Solana reward error: {e}")
        return None
def _can_reward(idx:dict, key:str, lat:float, lng:float, now_ts:int)->tuple[bool,str]:
    st=idx.get(key,{}); last_ts=st.get("last_ts"); last_lat=st.get("last_lat"); last_lng=st.get("last_lng")
    day_key=str(_midnight_epoch(now_ts)); day_counts=st.get("day_counts",{}); today=int(day_counts.get(day_key,0))
    if today>=MAX_REWARDS_PER_DAY: return False, f"Daily reward cap reached ({MAX_REWARDS_PER_DAY})."
    if last_ts is not None and (now_ts-int(last_ts))<COOLDOWN_SECONDS:
        return False, f"Cooldown active. Try again in {COOLDOWN_SECONDS-(now_ts-int(last_ts))}s."
    if last_lat is not None and last_lng is not None:
        dist=_haversine_m(float(last_lat), float(last_lng), lat, lng)
        if dist<MIN_DISTANCE_METERS: return False, f"Move at least {int(MIN_DISTANCE_METERS)}m (moved ~{int(dist)}m)."
    return True, "ok"

def _record_reward(idx:dict, key:str, lat:float, lng:float, now_ts:int):
    st=idx.get(key,{})
    st.update({"last_ts":now_ts,"last_lat":lat,"last_lng":lng})
    day_key=str(_midnight_epoch(now_ts)); dc=st.get("day_counts",{}); dc[day_key]=int(dc.get(day_key,0))+1
    st["day_counts"]=dc; idx[key]=st; _save_index(idx)

def report_obstacle(
    description: str,
    user_id: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    accuracy_m: Optional[float] = None,
    image_data_url: Optional[str] = None,
    image_url: Optional[str] = None,
    solana_recipient: Optional[str] = None,
) -> str:
    ts=_now()

    # Optional tool-level GPS fallback
    if not _valid_location(latitude, longitude) and USE_FALLBACK_GPS:
        latitude, longitude, accuracy_m = DEF_LAT, DEF_LON, accuracy_m or DEF_ACC

    # Persist report (truncate big data URLs)
    _append_jsonl(OBSTACLES_JSONL, {
        "ts": ts, "user_id": user_id, "description": description.strip(),
        "lat": latitude, "lng": longitude, "accuracy_m": accuracy_m,
        "image_data_url": (image_data_url[:64]+"...") if image_data_url else None,
        "image_url": image_url, "solana_recipient": solana_recipient,
    })

    # Require GPS for rewards?
    if REQUIRE_LOCATION and not _valid_location(latitude, longitude):
        return "Obstacle saved. Location required for SOL reward."

    # Choose recipient
    recipient = solana_recipient or os.getenv("SOLANA_USER_PUBLIC_KEY")
    if not recipient:
        return "Obstacle saved. No Solana recipient configured (set SOLANA_USER_PUBLIC_KEY)."

    # SDK?
    if not _SOLANA_ENABLED:
        return "Obstacle saved. Solana reward not sent (solana SDK not installed/configured)."

    # Anti-spam gates
    if not _valid_location(latitude, longitude):
        return "Obstacle saved. Provide latitude/longitude to receive a SOL reward."
    idx=_load_index(); reporter_key=(user_id or recipient).strip()
    ok, reason = _can_reward(idx, reporter_key, float(latitude), float(longitude), ts)
    if not ok: return f"Obstacle saved. No reward: {reason}"

    # Send reward
    sig=_try_solana_reward(recipient)
    if sig:
        _record_reward(idx, reporter_key, float(latitude), float(longitude), ts)
        sig_str = str(sig)
        return f"Obstacle saved. SOL reward sent to {recipient[:8]}… Signature: {sig_str[:16]}…"
    return "Obstacle saved. Reward attempt failed or not configured properly."