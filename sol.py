import os
import base58
from dotenv import load_dotenv
from solana.rpc.api import Client
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import transfer, TransferParams
from solders.message import MessageV0
from solders.transaction import VersionedTransaction

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Connection
# Using Helius because the default Solana RPC was resetting your connection
RPC_URL = "https://api.devnet.solana.com"
client = Client(RPC_URL)

def send_payment():
    # --- Load Data ---
    secret_str = os.getenv("SOLANA_PRIVATE_KEY_BASE58")
    if not secret_str:
        print("❌ Error: SOLANA_PRIVATE_KEY_BASE58 not found in .env")
        return

    sender_keypair = Keypair.from_bytes(base58.b58decode(secret_str))
    
    # Using your previous test recipient
    receiver_addr = "G36NVaeDo7LdJcKchwDXSw37uzUUonSH5CQSbR6Ny5ND"
    receiver_pubkey = Pubkey.from_string(receiver_addr)
    
    # 0.01 SOL = 10,000,000 Lamports
    amount_lamports = 10_000_000 

    print(f"Sender: {sender_keypair.pubkey()}")
    print(f"Receiver: {receiver_pubkey}")

    # --- Build Transaction ---
    ix = transfer(TransferParams(
        from_pubkey=sender_keypair.pubkey(),
        to_pubkey=receiver_pubkey,
        lamports=amount_lamports
    ))

    # Get fresh blockhash
    recent_blockhash = client.get_latest_blockhash().value.blockhash

    # Assemble modern Versioned Transaction
    message = MessageV0.try_compile(
        payer=sender_keypair.pubkey(),
        instructions=[ix],
        address_lookup_table_accounts=[],
        recent_blockhash=recent_blockhash
    )
    tx = VersionedTransaction(message, [sender_keypair])

    # --- Send ---
    try:
        response = client.send_transaction(tx)
        print(f"✅ Success! Signature: {response.value}")
        print(f"🔗 View on Explorer: https://explorer.solana.com/tx/{response.value}?cluster=devnet")
    except Exception as e:
        print(f"❌ Transaction Failed: {e}")

if __name__ == "__main__":
    send_payment()