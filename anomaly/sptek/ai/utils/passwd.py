import hashlib
import string
from . import encrypt 
from string import ascii_letters
from string import digits
from string import punctuation
import secrets

def _random_key_(length=50):
    string_pool = ascii_letters + digits + punctuation
    key = ''.join(secrets.choice(string_pool) for _ in range(length))
    return key

def encrypto(raw, key, hash, random_key):
    if random_key == True:
        key = _random_key_()
        
    if hash == 'sha256':
        hash_key = hashlib.sha256(key.encode('utf-8')).hexdigest()
        
    enc_raw = encrypt.encrypt(raw, hash_key)    
    print(f"\ninput password: {raw}")
    print(f"encrypt password: {enc_raw.decode('utf-8')}")
    print(f"input key: {key}")
    print(f"{hash} key: {hash_key}\n")
    
def decrypto(raw, key):
    # raw = bytes(raw, encoding='utf-8')
    dec_raw = encrypt.decrypt(raw, key)
    print(f"\ninput password: {raw}")
    print(f"decrypt password: {dec_raw.decode('utf-8')}")
    print(f"input key: {key}")    
    