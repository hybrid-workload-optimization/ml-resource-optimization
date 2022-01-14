import base64
import crypt

from Cryptodome import Random
from Cryptodome import Cipher
from Cryptodome.Cipher import AES
from Cryptodome.Protocol.KDF import PBKDF2

BLOCK_SIZE = 16
PAD = lambda s: s + (BLOCK_SIZE - len(s) % BLOCK_SIZE) * chr(BLOCK_SIZE - len(s) % BLOCK_SIZE)
UNPAD = lambda s: s[:-ord(s[len(s) - 1:])]

def get_private_key(random):
    salt = "|L-Dx#cm~lETHik1(_[8XG@*_!bYnh?@3#An*T7+VY{QLpS9-]"
    kdf = PBKDF2(random, salt, 64, 1000)
    key = kdf[:32]
    return key

def encrypt(raw, random):
    private_key = get_private_key(random)
    raw = PAD(raw).encode('utf-8')
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(private_key, AES.MODE_CBC, iv)
    return base64.b64encode(iv + cipher.encrypt(raw))

def decrypt(enc, random):
    private_key = get_private_key(random)
    enc = base64.b64decode(enc)
    iv = enc[:16]
    cipher = AES.new(private_key, AES.MODE_CBC, iv)
    return UNPAD(cipher.decrypt(enc[16:]))