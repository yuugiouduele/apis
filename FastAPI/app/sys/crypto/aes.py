from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

def aes_encrypt(plain_text, key):
    cipher = AES.new(key, AES.MODE_CBC)
    iv = cipher.iv
    padded_text = pad(plain_text.encode(), AES.block_size)
    cipher_text = cipher.encrypt(padded_text)
    return iv + cipher_text  # IVを先頭に付加して返す

def aes_decrypt(cipher_text, key):
    iv = cipher_text[:AES.block_size]
    actual_cipher_text = cipher_text[AES.block_size:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_plain = cipher.decrypt(actual_cipher_text)
    plain_text = unpad(padded_plain, AES.block_size)
    return plain_text.decode()

# 128ビット（16バイト）のランダムキー生成
key = get_random_bytes(16)

plain_text = "Hello, AES encryption!"
encrypted = aes_encrypt(plain_text, key)
decrypted = aes_decrypt(encrypted, key)
