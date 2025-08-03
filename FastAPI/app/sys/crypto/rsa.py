from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Random import get_random_bytes

# 鍵ペア生成（2048ビット）
key = RSA.generate(2048)
private_key = key.export_key()
public_key = key.publickey().export_key()

# 鍵の読み込み
private_key_obj = RSA.import_key(private_key)
public_key_obj = RSA.import_key(public_key)

# 暗号化用オブジェクト
cipher_rsa_enc = PKCS1_OAEP.new(public_key_obj)

# 復号用オブジェクト
cipher_rsa_dec = PKCS1_OAEP.new(private_key_obj)

message = b"Hello RSA encryption!"
cipher_text = cipher_rsa_enc.encrypt(message)
plain_text = cipher_rsa_dec.decrypt(cipher_text)

