import (
    "crypto/md5"
    "fmt"
    "io"
)

// MD5ハッシュ計算のユーティリティ
func MD5Hash(data string) string {
    h := md5.New()
    io.WriteString(h, data)
    return fmt.Sprintf("%x", h.Sum(nil))
}

// Digest認証レスポンス生成例
// H(A1) = MD5(username:realm:password)
// H(A2) = MD5(method:uri)
// response = MD5(H(A1):nonce:H(A2))
func ComputeDigestResponse(username, realm, password, method, uri, nonce string) string {
    HA1 := MD5Hash(fmt.Sprintf("%s:%s:%s", username, realm, password))
    HA2 := MD5Hash(fmt.Sprintf("%s:%s", method, uri))
    response := MD5Hash(fmt.Sprintf("%s:%s:%s", HA1, nonce, HA2))
    return response
}
