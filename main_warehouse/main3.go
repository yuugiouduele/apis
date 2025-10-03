package main

import (
    "bufio"
    "fmt"
    "strings"
)

// SIPRequestはSIPリクエストメッセージの構造体例
type SIPRequest struct {
    Method  string            // INVITE, ACK, BYE, REGISTERなど
    URI     string            // 送信先URI
    Version string            // SIPプロトコルバージョン
    Headers map[string]string // ヘッダーのマップ
    Body    string            // メッセージボディ（SDPなど）
}

// ParseSIPRequestは文字列をパースしSIPRequestに変換
func ParseSIPRequest(msg string) *SIPRequest {
    request := &SIPRequest{Headers: make(map[string]string)}
    scanner := bufio.NewScanner(strings.NewReader(msg))
    
    // リクエストラインの解析（例: "INVITE sip:bob@domain.com SIP/2.0"）
    if scanner.Scan() {
        parts := strings.Split(scanner.Text(), " ")
        if len(parts) == 3 {
            request.Method = parts[0]
            request.URI = parts[1]
            request.Version = parts[2]
        }
    }
    
    // ヘッダー読み取り
    for scanner.Scan() {
        line := scanner.Text()
        if len(line) == 0 {
            // 空行＝ヘッダー終了
            break
        }
        kv := strings.SplitN(line, ":", 2)
        if len(kv) == 2 {
            key := strings.TrimSpace(kv[0])
            value := strings.TrimSpace(kv[1])
            request.Headers[key] = value
        }
    }

    // ボディ読み取り（必要に応じて拡張可能）
    bodyLines := []string{}
    for scanner.Scan() {
        bodyLines = append(bodyLines, scanner.Text())
    }
    request.Body = strings.Join(bodyLines, "\n")
    
    return request
}

func main() {
    msg := "INVITE sip:bob@domain.com SIP/2.0\r\nTo: Bob <sip:bob@domain.com>\r\nFrom: Alice <sip:alice@domain.com>\r\nCall-ID: 123456\r\nCSeq: 1 INVITE\r\nContent-Length: 0\r\n\r\n"
    req := ParseSIPRequest(msg)
    
    fmt.Printf("Method: %s\nURI: %s\nHeaders:\n", req.Method, req.URI)
    for k, v := range req.Headers {
        fmt.Printf("  %s: %s\n", k, v)
    }
    fmt.Printf("Body:\n%s\n", req.Body)
}
