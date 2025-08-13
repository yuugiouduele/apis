package main

import (
    "fmt"
    "net"
)

func main() {
    addr := net.UDPAddr{
        Port: 8000,
        IP:   net.ParseIP("0.0.0.0"),
    }

    conn, err := net.ListenUDP("udp", &addr)
    if err != nil {
        panic(err)
    }
    defer conn.Close()

    fmt.Println("RTP Server listening on", addr.String())

    buffer := make([]byte, 1500)
    for {
        n, clientAddr, err := conn.ReadFromUDP(buffer)
        if err != nil {
            fmt.Println("Read error:", err)
            continue
        }
        fmt.Printf("Received RTP packet from %v, size %d bytes\n", clientAddr, n)

        // ここでRTPヘッダー解析や処理が可能
        // 送信例:
        // conn.WriteToUDP(buffer[:n], clientAddr)
    }
}
