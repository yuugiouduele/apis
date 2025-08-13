import "time"

func retransmit(conn *net.UDPConn, msg []byte, addr *net.UDPAddr, maxRetries int, interval time.Duration) {
    retries := 0
    for retries < maxRetries {
        _, err := conn.WriteToUDP(msg, addr)
        if err != nil {
            fmt.Println("Send error:", err)
            return
        }
        fmt.Println("Sent message, retry:", retries)
        time.Sleep(interval)
        retries++
        // 実際はACK受信確認実装と連動させる
    }
    fmt.Println("Max retries reached, giving up retransmission")
}
