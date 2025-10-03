package main

import (
	"encoding/binary"
	"fmt"
	"net"
	"os"
	"syscall"
	// "time"
	"math/rand"
)

// DNSヘッダー構造
type DNSHeader struct {
	ID      uint16
	Flags   uint16
	QdCount uint16
	AnCount uint16
	NsCount uint16
	ArCount uint16
}

// DNS質問構造
type DNSQuestion struct {
	Name  []byte
	QType  uint16
	QClass uint16
}

func main() {
	serverIP := "8.8.8.8"
	serverPort := 53

	// ソケット作成 (UDP)
	fd, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_DGRAM, syscall.IPPROTO_UDP)
	if err != nil {
		fmt.Println("Socket creation failed:", err)
		os.Exit(1)
	}
	defer syscall.Close(fd)

	// DNSサーバーアドレスセット
	var addr syscall.SockaddrInet4
	addr.Port = serverPort
	copy(addr.Addr[:], net.ParseIP(serverIP).To4())

	// DNSクエリパケット作成
	domain := "example.com"
	queryPacket := buildDNSQuery(domain)

	// 送信
	if err := syscall.Sendto(fd, queryPacket, 0, &addr); err != nil {
		fmt.Println("Sendto failed:", err)
		os.Exit(1)
	}

	// タイムアウト設定
	syscall.SetsockoptTimeval(fd, syscall.SOL_SOCKET, syscall.SO_RCVTIMEO, &syscall.Timeval{Sec: 5, Usec: 0})

	// 受信バッファ
	buf := make([]byte, 512)
	n, from, err := syscall.Recvfrom(fd, buf, 0)
	if err != nil {
		fmt.Println("Recvfrom failed:", err)
		os.Exit(1)
	}

	fmt.Printf("Received %d bytes from %v\n", n, from)
	fmt.Printf("Raw response: %x\n", buf[:n])

	// DNSレスポンス解析は必要に応じて実装してください
}

// DNSクエリパケットを作成する関数
func buildDNSQuery(domain string) []byte {
	id := uint16(rand.Intn(65535))
	header := DNSHeader{
		ID:      id,
		Flags:   0x0100, // 標準クエリ(0)、再帰要求(1)
		QdCount: 1,
		AnCount: 0,
		NsCount: 0,
		ArCount: 0,
	}

	buf := make([]byte, 12) // DNSヘッダーのサイズは12バイト
	binary.BigEndian.PutUint16(buf[0:2], header.ID)
	binary.BigEndian.PutUint16(buf[2:4], header.Flags)
	binary.BigEndian.PutUint16(buf[4:6], header.QdCount)
	binary.BigEndian.PutUint16(buf[6:8], header.AnCount)
	binary.BigEndian.PutUint16(buf[8:10], header.NsCount)
	binary.BigEndian.PutUint16(buf[10:12], header.ArCount)

	// 質問セクションの作成(domainをラベル形式に変換)
	question := encodeDomainName(domain)
	qtype := make([]byte, 2)
	qclass := make([]byte, 2)
	binary.BigEndian.PutUint16(qtype, 1)   // QTYPE A (IPv4)
	binary.BigEndian.PutUint16(qclass, 1)  // QCLASS IN

	packet := append(buf, question...)
	packet = append(packet, qtype...)
	packet = append(packet, qclass...)

	return packet
}

// ドット区切りのドメイン名をDNSラベル形式に変換
// 例: example.com -> [7]'example''com'
func encodeDomainName(domain string) []byte {
	parts := []byte{}
	for _, label := range splitLabels(domain) {
		parts = append(parts, byte(len(label)))
		parts = append(parts, []byte(label)...)
	}
	parts = append(parts, 0) // 終端
	return parts
}

func splitLabels(domain string) []string {
	res := []string{}
	start := 0
	for i := 0; i < len(domain); i++ {
		if domain[i] == '.' {
			res = append(res, domain[start:i])
			start = i + 1
		}
	}
	res = append(res, domain[start:])
	return res
}
