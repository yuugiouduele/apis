package main

import (
	"fmt"
	"log"
	"net"
	"strings"
)

func main() {
	// TCPリスナーを80番で作成（管理者権限必要な場合あり）
	ln, err := net.Listen("tcp", ":8080")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Server started at http://localhost:8080")

	for {
		conn, err := ln.Accept()
		if err != nil {
			log.Println("Accept error:", err)
			continue
		}

		go handleConnection(conn)
	}
}

func handleConnection(conn net.Conn) {
	defer conn.Close()

	// バッファ読み込み
	buf := make([]byte, 4096)
	n, err := conn.Read(buf)
	if err != nil {
		log.Println("Read error:", err)
		return
	}

	request := string(buf[:n])
	lines := strings.Split(request, "\r\n")
	if len(lines) > 0 {
		// 例：GET / HTTP/1.1
		requestLine := strings.Fields(lines[0])
		if len(requestLine) >= 2 {
			method, path := requestLine[0], requestLine
			fmt.Printf("Received %s request for %s\n", method, path)

			// 簡単なレスポンス作成
			body := "<html><body><h1>Hello from custom HTTP server!</h1></body></html>"
			resp := "HTTP/1.1 200 OK\r\n" +
				"Content-Type: text/html\r\n" +
				fmt.Sprintf("Content-Length: %d\r\n", len(body)) +
				"Connection: close\r\n\r\n" +
				body

			conn.Write([]byte(resp))
			return
		}
	}

	// 不正なリクエスト時
	conn.Write([]byte("HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n"))
}
