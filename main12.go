package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/dgrijalva/jwt-go/v4"
)

// JWTの署名に使用するシークレットキー
var jwtSecretKey = []byte("your-super-secret-key")

// PayloadはWebhookで受け取るデータの構造を定義します
type Payload struct {
	Event string `json:"event"`
	Data  string `json:"data"`
}

// 非同期処理用のチャネル
var payloadQueue = make(chan Payload, 100)

// handleWebhookはWebhookリクエストを処理する主要なハンドラです
func handleWebhook(w http.ResponseWriter, r *http.Request) {
	// 1. リクエストメソッドの検証
	if r.Method != http.MethodPost {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	// 2. JWT署名の検証
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		http.Error(w, "Authorization header is missing", http.StatusUnauthorized)
		return
	}

	// "Bearer "プレフィックスを削除
	tokenString := strings.Split(authHeader, " ")[1]
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		// 署名アルゴリズムの検証 (例: HS256)
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return jwtSecretKey, nil
	})

	if err != nil || !token.Valid {
		http.Error(w, "Invalid token", http.StatusUnauthorized)
		return
	}

	// 3. ペイロードのバリデーションと非同期処理
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read body", http.StatusInternalServerError)
		return
	}

	var payload Payload
	if err := json.Unmarshal(body, &payload); err != nil {
		http.Error(w, "Invalid JSON payload", http.StatusBadRequest)
		return
	}

	// バリデーションの例: eventフィールドが空でないか確認
	if payload.Event == "" {
		http.Error(w, "Payload validation failed: 'event' field is empty", http.StatusBadRequest)
		return
	}

	// 非同期処理キューにペイロードを送信
	payloadQueue <- payload

	fmt.Fprintf(w, "Webhook received and queued successfully!")
}

// workerはキューからペイロードを取り出して非同期で処理します
func worker(ctx context.Context) {
	for {
		select {
		case p := <-payloadQueue:
			// ここで実際の非同期処理を行う
			log.Printf("Processing payload: Event='%s', Data='%s'", p.Event, p.Data)
			time.Sleep(2 * time.Second) // 処理に時間がかかることをシミュレート
			log.Printf("Finished processing event '%s'", p.Event)
		case <-ctx.Done():
			// アプリケーション終了時にワーカーを停止する
			return
		}
	}
}

func main() {
	// ワーカーを1つ起動
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go worker(ctx)

	// Webhookハンドラの登録
	http.HandleFunc("/webhook", handleWebhook)

	fmt.Println("Webhook server is running on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
