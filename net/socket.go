package net

import (
	"bytes"
	"context"
	"errors"
	"log"
	"net/http"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/websocket"
)
// ==================== 認証サービス (JWT) ====================


// ==================== HTTPハンドラ ====================

func (s *WebSocketServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// 初回接続時にトークン発行
	userID := r.URL.Query().Get("user_id")
	if userID == "" {
		http.Error(w, "user_id required", http.StatusBadRequest)
		return
	}
	token, err := s.auth.GenerateToken(userID)
	if err != nil {
		http.Error(w, "token generation error", http.StatusInternalServerError)
		return
	}

	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("Upgrade error:", err)
		return
	}
	s.conn = conn
	defer s.conn.Close()

	// 接続成功時にクライアントへトークン送信
	if err := s.conn.WriteMessage(websocket.TextMessage, []byte("AUTH:"+token)); err != nil {
		log.Println("send token error:", err)
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go s.readLoop(ctx)         // 受信とトークン検証
	go s.batchSendLoop(ctx)    // バッチ送信
	go s.metricsMonitorLoop(ctx) // メトリクス監視

	<-ctx.Done()
}

// ==================== 受信処理 ====================

func (s *WebSocketServer) readLoop(ctx context.Context) {
	for {
		_, msg, err := s.conn.ReadMessage()
		if err != nil {
			log.Println("read error:", err)
			break
		}
		if !s.checkMessageAuth(msg) {
			log.Println("Unauthorized message, closing connection")
			_ = s.conn.WriteMessage(websocket.TextMessage, []byte("ERROR: Unauthorized"))
			break
		}
		log.Printf("Verified message: %s\n", msg)
	}
}

// メッセージ1個につき「TOKEN:xxxxx;BODY:...」形式を想定
func (s *WebSocketServer) checkMessageAuth(msg []byte) bool {
	parts := bytes.SplitN(msg, []byte(";BODY:"), 2)
	if len(parts) != 2 {
		return false
	}
	token := bytes.TrimPrefix(parts[0], []byte("TOKEN:"))
	_, err := s.auth.ValidateToken(string(token))
	return err == nil
}

// ==================== バッチ送信ループ ====================

func (s *WebSocketServer) batchSendLoop(ctx context.Context) {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if !s.batchMode.Load() {
				continue
			}
			s.mu.Lock()
			if len(s.sendBuf) > 0 {
				merged := bytes.Join(s.sendBuf, []byte{'\n'})
				if err := s.conn.WriteMessage(websocket.TextMessage, merged); err != nil {
					log.Println("batch write error:", err)
					s.mu.Unlock()
					return
				}
				atomic.AddInt32(&s.packetCount, 1)
				s.sendBuf = s.sendBuf[:0]
			}
			s.mu.Unlock()
		}
	}
}

// ==================== 外部送信インタフェース ====================

func (s *WebSocketServer) EnqueueMessage(msg []byte) error {
	if s.batchMode.Load() {
		s.mu.Lock()
		s.sendBuf = append(s.sendBuf, msg)
		bufLen := len(s.sendBuf)
		s.mu.Unlock()
		if bufLen > s.bufThreshold {
			s.trySwitchMode(false)
		}
		return nil
	}
	err := s.conn.WriteMessage(websocket.TextMessage, msg)
	if err == nil {
		atomic.AddInt32(&s.packetCount, 1)
	}
	return err
}
// ==================== メトリクス監視ループ ==================
// ==================== モード切替 ====================

func (s *WebSocketServer) trySwitchMode(batch bool) {
	now := time.Now()
	if now.Sub(s.lastSwitch) < s.switchCooldown {
		return
	}
	if s.batchMode.Load() == batch {
		return
	}
	s.batchMode.Store(batch)
	s.lastSwitch = now
	log.Printf("[ModeSwitch] BatchMode changed to %v", batch)
}