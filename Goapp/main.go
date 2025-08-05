package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"runtime"
	"sync"
	"syscall"
	"time"
)

type Block struct {
	Index     int    `json:"index"`
	Timestamp string `json:"timestamp"`
	Data      string `json:"data"`
	Nonce     int    `json:"nonce"`
	PrevHash  string `json:"prev_hash"`
	Hash      string `json:"hash"`
}

func getMemoryUsage() string {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return fmt.Sprintf("Alloc = %v KB\nTotalAlloc = %v KB\nSys = %v KB\nNumGC = %v\n",
		m.Alloc/1024, m.TotalAlloc/1024, m.Sys/1024, m.NumGC)
}

func getInodeUsage(path string) string {
	var stat syscall.Statfs_t
	err := syscall.Statfs(path, &stat)
	if err != nil {
		return fmt.Sprintf("inode情報取得失敗: %v\n", err)
	}
	total := stat.Files
	free := stat.Ffree
	used := total - free
	return fmt.Sprintf("inode使用状況: 使用中=%d, 空き=%d, 合計=%d\n", used, free, total)
}

func randomData(length int) string {
	letters := []rune("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
	b := make([]rune, length)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return string(b)
}

func calculateHash(block Block) string {
	record := fmt.Sprintf("%d%s%s%d%s", block.Index, block.Timestamp, block.Data, block.Nonce, block.PrevHash)
	h := sha256.New()
	h.Write([]byte(record))
	return hex.EncodeToString(h.Sum(nil))
}

func generateBlock(i int, prevHash string, wg *sync.WaitGroup, ch chan<- Block) {
	defer wg.Done()
	data := randomData(10)
	nonce := rand.Intn(9000) + 1000 // 4桁
	block := Block{
		Index:     i,
		Timestamp: time.Now().Format(time.RFC3339Nano),
		Data:      data,
		Nonce:     nonce,
		PrevHash:  prevHash,
	}
	block.Hash = calculateHash(block)
	ch <- block
}

func generateBlockchainParallel(numBlocks int) []Block {
	rand.Seed(time.Now().UnixNano())
	var blockchain = make([]Block, numBlocks)
	var prevHash string
	var wg sync.WaitGroup

	for i := 0; i < numBlocks; i++ {
		wg.Add(1)
		go func(i int, prev string) {
			defer wg.Done()
			data := randomData(10)
			nonce := rand.Intn(9000) + 1000 // 4桁
			block := Block{
				Index:     i,
				Timestamp: time.Now().Format(time.RFC3339Nano),
				Data:      data,
				Nonce:     nonce,
				PrevHash:  prev,
			}
			block.Hash = calculateHash(block)
			blockchain[i] = block
		}(i, prevHash)
		// prevHashは直列でしか正しく繋がらないため、厳密なチェーンにはなりません
		prevHash = "" // 並列生成のため、前のハッシュは空でOK
	}
	wg.Wait()
	return blockchain
}

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// SSE (Server-Sent Events) でリアルタイム配信
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming unsupported!", http.StatusInternalServerError)
			return
		}
		// 10回分まとめて一気に送信
		for {
			var batch string
			for i := 0; i < 10; i++ {
				mem := getMemoryUsage()
				inode := getInodeUsage("/")
				now := time.Now().Format(time.RFC3339)
				batch += fmt.Sprintf("data: %s現在時刻: %s\n%s\n\n", mem, now, inode)
			}
			fmt.Fprint(w, batch)
			flusher.Flush()
			time.Sleep(1 * time.Second)
		}
	})

	http.HandleFunc("/blockchain", func(w http.ResponseWriter, r *http.Request) {
		blocks := generateBlockchainParallel(1000000)
		w.Header().Set("Content-Type", "text/event-stream; application/json; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		enc := json.NewEncoder(w)
		batchSize := 100

		for i := 0; i < len(blocks); i += batchSize {
			end := i + batchSize
			if end > len(blocks) {
				end = len(blocks)
			}
			batch := blocks[i:end]
			enc.Encode(batch)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			time.Sleep(2000 * time.Millisecond) // 0.5秒待機
		}
	})

	fmt.Println("サーバーを http://localhost:8080 で起動します")
	http.ListenAndServe(":8080", nil)
}