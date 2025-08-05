package auth

import (
    "crypto/sha256"
    "encoding/hex"
    "time"
)

// Blockはブロックチェーンの1ブロック
type Block struct {
    Index        int
    Timestamp    int64
    Data         string
    PrevHash     string
    Hash         string
    BookingOrder int // ハッシュ化の順序制御
    Layer        int // 階層化
}

// Blockchainは階層化されたブロックチェーン
type Blockchain struct {
    Blocks      [][]Block // Layerごとにブロック配列
    Parameters  map[string]interface{}
}

// NewBlockを生成
func NewBlock(index int, data, prevHash string, bookingOrder, layer int) Block {
    timestamp := time.Now().Unix()
    hash := CalculateHash(index, timestamp, data, prevHash, bookingOrder, layer)
    return Block{
        Index:        index,
        Timestamp:    timestamp,
        Data:         data,
        PrevHash:     prevHash,
        Hash:         hash,
        BookingOrder: bookingOrder,
        Layer:        layer,
    }
}

// ハッシュ計算（BookingOrderで順序制御）
func CalculateHash(index int, timestamp int64, data, prevHash string, bookingOrder, layer int) string {
    input := ""
    switch bookingOrder {
    case 0:
        input = data + prevHash + string(index) + string(timestamp) + string(layer)
    case 1:
        input = prevHash + data + string(layer) + string(index) + string(timestamp)
    default:
        input = string(layer) + data + prevHash + string(index) + string(timestamp)
    }
    hash := sha256.Sum256([]byte(input))
    return hex.EncodeToString(hash[:])
}

// 新しいBlockchainを生成
func NewBlockchain(params map[string]interface{}, layers int) *Blockchain {
    blocks := make([][]Block, layers)
    for i := 0; i < layers; i++ {
        genesis := NewBlock(0, "Genesis", "", 0, i)
        blocks[i] = []Block{genesis}
    }
    return &Blockchain{
        Blocks:     blocks,
        Parameters: params,
    }
}

// ブロック追加
func (bc *Blockchain) AddBlock(data string, layer, bookingOrder int) {
    prevBlock := bc.Blocks[layer][len(bc.Blocks[layer])-1]
    newBlock := NewBlock(prevBlock.Index+1, data, prevBlock.Hash, bookingOrder, layer)
    bc.Blocks[layer] = append(bc.Blocks[layer], newBlock)
}