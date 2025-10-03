package main

import (
	"encoding/binary"
	"log"
	"math"
	"math/rand"
	"net/http"
	"time"
)

func generateHowlWav() ([]byte, error) {
	sampleRate := 16000
	durationSec := 1
	numSamples := sampleRate * durationSec
	pcmData := make([]int16, numSamples)
	rand.Seed(time.Now().UnixNano())

	baseFreq := 150.0
	formantFreqs := []float64{2800,3700,1800}
	amplitudes := []float64{0.7, 0.5, 0.2}

	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		envelope := math.Exp(-8 * t) // アタック音

		pitch := baseFreq + rand.Float64()*30 - 15
		value := math.Sin(2.0 * math.Pi * pitch * t)

		for j, f := range formantFreqs {
			value += amplitudes[j] * math.Sin(2.0*math.Pi*f*t)
		}

		value += 0.1 * (rand.Float64()*2 - 1)
		value *= envelope
		value *= 12000

		if value > 32767 {
			value = 32767
		}
		if value < -32768 {
			value = -32768
		}

		pcmData[i] = int16(value)
	}

	buf := make([]byte, 44+numSamples*2) // WAVヘッダー44バイト + PCMデータ
	// RIFFヘッダー
	copy(buf[0:], []byte("RIFF"))
	binary.LittleEndian.PutUint32(buf[4:], uint32(36+numSamples*2)) // ファイルサイズ-8
	copy(buf[8:], []byte("WAVE"))

	// fmtチャンク
	copy(buf[12:], []byte("fmt "))
	binary.LittleEndian.PutUint32(buf[16:], 16)        // fmtチャンクサイズ
	binary.LittleEndian.PutUint16(buf[20:], 1)         // PCMフォーマット
	binary.LittleEndian.PutUint16(buf[22:], 1)         // モノラル
	binary.LittleEndian.PutUint32(buf[24:], uint32(sampleRate))
	binary.LittleEndian.PutUint32(buf[28:], uint32(sampleRate*2)) // byteRate
	binary.LittleEndian.PutUint16(buf[32:], 2)                    // blockAlign
	binary.LittleEndian.PutUint16(buf[34:], 16)                   // bitsPerSample

	// dataチャンク
	copy(buf[36:], []byte("data"))
	binary.LittleEndian.PutUint32(buf[40:], uint32(numSamples*2)) // データチャンクサイズ

	// PCMデータ書き込み
	offset := 44
	for _, sample := range pcmData {
		binary.LittleEndian.PutUint16(buf[offset:], uint16(sample))
		offset += 2
	}

	return buf, nil
}

func audioHandler(w http.ResponseWriter, r *http.Request) {
	wavData, err := generateHowlWav()
	if err != nil {
		http.Error(w, "Failed to generate audio", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("Content-Length", string(len(wavData)))
	w.Write(wavData)
}

func main() {
	http.HandleFunc("/audio", audioHandler)
	log.Println("Server running at http://localhost:8080/audio")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
