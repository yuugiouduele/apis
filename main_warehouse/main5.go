package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
)

func generateWav(filename string) error {
	sampleRate := 16000
	durationSec := 10
	numSamples := sampleRate * durationSec
	pcmData := make([]int16, numSamples)

	formantFreqs := []float64{500, 900, 2400}    // F1, F2, F3 を低めに設定（日本語「お」の例に寄せる）
    amplitudes := []float64{0.8, 0.5, 0.3}

    baseFreq := 120.0 // 基音周波数（声帯の振動数）を低めに設定（男性の声は約85〜180Hz）

    for i := 0; i < numSamples; i++ {
        value := 0.0
        t := float64(i) / float64(sampleRate)
        
        // 基音（ピッチ）の波形 (声帯の振動)
        pitch := math.Sin(2.0 * math.Pi * baseFreq * t)

        // 各フォルマントのサイン波を合成（声道の共鳴）
        for j, f := range formantFreqs {
            value += amplitudes[j] * math.Sin(2.0*math.Pi*f*t)
        }
        
        // 基音とフォルマントの乗算で声質を模倣
        value *= pitch
        
        // 音量調整
        value *= 800

        pcmData[i] = int16(value)
    }



	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	file.WriteString("RIFF")
	binary.Write(file, binary.LittleEndian, uint32(36+numSamples*2))
	file.WriteString("WAVE")

	file.WriteString("fmt ")
	binary.Write(file, binary.LittleEndian, uint32(16))
	binary.Write(file, binary.LittleEndian, uint16(1)) // PCM
	binary.Write(file, binary.LittleEndian, uint16(1)) // モノラル
	binary.Write(file, binary.LittleEndian, uint32(sampleRate))
	byteRate := sampleRate * 2
	binary.Write(file, binary.LittleEndian, uint32(byteRate))
	blockAlign := uint16(2)
	binary.Write(file, binary.LittleEndian, blockAlign)
	binary.Write(file, binary.LittleEndian, uint16(16))

	file.WriteString("data")
	binary.Write(file, binary.LittleEndian, uint32(numSamples*2))

	for _, sample := range pcmData {
		binary.Write(file, binary.LittleEndian, sample)
	}

	return nil
}

func audioHandler(w http.ResponseWriter, r *http.Request) {
	filename := "temp_output.wav"

	err := generateWav(filename)
	if err != nil {
		http.Error(w, "Failed to generate WAV", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("Content-Disposition", "inline; filename=output.wav")

	data, err := os.ReadFile(filename)
	if err != nil {
		http.Error(w, "Failed to read WAV", http.StatusInternalServerError)
		return
	}

	_, err = w.Write(data)
	if err != nil {
		log.Println("Failed to write response:", err)
	}

	err = os.Remove(filename)
	if err != nil {
		log.Println("Failed to remove wav file:", err)
	}
}

func main() {
	http.HandleFunc("/audio", audioHandler)
	fmt.Println("Server started at http://localhost:8080/audio")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
