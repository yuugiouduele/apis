package main

import (
	"encoding/binary"
	"log"
	"math"
	"math/rand"
	"net/http"
	"time"
)

// メジャースケール
var scale = []float64{
	261.63, // C4
	293.66, // D4
	329.63, // E4
	349.23, // F4
	392.00, // G4
	440.00, // A4
	493.88, // B4
	523.25, // C5
}

func generateOrchestraWav() ([]byte, error) {
	sampleRate := 16000
	durationSec := 30 // 30秒
	numSamples := sampleRate * durationSec
	pcmData := make([]int16, numSamples)
	rand.Seed(time.Now().UnixNano())

	// リズム: 1小節=2.5秒, 12小節で30秒
	beatsPerBar := 4
	barDuration := 2.5
	beatDuration := barDuration / float64(beatsPerBar)

	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)

		// 小節・拍の位置
		bar := int(t / barDuration)
		beat := int((t - float64(bar)*barDuration) / beatDuration)

		// メロディ
		melodyFreq := scale[(bar+beat)%len(scale)]
		// コード構成音（和音）
		chordRoot := scale[(bar*2)%len(scale)]
		chordThird := chordRoot * math.Pow(2, 4.0/12.0) // 長三度
		chordFifth := chordRoot * math.Pow(2, 7.0/12.0) // 完全五度

		// 弦楽器のような滑らかなエンベロープ
		localT := t - float64(bar)*barDuration - float64(beat)*beatDuration
		attack := 0.2
		decay := 0.4
		sustain := 0.7
		release := 0.2
		env := 1.0
		if localT < attack {
			env = localT / attack
		} else if localT < attack+decay {
			env = 1.0 - (localT-attack)*(1.0-sustain)/decay
		} else if localT < beatDuration-release {
			env = sustain
		} else {
			env = sustain * (1 - (localT-(beatDuration-release))/release)
		}

		// 音色（弦 + 木管 + 金管風）
		value := 0.5*math.Sin(2*math.Pi*melodyFreq*t) +         // 弦（バイオリン風）
			0.3*math.Sin(2*math.Pi*chordRoot*t) +                 // 金管（トランペット風）
			0.25*math.Sin(2*math.Pi*chordThird*t) +               // 和音の3度
			0.25*math.Sin(2*math.Pi*chordFifth*t) +               // 和音の5度
			0.15*math.Sin(2*math.Pi*melodyFreq*2*t)               // 木管（オクターブ上）

		// ノイズはほぼゼロに
		// value += 0.01 * (rand.Float64()*2 - 1)

		// エンベロープ & 音量調整
		value *= env
		value *= 8000 // 音量抑制

		// クリップ防止
		if value > 32767 {
			value = 32767
		}
		if value < -32768 {
			value = -32768
		}
		pcmData[i] = int16(value)
	}

	// WAVヘッダ
	buf := make([]byte, 44+numSamples*2)
	copy(buf[0:], []byte("RIFF"))
	binary.LittleEndian.PutUint32(buf[4:], uint32(36+numSamples*2))
	copy(buf[8:], []byte("WAVE"))
	copy(buf[12:], []byte("fmt "))
	binary.LittleEndian.PutUint32(buf[16:], 16)
	binary.LittleEndian.PutUint16(buf[20:], 1)
	binary.LittleEndian.PutUint16(buf[22:], 1)
	binary.LittleEndian.PutUint32(buf[24:], uint32(sampleRate))
	binary.LittleEndian.PutUint32(buf[28:], uint32(sampleRate*2))
	binary.LittleEndian.PutUint16(buf[32:], 2)
	binary.LittleEndian.PutUint16(buf[34:], 16)
	copy(buf[36:], []byte("data"))
	binary.LittleEndian.PutUint32(buf[40:], uint32(numSamples*2))

	offset := 44
	for _, sample := range pcmData {
		binary.LittleEndian.PutUint16(buf[offset:], uint16(sample))
		offset += 2
	}

	return buf, nil
}

func audioHandler(w http.ResponseWriter, r *http.Request) {
	wavData, err := generateOrchestraWav()
	if err != nil {
		http.Error(w, "Failed to generate audio", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "audio/wav")
	w.Write(wavData)
}

func main() {
	http.HandleFunc("/audio", audioHandler)
	log.Println("Server running at http://localhost:8080/audio")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
