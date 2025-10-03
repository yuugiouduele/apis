package main

import (
    "fmt"
    "io/ioutil"
    "log"
    "net/http"
    "os"
    "os/exec"
)

func main() {
    http.HandleFunc("/tts", func(w http.ResponseWriter, r *http.Request) {
        text := r.URL.Query().Get("text")
        if text == "" {
            http.Error(w, "missing text parameter", http.StatusBadRequest)
            return
        }

        // 一時wavファイル名
        wavFile := "temp.wav"

        // espeakで音声合成（WAV出力）
        cmd := exec.Command("espeak", "-w", wavFile, text)
        if err := cmd.Run(); err != nil {
            http.Error(w, "espeak execution failed", http.StatusInternalServerError)
            return
        }

        // wavファイル読み込み
        data, err := ioutil.ReadFile(wavFile)
        if err != nil {
            http.Error(w, "failed to read wav file", http.StatusInternalServerError)
            return
        }

        // ファイル削除
        os.Remove(wavFile)

        // レスポンスヘッダ設定
        w.Header().Set("Content-Type", "audio/wav")
        w.Header().Set("Content-Disposition", "inline; filename=tts.wav")
        w.Write(data)
    })

    fmt.Println("Listening on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
