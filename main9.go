// main.go
package main

import (
	"bytes"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"flag"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"log"
	"math"
	"net"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

var (
	addr      = flag.String("addr", ":8080", "http listen address")
	fps       = flag.Int("fps", 20, "frames per second")
	width     = flag.Int("w", 640, "frame width")
	height    = flag.Int("h", 480, "frame height")
	seqGlobal uint32
)

func main() {
	flag.Parse()
	http.HandleFunc("/ws", wsHandler)
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(indexHTML()))
	})

	log.Printf("Listening %s (%dx%d, %d fps)", *addr, *width, *height, *fps)
	log.Fatal(http.ListenAndServe(*addr, nil))
}

// --------------------- WebSocket Handler ---------------------

func wsHandler(w http.ResponseWriter, r *http.Request) {
	if !isWebSocketRequest(r) {
		http.Error(w, "not a websocket request", http.StatusBadRequest)
		return
	}
	hj, ok := w.(http.Hijacker)
	if !ok {
		http.Error(w, "hijacking not supported", http.StatusInternalServerError)
		return
	}
	conn, _, err := hj.Hijack()
	if err != nil {
		log.Println("Hijack error:", err)
		return
	}
	defer conn.Close()

	key := r.Header.Get("Sec-WebSocket-Key")
	if key == "" {
		log.Println("Missing websocket key")
		return
	}
	acceptKey := computeAcceptKey(key)
	resp := "HTTP/1.1 101 Switching Protocols\r\n" +
		"Upgrade: websocket\r\n" +
		"Connection: Upgrade\r\n" +
		"Sec-WebSocket-Accept: " + acceptKey + "\r\n\r\n"
	if _, err := conn.Write([]byte(resp)); err != nil {
		log.Println("Handshake write error:", err)
		return
	}

	ticker := time.NewTicker(time.Duration(1000 / *fps) * time.Millisecond)
	defer ticker.Stop()

	for frame := 0; ; frame++ {
		<-ticker.C
		seq := atomic.AddUint32(&seqGlobal, 1)
		jpegBytes, err := generateJPEGFrame(*width, *height, frame)
		if err != nil {
			log.Println("Frame generation error:", err)
			return
		}

		header := make([]byte, 12)
		binary.BigEndian.PutUint32(header[0:4], seq)
		binary.BigEndian.PutUint64(header[4:12], uint64(time.Now().UnixNano()))
		packet := append(header, jpegBytes...)

		if err := wsWriteBinary(conn, packet); err != nil {
			log.Println("Write error:", err)
			return
		}
	}
}

// --------------------- WebSocket Helpers ---------------------

func isWebSocketRequest(r *http.Request) bool {
	return strings.Contains(strings.ToLower(r.Header.Get("Connection")), "upgrade") &&
		strings.ToLower(r.Header.Get("Upgrade")) == "websocket"
}

func computeAcceptKey(key string) string {
	h := sha1.New()
	h.Write([]byte(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"))
	return base64.StdEncoding.EncodeToString(h.Sum(nil))
}

func wsWriteBinary(conn net.Conn, payload []byte) error {
	var b bytes.Buffer
	b.WriteByte(0x82) // FIN=1, opcode=2
	plen := len(payload)
	if plen <= 125 {
		b.WriteByte(byte(plen))
	} else if plen <= 65535 {
		b.WriteByte(126)
		binary.Write(&b, binary.BigEndian, uint16(plen))
	} else {
		b.WriteByte(127)
		binary.Write(&b, binary.BigEndian, uint64(plen))
	}
	b.Write(payload)
	_, err := conn.Write(b.Bytes())
	return err
}

// --------------------- Frame Generation ---------------------

func generateJPEGFrame(W, H, frameNum int) ([]byte, error) {
	img := image.NewRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			rf := 0.5 + 0.5*math.Sin(float64(x)/20.0+float64(frameNum)/6.0)
			gf := 0.5 + 0.5*math.Cos(float64(y)/20.0+float64(frameNum)/8.0)
			bf := 0.5 + 0.5*math.Sin(float64(x+y)/40.0-float64(frameNum)/10.0)
			img.SetRGBA(x, y, color.RGBA{
				uint8(clampInt(int(rf*255), 0, 255)),
				uint8(clampInt(int(gf*255), 0, 255)),
				uint8(clampInt(int(bf*255), 0, 255)),
				255})
		}
	}
	// moving rectangle
	rw, rh := W/6, H/6
	rx := (frameNum*7)%(W+rw) - rw/2
	ry := (frameNum*3)%(H+rh) - rh/2
	rect := image.Rect(rx, ry, rx+rw, ry+rh).Intersect(img.Rect)
	draw.Draw(img, rect, &image.Uniform{color.RGBA{255, 255, 255, 180}}, image.Point{}, draw.Over)

	var buf bytes.Buffer
	err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 80})
	return buf.Bytes(), err
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// --------------------- HTML Client ---------------------

func indexHTML() string {
	return `<!doctype html>
<html>
<head><meta charset="utf-8"><title>WS Video Stream</title></head>
<body>
<h3>Low-latency WS Video Stream</h3>
<canvas id="cv"></canvas>
<script>
const ws = new WebSocket("ws://" + location.host + "/ws");
const canvas = document.getElementById("cv");
const ctx = canvas.getContext("2d");
ws.binaryType = "arraybuffer";
ws.onmessage = (ev) => {
	const buf = ev.data;
	if(buf.byteLength < 12) return;
	const pngBytes = new Uint8Array(buf, 12);
	const img = new Image();
	img.onload = ()=>{ canvas.width=img.width; canvas.height=img.height; ctx.drawImage(img,0,0); URL.revokeObjectURL(img.src);}
	img.src = URL.createObjectURL(new Blob([pngBytes],{type:'image/jpeg'}));
};
</script>
</body>
</html>`
}
