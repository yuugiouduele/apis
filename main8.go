// main.go
package main

import (
	"bytes"
	"container/list"
	"encoding/binary"
	"hash/fnv"
	"log"
	"math"
	"math/rand"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
)

const (
	sampleRate     = 16000
	channels       = 1
	bitsPerSample  = 16
	baseCharDur    = 0.23  // 1モーラ基準秒
	crossfadeRatio = 0.18  // 隣接音のクロスフェード比
	formantCount   = 50
	masterVolume   = 3400.0
)

type consType int

const (
	consNone consType = iota
	consPlosive       // 破裂
	consFricativeS    // /s/ さ行
	consFricativeSH   // /ʃ/ し系/拗音
	consNasal         // 鼻音
	consApproximant   // 半母音
)

type phoneme struct {
	cons      consType
	v         rune   // あ/い/う/え/お
	nasal     bool   // ん
	durScale  float64
	accentMul float64 // F0倍率（ピッチアクセント）
	leadCV    float64 // 子音先行係数（CV同化の微調整）
	trailCV   float64 // 母音の後部に子音余韻を乗せる係数
}

type FormantBank struct {
	Freqs []float64
	BW    []float64
	Amps  []float64
}

var vowelCore = map[rune][3]float64{
	'あ': {800, 1300, 2600},
	'い': {300, 2200, 3000},
	'う': {350, 800, 2700},
	'え': {500, 1900, 2700},
	'お': {500, 1000, 2500},
}
var nasalCore = [3]float64{250, 1200, 2500}

// ===== 正規化 =====
func hiraganaNormalize(s string) string {
	var b strings.Builder
	for _, r := range s {
		if unicode.In(r, unicode.Katakana) {
			r = r - 0x60
		}
		if unicode.In(r, unicode.Hiragana) || r == 'ー' || r == 'っ' || r == 'ゃ' || r == 'ゅ' || r == 'ょ' || r == ' ' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// ===== かな→(子音,母音) =====
func vowelOfKana(r rune) rune {
	switch r {
	case 'あ', 'か', 'が', 'さ', 'ざ', 'た', 'だ', 'な', 'は', 'ば', 'ぱ', 'ま', 'や', 'ら', 'わ':
		return 'あ'
	case 'い', 'き', 'ぎ', 'し', 'じ', 'ち', 'ぢ', 'に', 'ひ', 'び', 'ぴ', 'み', 'り':
		return 'い'
	case 'う', 'く', 'ぐ', 'す', 'ず', 'つ', 'づ', 'ぬ', 'ふ', 'ぶ', 'ぷ', 'む', 'ゆ', 'る':
		return 'う'
	case 'え', 'け', 'げ', 'せ', 'ぜ', 'て', 'で', 'ね', 'へ', 'べ', 'ぺ', 'め', 'れ':
		return 'え'
	case 'お', 'こ', 'ご', 'そ', 'ぞ', 'と', 'ど', 'の', 'ほ', 'ぼ', 'ぽ', 'も', 'よ', 'ろ', 'を':
		return 'お'
	default:
		return 0
	}
}

func decomposeKana(r rune) (consType, rune) {
	switch r {
	case 'あ', 'い', 'う', 'え', 'お':
		return consNone, r
	}
	if strings.ContainsRune("しじ", r) {
		return consFricativeSH, vowelOfKana(r)
	}
	if strings.ContainsRune("かがきぎくぐけげこご", r) {
		return consPlosive, vowelOfKana(r)
	}
	if strings.ContainsRune("さざすずせぜそぞ", r) {
		return consFricativeS, vowelOfKana(r)
	}
	if strings.ContainsRune("ただちぢつづてでとど", r) {
		return consPlosive, vowelOfKana(r)
	}
	if strings.ContainsRune("なにぬねの", r) {
		return consNasal, vowelOfKana(r)
	}
	if strings.ContainsRune("はばぱひびぴふぶぷへべぺほぼぽ", r) {
		return consFricativeS, vowelOfKana(r)
	}
	if strings.ContainsRune("まみむめも", r) {
		return consNasal, vowelOfKana(r)
	}
	if strings.ContainsRune("やゆよ", r) {
		return consApproximant, vowelOfKana(r)
	}
	if strings.ContainsRune("らりるれろ", r) {
		return consApproximant, vowelOfKana(r)
	}
	if strings.ContainsRune("わを", r) {
		return consApproximant, vowelOfKana(r)
	}
	return consNone, 0
}

// ===== テキスト→音素列 =====
func toPhonemes(text string) []phoneme {
	text = hiraganaNormalize(text)
	rs := []rune(text)
	var out []phoneme
	i := 0
	for i < len(rs) {
		r := rs[i]
		if r == ' ' {
			out = append(out, phoneme{cons: consNone, v: 'あ', durScale: 0.0001}) // 区切り用無音
			i++
			continue
		}
		// 拗音
		if i+1 < len(rs) {
			r2 := rs[i+1]
			if r2 == 'ゃ' || r2 == 'ゅ' || r2 == 'ょ' {
				v := map[rune]rune{'ゃ': 'あ', 'ゅ': 'う', 'ょ': 'お'}[r2]
				baseCons, _ := decomposeKana(r)
				if r == 'し' || r == 'じ' || baseCons == consFricativeS {
					out = append(out, phoneme{cons: consFricativeSH, v: v, durScale: 1.0, leadCV: 0.25, trailCV: 0.15})
				} else {
					out = append(out, phoneme{cons: consApproximant, v: 'い', durScale: 0.6, leadCV: 0.15})
					out = append(out, phoneme{cons: consNone, v: v, durScale: 0.9})
				}
				i += 2
				continue
			}
		}
		// 促音
		if r == 'っ' {
			out = append(out, phoneme{cons: consPlosive, v: 'あ', durScale: 0.35, leadCV: 0.32})
			i++
			continue
		}
		// 長音
		if r == 'ー' && len(out) > 0 {
			last := out[len(out)-1]
			last.durScale = 0.95
			last.trailCV = 0.1
			out = append(out, last)
			i++
			continue
		}
		// 撥音
		if r == 'ん' {
			out = append(out, phoneme{cons: consNasal, v: 'あ', nasal: true, durScale: 0.9, trailCV: 0.25})
			i++
			continue
		}
		// 通常
		c, v := decomposeKana(r)
		if v != 0 {
			out = append(out, phoneme{cons: c, v: v, durScale: 1.0, leadCV: 0.18, trailCV: 0.12})
		}
		i++
	}
	return out
}

// ===== 50フォルマント生成 =====
func buildFormants(core [3]float64, seed int64) FormantBank {
	rng := rand.New(rand.NewSource(seed))
	F := make([]float64, formantCount)
	B := make([]float64, formantCount)
	A := make([]float64, formantCount)

	targets := []float64{core[0], core[1], core[2], core[2] + 800, core[2] + 1400}
	for i := 0; i < formantCount; i++ {
		t := targets[i%len(targets)]
		jit := 1.0 + (rng.Float64()*0.16 - 0.08) // ±8%
		F[i] = t * jit
		B[i] = 70 + (F[i]/5000.0)*330.0 + rng.Float64()*20 // 70〜400Hz程度
		A[i] = math.Exp(-float64(i)*0.06) * (0.9 + rng.Float64()*0.2)
	}
	return FormantBank{Freqs: F, BW: B, Amps: A}
}

// ===== ピッチアクセント =====
func applyAccent(phs []phoneme, pattern []rune) {
	if len(pattern) == 0 {
		for i := range phs {
			phs[i].accentMul = 1.0
		}
		return
	}
	n := min(len(phs), len(pattern))
	for i := 0; i < n; i++ {
		switch pattern[i] {
		case 'H', 'ｈ', 'Ｈ', 'h':
			phs[i].accentMul = 1.10
		case 'L', 'ｌ', 'Ｌ', 'l':
			phs[i].accentMul = 0.92
		default:
			phs[i].accentMul = 1.0
		}
	}
	for i := n; i < len(phs); i++ {
		phs[i].accentMul = 1.0
	}
}

func applyAccentAuto(perWords [][]phoneme) {
	for _, w := range perWords {
		if len(w) == 0 {
			continue
		}
		drop := int(math.Max(1, math.Round(float64(len(w))*0.6)))
		for i := range w {
			if i < drop {
				alpha := float64(i) / float64(max(1, drop-1))
				w[i].accentMul = 0.92 + 0.18*alpha // 0.92→1.10
			} else {
				beta := float64(i-drop) / float64(max(1, len(w)-drop))
				w[i].accentMul = 1.06 - 0.12*beta // 1.06→0.94
			}
		}
	}
}

// ===== DSP小物 =====
func clamp16(x float64) int16 {
	if x > 32767 {
		return 32767
	}
	if x < -32768 {
		return -32768
	}
	return int16(x)
}

func adsr(t, total float64) float64 {
	a, d, s, r := 0.03, 0.07, 0.80, 0.08
	if t < a {
		return t / a
	}
	if t < a+d {
		return 1.0 - (t-a)*(1.0-s)/d
	}
	if t < total-r {
		return s
	}
	if t < total {
		return s * (1 - (t-(total-r))/r)
	}
	return 0
}

// ===== バイカッドBandpass（/s/・/sh/用） =====
type biquad struct {
	b0, b1, b2, a1, a2 float64
	z1, z2             float64
}

func newBiquadBandpass(fc, Q float64) *biquad {
	w0 := 2 * math.Pi * fc / float64(sampleRate)
	alpha := math.Sin(w0) / (2 * Q)
	cosw0 := math.Cos(w0)

	b0 := Q * alpha
	b1 := 0
	b2 := -Q * alpha
	a0 := 1 + alpha
	a1 := -2 * cosw0
	a2 := 1 - alpha

	return &biquad{
		b0: b0 / a0,
		b1: b1 / a0,
		b2: b2 / a0,
		a1: a1 / a0,
		a2: a2 / a0,
	}
}

func (f *biquad) Process(x float64) float64 {
	y := f.b0*x + f.z1
	f.z1 = f.b1*x - f.a1*y + f.z2
	f.z2 = f.b2*x - f.a2*y
	return y
}

// DCカット（1次HPF）
type dcHipass struct {
	a, z float64
}

func newDCHP() *dcHipass { return &dcHipass{a: 0.995} }

func (h *dcHipass) Process(x float64) float64 {
	y := x - h.z + h.a*yPrev(h) // y[n] depends on previous y; approximate with state-only update
	h.z = x
	return y
}
func yPrev(*dcHipass) float64 { return 0 } // 近似（16kHz/短音なら十分に抑制）

// ===== /s/ /sh/ フィルタ重み =====
func fricativeFilter(cons consType) *biquad {
	switch cons {
	case consFricativeS:
		return newBiquadBandpass(6500, 4.0) // Nyquist(8k)内
	case consFricativeSH:
		return newBiquadBandpass(3400, 3.0)
	default:
		return nil
	}
}

// ===== 共鳴の線形補間（連続音でフォルマント遷移） =====
func lerp(a, b float64, t float64) float64 { return a + (b-a)*t }

// ===== 1音素合成 =====
func synthPhoneme(p phoneme, nextVowel *rune, seed int64, baseF0 float64, startBreath bool) []int16 {
	dur := baseCharDur * p.durScale
	// ランダム微変動（人間らしさ）
	dur *= 0.98 + (rand.Float64() * 0.04) // ±2%

	N := int(float64(sampleRate) * dur)
	out := make([]int16, N)

	// 共鳴コア（CV遷移補間）
	core := vowelCore['あ']
	if p.nasal {
		core = nasalCore
	} else if c, ok := vowelCore[p.v]; ok {
		core = c
	}
	var nextCore [3]float64
	if nextVowel != nil {
		if c2, ok := vowelCore[*nextVowel]; ok {
			nextCore = c2
		} else {
			nextCore = core
		}
	} else {
		nextCore = core
	}

	// 50フォルマント
	bank := buildFormants(core, seed)
	rng := rand.New(rand.NewSource(seed ^ 0x5bd1e995))

	// ビブラート＋ジッタ/シマー
	vibHz := 5.3 + (rand.Float64()*0.6-0.3)
	vibAmt := 0.14 + (rand.Float64()*0.06-0.03) // ±0.14st +-0.03
	jitter := (rand.Float64()*0.06 - 0.03)      // F0微揺れ(半音単位)
	shimmer := 1.0 + (rand.Float64()*0.04 - 0.02)

	// fricative filter
	var fric *biquad
	if p.cons == consFricativeS || p.cons == consFricativeSH {
		fric = fricativeFilter(p.cons)
	}

	// 語頭ブレス（小）
	breathAmt := 0.0
	if startBreath {
		breathAmt = 0.06
	}

	for i := 0; i < N; i++ {
		t := float64(i) / float64(sampleRate)
		// F0
		f0 := baseF0 * p.accentMul
		f0 *= math.Pow(2, (vibAmt*math.Sin(2*math.Pi*vibHz*t)+jitter)/12.0)

		// 声帯源
		src := math.Sin(2 * math.Pi * f0 * t)

		// フォルマント微揺れ
		if i%220 == 0 {
			j := 1.0 + (rng.Float64()*2-1)*0.015
			for k := range bank.Freqs {
				bank.Freqs[k] *= j
			}
		}

		// CV遷移補間：前半は子音寄せ、後半は次母音へ寄せる
		u := t / dur
		cvLead := p.leadCV
		cvTrail := p.trailCV
		alpha := 0.0
		if u < 0.4 {
			alpha = cvLead * (0.4 - u) / 0.4 // 立ち上がりで子音寄り（実際は帯域処理で代用）
		} else if u > 0.6 {
			beta := (u - 0.6) / 0.4
			alpha = -cvTrail * beta // 終わりで次母音へ気持ち寄せ
		}
		// ここでは core 自体の粗い変位として適用
		for k := 0; k < 3; k++ {
			target := lerp(core[k], nextCore[k], math.Max(0, (u-0.5)*1.2))
			bank.Freqs[k] = lerp(bank.Freqs[k], target, 0.03+0.02*alpha)
		}

		// 簡易共鳴（減衰で包絡）
		voc := 0.0
		for k := 0; k < len(bank.Freqs); k++ {
			fi := bank.Freqs[k]
			bw := bank.BW[k]
			amp := bank.Amps[k]
			env := math.Exp(-2 * math.Pi * bw * t / float64(sampleRate))
			voc += amp * env * math.Sin(2*math.Pi*fi*t)
		}

		// 子音成分
		cons := 0.0
		switch p.cons {
		case consPlosive:
			if t < 0.028 {
				cons = (rand.Float64()*2 - 1) * (1.0 - t/0.028) * 0.85
			}
		case consFricativeS, consFricativeSH:
			noise := (rand.Float64()*2 - 1)
			if fric != nil {
				noise = fric.Process(noise)
			}
			cons = noise * (0.30 + 0.05*math.Sin(2*math.Pi*7.0*t))
		case consNasal:
			cons = 0.24 * math.Sin(2 * math.Pi * 120 * t)
		case consApproximant:
			cons = 0.06 * (rand.Float64()*2 - 1)
		}

		// 語頭ブレス
		cons += breathAmt * (rand.Float64()*2 - 1) * math.Exp(-t*25)

		// 合成
		sample := (0.55*src + 0.85*voc + cons)
		sample *= adsr(t, dur)
		sample *= shimmer
		sample *= masterVolume

		out[i] = clamp16(sample)
	}
	return out
}

// ===== バッファ結合 =====
func concatWithCrossfade(buffers [][]int16, crossRatio float64) []int16 {
	if len(buffers) == 0 {
		return nil
	}
	if len(buffers) == 1 {
		return buffers[0]
	}
	out := make([]int16, 0, len(buffers[0]))
	out = append(out, buffers[0]]...)
	for idx := 1; idx < len(buffers); idx++ {
		a := out
		b := buffers[idx]
		xf := int(float64(min(len(a), len(b))) * crossRatio)
		if xf <= 0 {
			out = append(out, b...)
			continue
		}
		startA := len(a) - xf
		mixed := make([]int16, 0, startA+xf+(len(b)-xf))
		mixed = append(mixed, a[:startA]...)
		for i := 0; i < xf; i++ {
			wa := 1.0 - float64(i)/float64(xf)
			wb := float64(i) / float64(xf)
			s := float64(a[startA+i])*wa + float64(b[i])*wb
			mixed = append(mixed, clamp16(s))
		}
		mixed = append(mixed, b[xf:]...)
		out = mixed
	}
	return out
}

func normalizePCM(pcm []int16, peak float64) {
	maxAbs := 1.0
	for _, s := range pcm {
		if a := math.Abs(float64(s)); a > maxAbs {
			maxAbs = a
		}
	}
	scale := peak * 32767.0 / maxAbs
	for i, s := range pcm {
		pcm[i] = clamp16(float64(s) * scale)
	}
}

func pcmToWavLE(pcm []int16) []byte {
	numSamples := len(pcm)
	dataSize := numSamples * 2
	totalSize := 44 + dataSize

	buf := bytes.NewBuffer(make([]byte, 0, totalSize))
	buf.WriteString("RIFF")
	binary.Write(buf, binary.LittleEndian, uint32(36+dataSize))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	binary.Write(buf, binary.LittleEndian, uint32(16))
	binary.Write(buf, binary.LittleEndian, uint16(1))
	binary.Write(buf, binary.LittleEndian, uint16(channels))
	binary.Write(buf, binary.LittleEndian, uint32(sampleRate))
	byteRate := sampleRate * channels * bitsPerSample / 8
	binary.Write(buf, binary.LittleEndian, uint32(byteRate))
	blockAlign := channels * bitsPerSample / 8
	binary.Write(buf, binary.LittleEndian, uint16(blockAlign))
	binary.Write(buf, binary.LittleEndian, uint16(bitsPerSample))
	buf.WriteString("data")
	binary.Write(buf, binary.LittleEndian, uint32(dataSize))
	for _, s := range pcm {
		binary.Write(buf, binary.LittleEndian, s)
	}
	return buf.Bytes()
}

// ===== LRUキャッシュ =====
type cacheEntry struct {
	key string
	wav []byte
}
type LRUCache struct {
	mu       sync.Mutex
	capacity int
	ll       *list.List
	table    map[string]*list.Element
}
func newLRU(cap int) *LRUCache {
	return &LRUCache{capacity: cap, ll: list.New(), table: make(map[string]*list.Element)}
}
func (c *LRUCache) Get(k string) ([]byte, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if ele, ok := c.table[k]; ok {
		c.ll.MoveToFront(ele)
		return ele.Value.(*cacheEntry).wav, true
	}
	return nil, false
}
func (c *LRUCache) Put(k string, v []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if ele, ok := c.table[k]; ok {
		ele.Value.(*cacheEntry).wav = v
		c.ll.MoveToFront(ele)
		return
	}
	ele := c.ll.PushFront(&cacheEntry{key: k, wav: v})
	c.table[k] = ele
	if c.ll.Len() > c.capacity {
		back := c.ll.Back()
		if back != nil {
			delete(c.table, back.Value.(*cacheEntry).key)
			c.ll.Remove(back)
		}
	}
}

// ===== 合成パイプライン =====
var ttsCache = newLRU(64)

func hashKey(s string) string {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return strconv.FormatUint(h.Sum64(), 16)
}

func splitWordsPreserveSpaces(s string) []string {
	parts := strings.Fields(s)
	return parts
}

func synthText(text, accent string) []byte {
	key := hashKey("v2|" + text + "|" + accent)
	if v, ok := ttsCache.Get(key); ok {
		return v
	}

	words := splitWordsPreserveSpaces(text)
	var allPCM []int16
	baseF0 := 122.0 // 男性寄り

	for wi, word := range words {
		if strings.TrimSpace(word) == "" {
			// 語間無音
			sil := int(0.08 * float64(sampleRate))
			for i := 0; i < sil; i++ {
				allPCM = append(allPCM, 0)
			}
			continue
		}
		phs := toPhonemes(word)

		// アクセント
		if accent == "" || accent == "auto" {
			applyAccentAuto([][]phoneme{phs})
		} else {
			acParts := strings.Fields(accent)
			pat := ""
			if wi < len(acParts) {
				pat = acParts[wi]
			}
			applyAccent(phs, []rune(pat))
			for i := range phs {
				if phs[i].accentMul == 0 {
					phs[i].accentMul = 1.0
				}
			}
		}

		// 合成
		buffers := make([][]int16, 0, len(phs))
		seed := int64(time.Now().UnixNano() ^ int64(wi*7919))
		for i, p := range phs {
			// 次母音（CV遷移用）
			var nextV *rune
			if i+1 < len(phs) {
				nextV = &phs[i+1].v
			}
			startBreath := (i == 0) // 語頭のみ
			buf := synthPhoneme(p, nextV, seed+int64(i*1337), baseF0, startBreath)
			buffers = append(buffers, buf)
		}
		wordPCM := concatWithCrossfade(buffers, crossfadeRatio)
		allPCM = append(allPCM, wordPCM...)

		// 語間の短い無音
		sil := int(0.06 * float64(sampleRate))
		for i := 0; i < sil; i++ {
			allPCM = append(allPCM, 0)
		}
	}

	// 正規化
	normalizePCM(allPCM, 0.95)
	wav := pcmToWavLE(allPCM)
	ttsCache.Put(key, wav)
	return wav
}

// ===== HTTP =====
func speakHandler(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	txt := q.Get("text")
	if txt == "" {
		http.Error(w, "text required (ひらがな/カタカナ/拗音/促音/長音対応。スペースで語区切り)", http.StatusBadRequest)
		return
	}
	if t2, err := url.QueryUnescape(txt); err == nil {
		txt = t2
	}
	accent := q.Get("accent") // "auto" / "" / "HLHL" 等（語ごとスペース区切り）
	wav := synthText(txt, accent)
	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("Content-Length", strconv.Itoa(len(wav)))
	_, _ = w.Write(wav)
}

const demoHTML = `<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>TTS Demo</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;
margin:40px;line-height:1.6}
input,button{font-size:16px;padding:8px;border-radius:10px;border:1px solid #ccc}
button{cursor:pointer}
.row{display:flex;gap:10px;flex-wrap:wrap}
.small{font-size:12px;color:#444}
code{background:#f4f4f4;padding:2px 6px;border-radius:6px}
</style>
</head>
<body>
<h1>日本語TTSデモ（WAV）</h1>
<div class="row">
<input id="text" size="40" value="こんにちは ししゃしゅしょ さしすせそ" />
<input id="accent" size="20" placeholder="auto / HLHL ..." value="auto" />
<button id="speak">Speak</button>
</div>
<p class="small">例: <code>?text=あめ です&accent=HL LL</code> / <code>?text=きょうは いい てんき&accent=auto</code></p>
<audio id="player" controls></audio>
<script>
document.getElementById('speak').onclick = async () => {
  const t = encodeURIComponent(document.getElementById('text').value);
  const a = encodeURIComponent(document.getElementById('accent').value || 'auto');
  const url = '/speak?text=' + t + '&accent=' + a;
  const res = await fetch(url);
  const blob = await res.blob();
  const obj = URL.createObjectURL(blob);
  const audio = document.getElementById('player');
  audio.src = obj;
  audio.play();
};
</script>
</body>
</html>`

func demoHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(demoHTML))
}

func main() {
	rand.Seed(time.Now().UnixNano())
	http.HandleFunc("/speak", speakHandler)
	http.HandleFunc("/demo", demoHandler)
	log.Println("Server at http://localhost:8080/demo")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

// ===== helpers =====
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
