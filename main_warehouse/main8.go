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
	baseCharDur    = 0.23 // 1モーラ基準秒
	crossfadeRatio = 0.18 // 隣接音のクロスフェード比
	formantCount   = 50
	masterVolume   = 3400.0
)

// ===== voice mode =====
type VoiceMode struct {
	BaseF0          float64
	FormantScale    float64
	MasterVolumeMul float64
}

var VoiceModes = map[string]VoiceMode{
	"male":   {BaseF0: 122.0, FormantScale: 1.00, MasterVolumeMul: 1.0},
	"female": {BaseF0: 210.0, FormantScale: 1.12, MasterVolumeMul: 0.92},
	"child":  {BaseF0: 290.0, FormantScale: 1.26, MasterVolumeMul: 0.85},
}

// ===== original types =====
type consType int

const (
	consNone        consType = iota
	consPlosive              // 破裂
	consFricativeS           // /s/ さ行
	consFricativeSH          // /ʃ/ し系/拗音
	consNasal                // 鼻音
	consApproximant          // 半母音
)

type phoneme struct {
	cons      consType
	v         rune // あ/い/う/え/お
	nasal     bool // ん
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
func buildFormants(core [3]float64, seed int64, scale float64) FormantBank {
	rng := rand.New(rand.NewSource(seed))
	F := make([]float64, formantCount)
	B := make([]float64, formantCount)
	A := make([]float64, formantCount)

	targets := []float64{core[0] * scale, core[1] * scale, core[2] * scale, core[2]*scale + 800*scale, core[2]*scale + 1400*scale}
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
			phs[i].accentMul = 1.14 // 少し強めに（基準1.10→1.14)
		case 'L', 'ｌ', 'Ｌ', 'l':
			phs[i].accentMul = 0.90 // 少し抑える
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
				w[i].accentMul = 0.90 + 0.28*alpha // 0.90→1.18
			} else {
				beta := float64(i-drop) / float64(max(1, len(w)-drop))
				w[i].accentMul = 1.08 - 0.14*beta // 1.08→0.94
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
	b1 := 0.0
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

// DCカット（1次HPF） -- 元実装の近似を維持だが改良版を追加で使う
type dcHipass struct {
	a, z float64
}

func newDCHP() *dcHipass { return &dcHipass{a: 0.995} }

func (h *dcHipass) Process(x float64) float64 {
	y := x - h.z + h.a*yPrev(h) // 近似
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

// ===== FIR bandpass (windowed-sinc) generator and apply =====
func makeFIRBandpass(fc1, fc2, fs float64, taps int) []float64 {
	coeffs := make([]float64, taps)
	m := taps - 1
	for n := 0; n < taps; n++ {
		norm := float64(n - m/2)
		if n == m/2 {
			coeffs[n] = 2 * (fc2 - fc1) / fs
		} else {
			coeffs[n] = (math.Sin(2*math.Pi*fc2*norm/fs) - math.Sin(2*math.Pi*fc1*norm/fs)) / (math.Pi * norm)
		}
		// Hamming window
		coeffs[n] *= 0.54 - 0.46*math.Cos(2*math.Pi*float64(n)/float64(m))
	}
	// normalize energy-ish
	sum := 0.0
	for _, c := range coeffs {
		sum += math.Abs(c)
	}
	if sum > 0 {
		for i := range coeffs {
			coeffs[i] /= sum
		}
	}
	return coeffs
}

func applyFIR(input, coeffs []float64) []float64 {
	taps := len(coeffs)
	out := make([]float64, len(input))
	for i := 0; i < len(input); i++ {
		var sum float64
		for j := 0; j < taps; j++ {
			idx := i - j
			if idx >= 0 {
				sum += input[idx] * coeffs[j]
			}
		}
		out[i] = sum
	}
	return out
}

// ===== simple 1-pole lowpass and highpass for post-processing =====
func applyOnePoleLP(input []float64, fc float64) []float64 {
	out := make([]float64, len(input))
	if len(input) == 0 {
		return out
	}
	dt := 1.0 / float64(sampleRate)
	RC := 1.0 / (2 * math.Pi * fc)
	alpha := dt / (RC + dt)
	y := 0.0
	for i := 0; i < len(input); i++ {
		y = alpha*input[i] + (1-alpha)*y
		out[i] = y
	}
	return out
}

func applyOnePoleHP(input []float64, fc float64) []float64 {
	out := make([]float64, len(input))
	if len(input) == 0 {
		return out
	}
	dt := 1.0 / float64(sampleRate)
	RC := 1.0 / (2 * math.Pi * fc)
	alpha := RC / (RC + dt)
	prevY := 0.0
	prevX := input[0]
	for i := 0; i < len(input); i++ {
		x := input[i]
		y := alpha * (prevY + x - prevX)
		out[i] = y
		prevY = y
		prevX = x
	}
	return out
}

// ===== 1音素合成 (float版) with accent envelope and noise reduction =====
func synthPhonemeF(p phoneme, nextVowel *rune, seed int64, baseF0 float64, startBreath bool, formantScale float64, vmGain float64, firCoeffs []float64) []float64 {
	dur := baseCharDur * p.durScale
	dur *= 0.99 + (rand.Float64() * 0.02) // ±1% に減らして安定化

	N := int(float64(sampleRate) * dur)
	if N <= 0 {
		return nil
	}
	out := make([]float64, N)

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

	// 50フォルマント（モードによりスケール）
	bank := buildFormants(core, seed, formantScale)
	rng := rand.New(rand.NewSource(seed ^ 0x5bd1e995))

	// ビブラート＋ジッタ/シマー（ジッタを小さくして滑らかさ向上）
	vibHz := 5.2 + (rand.Float64()*0.4 - 0.2)
	vibAmt := 0.10 + (rand.Float64()*0.03 - 0.015) // 減らす
	jitter := (rand.Float64()*0.02 - 0.01)         // ジッタを弱める
	shimmer := 1.0 + (rand.Float64()*0.03 - 0.015)

	// fricative biquad (backup)
	var fric *biquad
	if p.cons == consFricativeS || p.cons == consFricativeSH {
		fric = fricativeFilter(p.cons)
	}

	// 語頭ブレス（更に控えめ）
	breathAmt := 0.0
	if startBreath {
		breathAmt = 0.035
	}

	// 子音種別の追加ゲイン（抑え気味）
	consGain := 1.0
	switch p.cons {
	case consPlosive:
		consGain = 1.08 // さらに抑制
	case consFricativeS, consFricativeSH:
		consGain = 1.05 // 抑制
	case consNasal:
		consGain = 1.08
	case consApproximant:
		consGain = 1.03
	default:
		consGain = 1.0
	}

	// accent envelope parameters
	// accentMul >1.0 => treat as accented; build a bell-shaped envelope early in the phoneme
	accentIsHigh := p.accentMul > 1.02
	// gaussian center early (0.25~0.35), sigma small for sharper yet natural rise
	center := 0.30
	sigma := 0.12

	for i := 0; i < N; i++ {
		t := float64(i) / float64(sampleRate)
		u := t / dur

		// accent envelope (0..1)
		accentEnv := 0.0
		if accentIsHigh {
			// gaussian centered at 'center'
			x := (u - center) / sigma
			accentEnv = math.Exp(-0.5 * x * x)
			// scale by (accentMul - 1) to modulate strength
			accentEnv *= (p.accentMul - 1.0)
			// clamp
			if accentEnv > 0.65 {
				accentEnv = 0.65
			}
		} else if p.accentMul < 0.98 {
			// depressed accent - slight valley
			x := (u - 0.45) / (sigma * 1.2)
			accentEnv = -0.3 * math.Exp(-0.5*x*x)
		}

		// F0 with accent applied smoothly
		f0 := baseF0 * (1.0 + accentEnv*0.22) // accentEnv increases F0 by up to ~22%
		// vibrato/jitter
		f0 *= math.Pow(2, (vibAmt*math.Sin(2*math.Pi*vibHz*t)+jitter)/12.0)

		// source
		src := math.Sin(2 * math.Pi * f0 * t)

		// small periodic perturbation of formants
		if i%240 == 0 {
			j := 1.0 + (rng.Float64()*2-1)*0.012
			for k := range bank.Freqs {
				bank.Freqs[k] *= j
			}
		}

		// CV遷移補間
		cvLead := p.leadCV
		cvTrail := p.trailCV
		alpha := 0.0
		if u < 0.4 {
			alpha = cvLead * (0.4 - u) / 0.4
		} else if u > 0.6 {
			beta := (u - 0.6) / 0.4
			alpha = -cvTrail * beta
		}
		for k := 0; k < 3; k++ {
			target := lerp(core[k], nextCore[k], math.Max(0, (u-0.5)*1.2))
			// accent briefly increases formant amplitude by boosting bank.Amps not frequencies
			bank.Freqs[k] = lerp(bank.Freqs[k], target, 0.02+0.02*alpha)
		}

		// 共鳴（フォルマント合計） — accent で Amps を強める
		voc := 0.0
		for k := 0; k < len(bank.Freqs); k++ {
			fi := bank.Freqs[k]
			bw := bank.BW[k]
			amp := bank.Amps[k]
			// small accent-based amp boost for clarity
			ampBoost := 1.0
			if accentEnv > 0 {
				ampBoost += 0.28 * accentEnv // up to ~28% boost for accented peak
			}
			env := math.Exp(-2 * math.Pi * bw * t / float64(sampleRate))
			voc += amp * ampBoost * env * math.Sin(2*math.Pi*fi*t)
		}

		// 子音成分（ノイズは抑えめ）
		consVal := 0.0
		switch p.cons {
		case consPlosive:
			if t < 0.028 {
				consVal = (rand.Float64()*2 - 1) * (1.0 - t/0.028) * 0.48 * consGain
			}
		case consFricativeS, consFricativeSH:
			noise := (rand.Float64()*2 - 1)
			if fric != nil {
				noise = fric.Process(noise) // band-limited via biquad
			}
			// apply smaller noise amplitude, but accentEnv slightly increases clarity
			consVal = noise * (0.18 + 0.03*math.Sin(2*math.Pi*7.0*t)) * consGain * (1.0 + 0.6*accentEnv)
		case consNasal:
			consVal = 0.20 * math.Sin(2*math.Pi*120*t) * consGain
		case consApproximant:
			consVal = 0.038 * (rand.Float64()*2 - 1) * consGain
		default:
			consVal = 0.0
		}

		// 語頭ブレス
		consVal += breathAmt * (rand.Float64()*2 - 1) * math.Exp(-t*25)

		// 合成（母音寄せで滑らかに）
		sample := (0.48*src + 1.02*voc + consVal)
		// accent also slightly sharpens attack via envelope shaping on ADSR
		env := adsr(t, dur)
		// make attack slightly stronger when accented
		if accentEnv > 0 {
			env *= (1.0 + 0.08*accentEnv)
		}
		sample *= env
		sample *= shimmer

		// global scaling; vmGain provided by mode
		sample *= masterVolume * vmGain

		out[i] = sample
	}

	// fricative の場合は FIR を適用して質感をクリアに（ミックス）
	if (p.cons == consFricativeS || p.cons == consFricativeSH) && len(firCoeffs) > 0 {
		filtered := applyFIR(out, firCoeffs)
		for i := range out {
			out[i] = 0.70*filtered[i] + 0.30*out[i] // slightly more filtered weight for clarity
		}
	}

	// 各音素ピーク管理
	peak := maxAbsFloat(out)
	if peak > 0 {
		if peak < 0.02 {
			scale := 0.02 / peak
			for i := range out {
				out[i] *= scale
			}
		}
		if peak > 0.98 {
			scale := 0.98 / peak
			for i := range out {
				out[i] *= scale
			}
		}
	}
	return out
}

// ===== バッファ結合（float版、クロスフェード） =====
func concatWithCrossfadeFloat(buffers [][]float64, crossRatio float64) []float64 {
	if len(buffers) == 0 {
		return nil
	}
	if len(buffers) == 1 {
		return buffers[0]
	}
	out := make([]float64, 0, len(buffers[0]))
	out = append(out, buffers[0]...)
	for idx := 1; idx < len(buffers); idx++ {
		a := out
		b := buffers[idx]
		xf := int(float64(min(len(a), len(b))) * crossRatio)
		if xf <= 0 {
			out = append(out, b...)
			continue
		}
		startA := len(a) - xf
		mixed := make([]float64, 0, startA+xf+(len(b)-xf))
		mixed = append(mixed, a[:startA]...)
		for i := 0; i < xf; i++ {
			wa := 1.0 - float64(i)/float64(xf)
			wb := float64(i) / float64(xf)
			s := a[startA+i]*wa + b[i]*wb
			mixed = append(mixed, s)
		}
		mixed = append(mixed, b[xf:]...)
		out = mixed
	}
	return out
}

func maxAbsFloat(buf []float64) float64 {
	max := 0.0
	for _, v := range buf {
		if v < 0 {
			v = -v
		}
		if v > max {
			max = v
		}
	}
	return max
}

// ===== PCM正規化（float版） =====
func normalizePCMFloat(pcm []float64, peak float64) {
	maxAbs := 1e-9
	for _, s := range pcm {
		if a := math.Abs(s); a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs == 0 {
		return
	}
	scale := peak * 1.0 / maxAbs
	for i := range pcm {
		pcm[i] *= scale
	}
}

// ===== float -> int16 に安全に変換するユーティリティ =====
func floatSliceToInt16(pcm []float64) []int16 {
	out := make([]int16, len(pcm))
	for i, v := range pcm {
		if v > 1.0 {
			v = 1.0
		} else if v < -1.0 {
			v = -1.0
		}
		out[i] = int16(v * 32767.0)
	}
	return out
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

// ===== 合成パイプライン（mode対応） =====
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

// 再帰で合成（単語内の音素列）
func synthPhonemesRecursive(phs []phoneme, baseF0 float64, startSeed int64, vm VoiceMode, firCoeffs []float64) []float64 {
	n := len(phs)
	if n == 0 {
		return nil
	}
	if n == 1 {
		p := phs[0]
		buf := synthPhonemeF(p, nil, startSeed, baseF0, true, vm.FormantScale, vm.MasterVolumeMul, firCoeffs)
		return buf
	}
	mid := n / 2
	left := synthPhonemesRecursive(phs[:mid], baseF0, startSeed, vm, firCoeffs)
	right := synthPhonemesRecursive(phs[mid:], baseF0, startSeed+int64(mid*1337), vm, firCoeffs)
	return concatWithCrossfadeFloat([][]float64{left, right}, crossfadeRatio)
}

func synthText(text, accent, mode string) []byte {
	key := hashKey("v5|" + text + "|" + accent + "|" + mode)
	if v, ok := ttsCache.Get(key); ok {
		return v
	}

	words := splitWordsPreserveSpaces(text)
	var allPCMFloat []float64

	vm, ok := VoiceModes[mode]
	if !ok {
		vm = VoiceModes["male"]
	}
	baseF0 := vm.BaseF0

	// FIRフィルタ（摩擦音帯域 3500–8000Hz, 63tap）
	firCoeffs := makeFIRBandpass(3500, math.Min(8000, float64(sampleRate)/2.0), float64(sampleRate), 63)

	for wi, word := range words {
		if strings.TrimSpace(word) == "" {
			// 語間無音
			sil := int(0.08 * float64(sampleRate))
			for i := 0; i < sil; i++ {
				allPCMFloat = append(allPCMFloat, 0.0)
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

		// 単語内合成（再帰）
		seed := int64(time.Now().UnixNano() ^ int64(wi*7919))
		wordPCM := synthPhonemesRecursive(phs, baseF0, seed, vm, firCoeffs)

		allPCMFloat = append(allPCMFloat, wordPCM...)

		// 語間の短い無音
		sil := int(0.06 * float64(sampleRate))
		for i := 0; i < sil; i++ {
			allPCMFloat = append(allPCMFloat, 0.0)
		}
	}

	// ポスト処理：全体のハイパス(40Hz)で低域の淀みを除去、その後ローパス(7000Hz)で高域ノイズを丸める
	allPCMFloat = applyOnePoleHP(allPCMFloat, 40.0)
	allPCMFloat = applyOnePoleLP(allPCMFloat, 7000.0)

	// 全体正規化（float版）
	normalizePCMFloat(allPCMFloat, 0.95)

	// 最終段で int16 に変換
	outInt := make([]int16, len(allPCMFloat))
	for i, v := range allPCMFloat {
		if v > 1.0 {
			v = 1.0
		} else if v < -1.0 {
			v = -1.0
		}
		outInt[i] = int16(v * 32767.0)
	}

	wav := pcmToWavLE(outInt)
	ttsCache.Put(key, wav)
	return wav
}

// ===== WAV 出力 =====
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
	mode := q.Get("mode")
	if mode == "" {
		mode = "male"
	}
	wav := synthText(txt, accent, mode)
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
input,button,select{font-size:16px;padding:8px;border-radius:10px;border:1px solid #ccc}
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
<select id="mode">
  <option value="male">male</option>
  <option value="female">female</option>
  <option value="child">child</option>
</select>
<button id="speak">Speak</button>
</div>
<p class="small">例: <code>?text=あめ です&accent=HL LL</code> / <code>?text=きょうは いい てんき&accent=auto</code></p>
<audio id="player" controls></audio>
<script>
document.getElementById('speak').onclick = async () => {
  const t = encodeURIComponent(document.getElementById('text').value);
  const a = encodeURIComponent(document.getElementById('accent').value || 'auto');
  const m = encodeURIComponent(document.getElementById('mode').value || 'male');
  const url = '/speak?text=' + t + '&accent=' + a + '&mode=' + m;
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
