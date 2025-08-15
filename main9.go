package main

import (
	"log"
	"net/http"
)

func htmlHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	html := `<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>パンくずリストとアニメーション</title>
<style>
  body {
    font-family: Arial, sans-serif;
    padding: 20px;
  }
  /* パンくずリストスタイル */
  .breadcrumb {
    list-style: none;
    display: flex;
    padding: 0;
    margin-bottom: 30px;
    background: #f0f0f0;
    border-radius: 5px;
  }
  .breadcrumb li {
    padding: 8px 15px;
    cursor: pointer;
    position: relative;
    background: #ddd;
    margin-right: 5px;
    border-radius: 3px;
    transition: background-color 0.3s ease;
  }
  .breadcrumb li:last-child {
    background: #4CAF50;
    color: white;
    cursor: default;
  }
  .breadcrumb li:hover:not(:last-child) {
    background: #aaa;
  }
  .breadcrumb li::after {
    content: '>';
    position: absolute;
    right: -15px;
    top: 50%;
    transform: translateY(-50%);
    color: #555;
  }
  .breadcrumb li:last-child::after {
    content: '';
  }

  /* スライドショーコンテナ */
  .slider {
    width: 100%;
    max-width: 600px;
    height: 300px;
    position: relative;
    overflow: hidden;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    margin: 0 auto;
  }
  .slides {
    display: flex;
    width: 500%;
    height: 100%;
    transition: transform 0.5s ease-in-out;
  }
  .slide {
    width: 20%;
    height: 100%;
    flex-shrink: 0;
    background-size: cover;
    background-position: center;
  }

  /* スライド用ボタン */
  .slider-controls {
    text-align: center;
    margin-top: 10px;
  }
  .slider-controls button {
    background-color: #4CAF50;
    border: none;
    color: white;
    padding: 8px 15px;
    margin: 0 5px;
    border-radius: 3px;
    cursor: pointer;
    font-size: 16px;
  }
  .slider-controls button:hover {
    background-color: #45a049;
  }

</style>
</head>
<body>

<!-- パンくずリスト -->
<ul class="breadcrumb" id="breadcrumb">
  <li data-index="0">ホーム</li>
  <li data-index="1">カテゴリ</li>
  <li data-index="2">サブカテゴリ</li>
  <li data-index="3">現在のページ</li>
</ul>

<!-- スライドショー -->
<div class="slider">
  <div class="slides" id="slides">
    <div class="slide" style="background-image:url('https://via.placeholder.com/600x300/FF5733/ffffff?text=Slide+1');"></div>
    <div class="slide" style="background-image:url('https://via.placeholder.com/600x300/33C1FF/ffffff?text=Slide+2');"></div>
    <div class="slide" style="background-image:url('https://via.placeholder.com/600x300/8E44AD/ffffff?text=Slide+3');"></div>
    <div class="slide" style="background-image:url('https://via.placeholder.com/600x300/27AE60/ffffff?text=Slide+4');"></div>
    <div class="slide" style="background-image:url('https://via.placeholder.com/600x300/F1C40F/ffffff?text=Slide+5');"></div>
  </div>
</div>

<div class="slider-controls">
  <button id="prevBtn">前へ</button>
  <button id="nextBtn">次へ</button>
</div>

<script>
// パンくずクリックで現在の位置を変更（例としてalert表示）
document.getElementById('breadcrumb').addEventListener('click', function(e) {
  if(e.target && e.target.tagName === 'LI' && !e.target.classList.contains('active')) {
    alert("パンくずクリック：" + e.target.textContent);
  }
});

// スライドショー制御
const slides = document.getElementById('slides');
const totalSlides = slides.children.length;
let currentIndex = 0;

function showSlide(index) {
  if(index < 0) index = totalSlides - 1;
  if(index >= totalSlides) index = 0;
  currentIndex = index;
  slides.style.transform = 'translateX(' + (-index * 100/totalSlides) + '%)';
}

document.getElementById('prevBtn').addEventListener('click', () => {
  showSlide(currentIndex - 1);
});
document.getElementById('nextBtn').addEventListener('click', () => {
  showSlide(currentIndex + 1);
});

// 自動スライド（3秒ごと）
setInterval(() => {
  showSlide(currentIndex + 1);
}, 3000);
</script>

</body>
</html>`

	_, _ = w.Write([]byte(html))
}

func main() {
	http.HandleFunc("/", htmlHandler)
	log.Println("Server running on http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
