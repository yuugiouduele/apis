import React, { useState, useRef, useCallback, useEffect } from "react";
import Webcam from "react-webcam";
import { Camera, Mic, MicOff, Play, Pause, Download } from "lucide-react";

export const ReactCameraComponent: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [audioChunks, setAudioChunks] = useState<BlobPart[]>([]);
  const [audioURL, setAudioURL] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [ready, setReady] = useState(false);

  // カメラ＋マイクのストリーム取得
  const handleGetUserMedia = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: { facingMode: "user" },
      });
      setStream(mediaStream);
      setReady(true);
    } catch (error) {
      console.error("getUserMedia error:", error);
      setReady(false);
    }
  }, []);

  // ストリーム取得・クリーンアップ
  useEffect(() => {
    handleGetUserMedia();
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [handleGetUserMedia, stream]);

  // MediaRecorder初期化とコールバック設定
  useEffect(() => {
    if (stream) {
      const options = { mimeType: "audio/webm" };
      const recorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          setAudioChunks((prev) => [...prev, event.data]);
        }
      };

      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
        const url = URL.createObjectURL(audioBlob);
        setAudioURL(url);
        setAudioChunks([]);
      };
    }
  }, [stream, audioChunks]);

  // 録音開始
  const startRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "inactive") {
      mediaRecorderRef.current.start();
      setIsRecording(true);
      setAudioChunks([]);
      setAudioURL(null);
    }
  };

  // 録音停止
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // カメラ撮影（スクリーンショット）
  const capture = useCallback(() => {
    const image = webcamRef.current?.getScreenshot();
    if (image) {
      setImageSrc(image);
    }
  }, []);

  // 音声再生制御
  const toggleAudioPlayback = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  // 録音音声ダウンロード
  const downloadAudio = () => {
    if (audioURL) {
      const a = document.createElement("a");
      a.href = audioURL;
      a.download = `recording_${Date.now()}.webm`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  // 撮影画像ダウンロード
  const downloadImage = () => {
    if (imageSrc) {
      const a = document.createElement("a");
      a.href = imageSrc;
      a.download = `photo_${Date.now()}.jpg`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          React Camera & Audio Recorder
        </h1>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* カメラセクション */}
          <div className="bg-gray-800 rounded-xl p-6 shadow-2xl">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Camera className="w-5 h-5" />
              Camera
            </h2>

            <div className="mb-4">
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                width={400}
                height={300}
                videoConstraints={{ facingMode: "user" }}
                className="rounded-lg border border-gray-600"
              />
            </div>

            <button
              onClick={capture}
              disabled={!ready}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-4 py-2 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
            >
              <Camera className="w-4 h-4" />
              Take Photo
            </button>
          </div>

          {/* 音声録音セクション */}
          <div className="bg-gray-800 rounded-xl p-6 shadow-2xl">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              {isRecording ? <MicOff className="w-5 h-5 text-red-500" /> : <Mic className="w-5 h-5" />}
              Audio Recording
            </h2>

            <div className="mb-4 text-center">
              {isRecording ? (
                <div className="text-red-500">
                  <div className="animate-pulse text-lg font-semibold">● Recording...</div>
                  <div className="text-sm mt-2">Click stop to finish recording</div>
                </div>
              ) : (
                <div className="text-gray-400">
                  <div className="text-lg">Ready to record</div>
                  <div className="text-sm mt-2">Click start to begin recording</div>
                </div>
              )}
            </div>

            <div className="space-y-3">
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  disabled={!ready}
                  className="w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-4 py-2 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
                >
                  <Mic className="w-4 h-4" />
                  Start Recording
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="w-full bg-gray-600 hover:bg-gray-700 px-4 py-2 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
                >
                  <MicOff className="w-4 h-4" />
                  Stop Recording
                </button>
              )}
            </div>
          </div>
        </div>

        {/* 結果表示セクション */}
        <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* 撮影画像 */}
          {imageSrc && (
            <div className="bg-gray-800 rounded-xl p-6 shadow-2xl">
              <h3 className="text-xl font-semibold mb-4">Captured Photo</h3>
              <div className="mb-4">
                <img src={imageSrc} alt="Captured" className="w-full rounded-lg border border-gray-600" />
              </div>
              <button
                onClick={downloadImage}
                className="w-full bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
              >
                <Download className="w-4 h-4" />
                Download Image
              </button>
            </div>
          )}

          {/* 録音結果 */}
          {audioURL && (
            <div className="bg-gray-800 rounded-xl p-6 shadow-2xl">
              <h3 className="text-xl font-semibold mb-4">Recorded Audio</h3>

              <div className="mb-4">
                <audio
                  ref={audioRef}
                  src={audioURL}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onEnded={() => setIsPlaying(false)}
                  className="w-full"
                  controls
                />
              </div>

              <div className="space-y-2">
                <button
                  onClick={toggleAudioPlayback}
                  className="w-full bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  {isPlaying ? "Pause" : "Play"} Audio
                </button>

                <button
                  onClick={downloadAudio}
                  className="w-full bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download Audio
                </button>
              </div>
            </div>
          )}
        </div>

        {/* ステータス表示 */}
        <div className="mt-8 text-center">
          <div
            className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm ${
              ready ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"
            }`}
          >
            <div className={`w-2 h-2 rounded-full ${ready ? "bg-green-400" : "bg-red-400"}`}></div>
            {ready ? "Camera & Microphone Ready" : "Waiting for permissions..."}
          </div>
        </div>
      </div>
    </div>
  );
};

