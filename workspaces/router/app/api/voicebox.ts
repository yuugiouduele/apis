import axios from "axios";

const BASE_URL = "http://localhost:50021";

export async function getVoiceVoxAudio(text: string, speaker: number = 1): Promise<Blob> {
  // audio_query取得
  const queryRes = await axios.post(
    `${BASE_URL}/audio_query?text=${encodeURIComponent(text)}&speaker=${speaker}`
  );
  const queryData = queryRes.data;

  // ポーズを短くする調整（moras配列があれば）
  if (queryData.moras) {
    queryData.moras.forEach((mora: any) => {
      if (mora.length) {
        mora.length = Math.min(mora.length, 0.001); // 無音を短く調整
      }
    });
  }

  // 合成時パラメータ（flowingにするには必要なら変更）
  const synthesisParams = {
    enable_interrogative_upspeak: true,
    // speedScale: 1.0,
    // pitchScale: 0,
    // volumeScale: 1.0,
    //追加変更可
  };

  // synthesis API呼び出し（パラメータ合成）
  const synthRes = await axios.post(
    `${BASE_URL}/synthesis?speaker=${speaker}&enable_interrogative_upspeak=true`,
    queryData,
    { headers: { "Content-Type": "application/json" }, responseType: "blob" }
  );

  return synthRes.data; // Blobにて返る
}
