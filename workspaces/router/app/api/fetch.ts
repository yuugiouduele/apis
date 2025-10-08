const GEMINI_API_KEY = 'AIzaSyAXPTTQpfDyTrgV-8TivhWWu0s9VmgJk_8';
if (!GEMINI_API_KEY) {
  throw new Error("❌ GEMINI_API_KEY が設定されていません。");
}

// Gemini 2.5 Flash のエンドポイント
const GEMINI_API_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent";

// TypeScript用の設定型
interface GenerateConfig {
  thinking_budget?: number;
  max_output_tokens?: number;
  temperature?: number;
  top_p?: number;
  candidate_count?: number;
}

// Gemini推論実行関数
export async function generateContent(
    role:string,
  con: string,
  format: string,
  i: number = 1,
  config: GenerateConfig = {}
) {
  const body = {
    contents: [
      {
        role: "user",
        parts: [
          {
            text: `あなたは${role}です。記号は何文字でもいいです。以下のコンテンツに${format}の形式で答えて。${con}`,
          },
        ],
      },
    ],
    // GenerateContentConfig に相当
    generationConfig: {
      maxOutputTokens: config.max_output_tokens ?? 1024,
      temperature: config.temperature ?? 0.2,
      topP: config.top_p ?? 0.9,
      candidateCount: config.candidate_count ?? 1,
      // thinking強化に相当 (Geminiの一部モデルで対応)
      thinkingConfig: {
        thinkingBudget: Math.pow(512, i),
      },
    },
  };

  const res = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Gemini API Error: ${res.status}\n${errText}`);
  }

  const data = await res.json();
  const output = data?.candidates?.[0]?.content?.parts?.[0]?.text ?? "(応答なし)";
  return output;
}
