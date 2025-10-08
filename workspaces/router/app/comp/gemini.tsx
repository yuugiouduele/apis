"use client";
import React, { useState } from "react";
import { generateContent } from "../api/fetch";

// === Roleごとのテンプレート ===
const roleTemplates = {
  医者: "【医療分析テンプレート】\n概要：○○ \n構造：△△ \n医学的観点：□□\n\n質問：",
  マネージャー: "【マネジメント提案テンプレート】\n提案：☆☆ \n要件定義：△△ \n機能：□□ \n非機能：♢♢\n\n質問：",
  営業: "【営業戦略テンプレート】\n企業分析：☆☆ \n提案：△△ \n商品効果紹介：□□ \n商品価格算出：♢♢\n\n質問：",
};

type Role = keyof typeof roleTemplates;

export const GeminiSearch: React.FC = () => {
  const [role, setRole] = useState<Role>("医者");
  const [input, setInput] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  // model設定パラメータ
  const [temperature, setTemperature] = useState(0.6);
  const [topP, setTopP] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(100000);

  // モーダル表示制御
  const [showModal, setShowModal] = useState(false);

  const handleSearch = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setResponse("");

    // 選択されたRoleのテンプレートと入力を結合
    const prompt = `${roleTemplates[role]}`;

    try {
      const result = await generateContent(
        role,
        prompt,
        input, // format削除済み
        1,
        {
          thinking_budget: 2048,
          max_output_tokens: maxTokens,
          temperature: temperature,
          top_p: topP,
          candidate_count: 1,
        }
      );
      setResponse(result);
    } catch (error: any) {
      console.error(error);
      setResponse("❌ エラーが発生しました。\n" + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-4 bg-gray-950 rounded-xl shadow-lg text-gray-100">
      <div className="flex items-center gap-2">
        {/* 入力バー */}
        <input
          type="text"
          placeholder="質問または専門用語を入力..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          className="flex-1 border border-gray-700 bg-gray-800 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder-gray-400"
        />

        {/* 設定アイコン */}
        <button
          onClick={() => setShowModal(true)}
          className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm"
          title="設定"
        >
          ⚙️
        </button>

        {/* 実行ボタン */}
        <button
          onClick={handleSearch}
          disabled={loading}
          className={`px-4 py-2 rounded text-white text-sm ${
            loading
              ? "bg-gray-500 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {loading ? "生成中..." : "生成"}
        </button>
      </div>

      {/* 結果表示 */}
      <div className="border border-gray-700 rounded p-4 bg-gray-900 whitespace-pre-wrap text-sm leading-relaxed min-h-[150px] overflow-auto">
        {loading ? (
    <div className="flex flex-col items-center text-gray-400">
      <div className="animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent mb-2"></div>
      <span>生成中です...</span>
    </div>
  ) : (
    <div>{response || "💡 結果がここに表示されます。"}</div>
  )}
      </div>

      {/* ===== モーダル ===== */}
      {showModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-gray-900 rounded-xl p-6 w-full max-w-md shadow-xl border border-gray-700">
            <h2 className="text-lg font-semibold mb-4 text-blue-400 flex items-center gap-2">
              ⚙️ モデル設定
            </h2>

            {/* Role選択 */}
            <div className="mb-4">
              <label className="block text-sm mb-2 text-gray-300">ロール選択</label>
              <div className="flex gap-2">
                {Object.keys(roleTemplates).map((r) => (
                  <button
                    key={r}
                    onClick={() => setRole(r as Role)}
                    className={`flex-1 px-3 py-2 rounded text-sm border ${
                      role === r
                        ? "bg-blue-600 border-blue-400 text-white"
                        : "bg-gray-800 border-gray-600 hover:bg-gray-700 text-gray-300"
                    }`}
                  >
                    {r}
                  </button>
                ))}
              </div>
            </div>

            {/* テンプレートプレビュー */}
            <div className="border border-gray-700 rounded p-3 bg-gray-800 text-xs text-gray-300 whitespace-pre-wrap mb-4">
              {roleTemplates[role]}
            </div>

            {/* スライダー設定 */}
            <div className="space-y-3 mb-4">
              <div>
                <label className="block text-sm">Temperature: {temperature}</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm">Top P: {topP}</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={topP}
                  onChange={(e) => setTopP(parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm">Max Tokens: {maxTokens}</label>
                <input
                  type="range"
                  min="256"
                  max="100000"
                  step="64"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>
            </div>

            {/* ボタン */}
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowModal(false)}
                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
              >
                閉じる
              </button>
              <button
                onClick={() => setShowModal(false)}
                className="px-4 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm text-white"
              >
                保存
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
