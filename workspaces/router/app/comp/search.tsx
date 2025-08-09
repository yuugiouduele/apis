import React, { useState } from 'react';
import { Search, Paperclip, Send, Bot, Zap, Brain, Sparkles } from 'lucide-react';

export const SearchBar = () => {
  const [query, setQuery] = useState('');
  const [selectedModel, setSelectedModel] = useState('gpt-4');
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);

  const aiModels = [
    { id: 'gpt-4', name: 'GPT-4', icon: Brain, color: 'text-purple-400' },
    { id: 'claude', name: 'Claude', icon: Bot, color: 'text-blue-400' },
    { id: 'gemini', name: 'Gemini', icon: Sparkles, color: 'text-green-400' },
    { id: 'gpt-3.5', name: 'GPT-3.5', icon: Zap, color: 'text-yellow-400' }
  ];
// フォーム送信時
const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
  e.preventDefault();
  if (query.trim()) {
    console.log('検索クエリ:', query);
    console.log('選択モデル:', selectedModel);
    console.log('添付ファイル:', attachedFiles);
    // ここで実際の検索処理を実装
  }
};

// キー押下時
const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSubmit(e as unknown as React.FormEvent<HTMLFormElement>);
  }
};

// ファイル添付時
const handleFileAttach = (e: React.ChangeEvent<HTMLInputElement>) => {
  const files = Array.from(e.target.files ?? []);
  setAttachedFiles(prev => [...prev, ...files]);
};

// ファイル削除時（イベント引数はないので型不要）
const removeFile = (index: number) => {
  setAttachedFiles(prev => prev.filter((_, i) => i !== index));
};


  const selectedModelData = aiModels.find(model => model.id === selectedModel);
  const ModelIcon = selectedModelData?.icon || Bot;

  return (
    <div className={`min-h-screen transition-colors duration-300 ${
      isDarkMode ? 'bg-gray-900' : 'bg-gray-50'
    }`}>
      <div className="container mx-auto px-4 py-8">
        {/* ダークモード切り替えボタン */}
        <div className="flex justify-end mb-6">
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className={`px-4 py-2 rounded-lg transition-colors ${
              isDarkMode 
                ? 'bg-gray-800 text-gray-200 hover:bg-gray-700' 
                : 'bg-white text-gray-700 hover:bg-gray-100 border'
            }`}
          >
            {isDarkMode ? '🌙 ダーク' : '☀️ ライト'}
          </button>
        </div>

        {/* メインタイトル */}
        <div className="text-center mb-8">
          <h1 className={`text-3xl font-bold mb-2 ${
            isDarkMode ? 'text-white' : 'text-gray-900'
          }`}>
            AI検索アシスタント
          </h1>
          <p className={`text-lg ${
            isDarkMode ? 'text-gray-400' : 'text-gray-600'
          }`}>
            お好みのAIモデルで検索してください
          </p>
        </div>

        {/* 検索バー */}
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* AIモデル選択 */}
            <div className="flex flex-wrap gap-3 justify-center">
              {aiModels.map((model) => {
                const Icon = model.icon;
                return (
                  <button
                    key={model.id}
                    type="button"
                    onClick={() => setSelectedModel(model.id)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all duration-200 ${
                      selectedModel === model.id
                        ? isDarkMode
                          ? 'bg-gray-700 text-white ring-2 ring-blue-500'
                          : 'bg-blue-100 text-blue-700 ring-2 ring-blue-300'
                        : isDarkMode
                          ? 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                          : 'bg-white text-gray-600 hover:bg-gray-50 border'
                    }`}
                  >
                    <Icon size={18} className={selectedModel === model.id ? model.color : ''} />
                    <span className="font-medium">{model.name}</span>
                  </button>
                );
              })}
            </div>

            {/* メイン検索バー */}
            <div className={`relative rounded-2xl shadow-lg transition-all duration-300 ${
              isDarkMode 
                ? 'bg-gray-800 border border-gray-700' 
                : 'bg-white border-2 border-gray-200'
            } focus-within:ring-4 ${
              isDarkMode ? 'focus-within:ring-blue-500/20' : 'focus-within:ring-blue-200'
            }`}>
              <div className="flex items-center p-4">
                {/* 検索アイコン */}
                <div className="flex-shrink-0 mr-3">
                  <Search className={`w-5 h-5 ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`} />
                </div>

                {/* 選択中のAIモデル表示 */}
                <div className={`flex items-center gap-2 px-3 py-1 rounded-full mr-3 ${
                  isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
                }`}>
                  <ModelIcon size={16} className={selectedModelData?.color} />
                  <span className={`text-sm font-medium ${
                    isDarkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>
                    {selectedModelData?.name}
                  </span>
                </div>

                {/* テキスト入力 */}
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="何でも聞いてください..."
                  rows={1}
                  className={`flex-1 resize-none outline-none text-lg ${
                    isDarkMode 
                      ? 'bg-transparent text-white placeholder-gray-400' 
                      : 'bg-transparent text-gray-900 placeholder-gray-500'
                  }`}
                  style={{
                    minHeight: '28px',
                    maxHeight: '120px'
                  }}
                />

                {/* アクションボタン */}
                <div className="flex items-center gap-2 ml-3">
                  {/* ファイル添付ボタン */}
                  <label className={`p-2 rounded-lg cursor-pointer transition-colors ${
                    isDarkMode 
                      ? 'hover:bg-gray-700 text-gray-400 hover:text-gray-300' 
                      : 'hover:bg-gray-100 text-gray-500 hover:text-gray-600'
                  }`}>
                    <Paperclip size={20} />
                    <input
                      type="file"
                      multiple
                      onChange={handleFileAttach}
                      className="hidden"
                    />
                  </label>

                  {/* 送信ボタン */}
                  <button
                    type="submit"
                    disabled={!query.trim()}
                    className={`p-2 rounded-lg transition-all duration-200 ${
                      query.trim()
                        ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-md hover:shadow-lg'
                        : isDarkMode
                          ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                          : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    }`}
                  >
                    <Send size={20} />
                  </button>
                </div>
              </div>

              {/* 添付ファイル表示 */}
              {attachedFiles.length > 0 && (
                <div className={`px-4 pb-4 border-t ${
                  isDarkMode ? 'border-gray-700' : 'border-gray-200'
                }`}>
                  <div className="flex flex-wrap gap-2 mt-3">
                    {attachedFiles.map((file, index) => (
                      <div
                        key={index}
                        className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
                          isDarkMode 
                            ? 'bg-gray-700 text-gray-300' 
                            : 'bg-gray-100 text-gray-600'
                        }`}
                      >
                        <Paperclip size={14} />
                        <span className="truncate max-w-32">{file.name}</span>
                        <button
                          type="button"
                          onClick={() => removeFile(index)}
                          className={`ml-1 rounded-full w-4 h-4 flex items-center justify-center text-xs ${
                            isDarkMode 
                              ? 'hover:bg-gray-600 text-gray-400' 
                              : 'hover:bg-gray-200 text-gray-500'
                          }`}
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* ヒント */}
            <div className="text-center">
              <p className={`text-sm ${
                isDarkMode ? 'text-gray-500' : 'text-gray-400'
              }`}>
                Enter で送信 • Shift + Enter で改行 • ファイルをドラッグ&ドロップも可能
              </p>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};
