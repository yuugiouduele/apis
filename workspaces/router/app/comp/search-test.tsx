import React, { useState, useRef, useEffect } from 'react';
import { Search, Paperclip, Send, Bot, Zap, Brain, Sparkles, Play, Film, Youtube, Video, Tv } from 'lucide-react';

const SearchBar = () => {
  const [query, setQuery] = useState('');
  const [selectedModel, setSelectedModel] = useState('gpt-4');
  const [selectedVideoEngine, setSelectedVideoEngine] = useState('youtube');
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [currentEngine, setCurrentEngine] = useState('ai'); // 'ai' or 'video'
  const [isTransitioning, setIsTransitioning] = useState(false);
  
  const containerRef = useRef(null);
  const lastWheelTime = useRef(0);
  const wheelDeltaX = useRef(0);

  const aiModels = [
    { id: 'gpt-4', name: 'GPT-4', icon: Brain, color: 'text-purple-400' },
    { id: 'claude', name: 'Claude', icon: Bot, color: 'text-blue-400' },
    { id: 'gemini', name: 'Gemini', icon: Sparkles, color: 'text-green-400' },
    { id: 'gpt-3.5', name: 'GPT-3.5', icon: Zap, color: 'text-yellow-400' }
  ];

  const videoEngines = [
    { id: 'youtube', name: 'YouTube', icon: Youtube, color: 'text-red-400' },
    { id: 'vimeo', name: 'Vimeo', icon: Play, color: 'text-blue-400' },
    { id: 'dailymotion', name: 'Dailymotion', icon: Film, color: 'text-orange-400' },
    { id: 'twitch', name: 'Twitch', icon: Tv, color: 'text-purple-400' },
    { id: 'tiktok', name: 'TikTok', icon: Video, color: 'text-pink-400' }
  ];

  const handleSubmit = () => {
    if (query.trim()) {
      if (currentEngine === 'ai') {
        console.log('AI検索クエリ:', query);
        console.log('選択モデル:', selectedModel);
      } else {
        console.log('動画検索クエリ:', query);
        console.log('選択動画エンジン:', selectedVideoEngine);
      }
      console.log('添付ファイル:', attachedFiles);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFileAttach = (e) => {
    const files = Array.from(e.target.files);
    setAttachedFiles(prev => [...prev, ...files]);
  };

  const removeFile = (index) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const switchEngine = () => {
    if (isTransitioning) return;
    
    setIsTransitioning(true);
    
    // フェードアウト後にエンジン切り替え
    setTimeout(() => {
      setCurrentEngine(prev => prev === 'ai' ? 'video' : 'ai');
      setQuery(''); // クエリをクリア
    }, 500);
    
    // フェードイン完了
    setTimeout(() => {
      setIsTransitioning(false);
    }, 1000);
  };

  // マウスパッド/トラックパッドのスワイプイベントハンドラー
  const handleWheel = (e) => {
    if (isTransitioning) return;
    
    const now = Date.now();
    const timeDelta = now - lastWheelTime.current;
    
    // 横方向のスクロールを検出（trackpadのスワイプ）
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      e.preventDefault(); // 横スクロールを防止
      
      wheelDeltaX.current += e.deltaX;
      
      // スワイプの蓄積値が閾値を超えた場合
      const threshold = 150;
      if (Math.abs(wheelDeltaX.current) > threshold) {
        // スワイプ方向に関係なく切り替え
        switchEngine();
        wheelDeltaX.current = 0; // リセット
        lastWheelTime.current = now;
        return;
      }
    }
    
    // 一定時間経過後にデルタをリセット（連続スワイプでない場合）
    if (timeDelta > 200) {
      wheelDeltaX.current = e.deltaX || 0;
    }
    
    lastWheelTime.current = now;
    
    // 300ms後にデルタを段階的にリセット
    setTimeout(() => {
      if (Date.now() - lastWheelTime.current > 250) {
        wheelDeltaX.current *= 0.5; // 段階的に減少
        if (Math.abs(wheelDeltaX.current) < 10) {
          wheelDeltaX.current = 0;
        }
      }
    }, 300);
  };

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // ホイール/トラックパッドイベント
    container.addEventListener('wheel', handleWheel, { passive: false });

    return () => {
      container.removeEventListener('wheel', handleWheel);
    };
  }, [isTransitioning]);

  const selectedModelData = aiModels.find(model => model.id === selectedModel);
  const selectedVideoData = videoEngines.find(engine => engine.id === selectedVideoEngine);
  const ModelIcon = selectedModelData?.icon || Bot;
  const VideoIcon = selectedVideoData?.icon || Youtube;

  const getEngineStyles = () => {
    if (currentEngine === 'ai') {
      return {
        gradient: 'from-blue-600 to-purple-600',
        ring: 'ring-blue-500',
        button: 'bg-blue-600 hover:bg-blue-700',
        title: 'AI検索アシスタント',
        subtitle: 'お好みのAIモデルで検索してください',
        placeholder: '何でも聞いてください...'
      };
    } else {
      return {
        gradient: 'from-red-600 to-pink-600',
        ring: 'ring-red-500',
        button: 'bg-red-600 hover:bg-red-700',
        title: '動画検索エンジン',
        subtitle: 'お好みのプラットフォームで動画を検索',
        placeholder: '動画を検索してください...'
      };
    }
  };

  const styles = getEngineStyles();

  return (
    <div className={`min-h-screen transition-colors duration-300 ${
      isDarkMode ? 'bg-gray-900' : 'bg-gray-50'
    }`}>
      <div className="container mx-auto px-4 py-8">
        {/* ダークモード切り替えボタンのみ */}
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

        {/* フェードイン・フェードアウト コンテナ */}
        <div 
          ref={containerRef}
          className={`transition-all duration-500 ease-in-out ${
            isTransitioning 
              ? 'opacity-0 transform translate-y-4 scale-95' 
              : 'opacity-100 transform translate-y-0 scale-100'
          }`}
          style={{
            userSelect: 'none'
          }}
        >
          {/* メインタイトル */}
          <div className="text-center mb-8">
            <h1 className={`text-3xl font-bold mb-2 bg-gradient-to-r ${styles.gradient} bg-clip-text text-transparent`}>
              {styles.title}
            </h1>
            <p className={`text-lg ${
              isDarkMode ? 'text-gray-400' : 'text-gray-600'
            }`}>
              {styles.subtitle}
            </p>
            <div className="mt-4 text-sm text-gray-500">
              ← マウスパッドでスワイプして切り替え →
            </div>
          </div>

          {/* 検索バー */}
          <div className="max-w-4xl mx-auto">
            <div className="space-y-4">
              {/* モデル/エンジン選択 */}
              <div className="flex flex-wrap gap-3 justify-center">
                {(currentEngine === 'ai' ? aiModels : videoEngines).map((item) => {
                  const Icon = item.icon;
                  const isSelected = currentEngine === 'ai' 
                    ? selectedModel === item.id 
                    : selectedVideoEngine === item.id;
                  
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => {
                        if (currentEngine === 'ai') {
                          setSelectedModel(item.id);
                        } else {
                          setSelectedVideoEngine(item.id);
                        }
                      }}
                      className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all duration-200 ${
                        isSelected
                          ? isDarkMode
                            ? `bg-gray-700 text-white ring-2 ${styles.ring}`
                            : `bg-blue-100 text-blue-700 ring-2 ring-blue-300`
                          : isDarkMode
                            ? 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                            : 'bg-white text-gray-600 hover:bg-gray-50 border'
                      }`}
                    >
                      <Icon size={18} className={isSelected ? item.color : ''} />
                      <span className="font-medium">{item.name}</span>
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
                isDarkMode ? `focus-within:${styles.ring}/20` : 'focus-within:ring-blue-200'
              }`}>
                <div className="flex items-center p-4">
                  {/* 検索アイコン */}
                  <div className="flex-shrink-0 mr-3">
                    <Search className={`w-5 h-5 ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`} />
                  </div>

                  {/* 選択中のモデル/エンジン表示 */}
                  <div className={`flex items-center gap-2 px-3 py-1 rounded-full mr-3 ${
                    isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
                  }`}>
                    {currentEngine === 'ai' ? (
                      <>
                        <ModelIcon size={16} className={selectedModelData?.color} />
                        <span className={`text-sm font-medium ${
                          isDarkMode ? 'text-gray-300' : 'text-gray-600'
                        }`}>
                          {selectedModelData?.name}
                        </span>
                      </>
                    ) : (
                      <>
                        <VideoIcon size={16} className={selectedVideoData?.color} />
                        <span className={`text-sm font-medium ${
                          isDarkMode ? 'text-gray-300' : 'text-gray-600'
                        }`}>
                          {selectedVideoData?.name}
                        </span>
                      </>
                    )}
                  </div>

                  {/* テキスト入力 */}
                  <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder={styles.placeholder}
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
                      onClick={handleSubmit}
                      className={`p-2 rounded-lg transition-all duration-200 ${
                        query.trim()
                          ? `${styles.button} text-white shadow-md hover:shadow-lg`
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
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SearchBar;