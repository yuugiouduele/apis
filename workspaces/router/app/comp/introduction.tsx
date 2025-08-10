import React, { useState, useEffect, useCallback } from 'react';
import { Search, X, Download, ChevronLeft, ChevronRight, Mail, BarChart, GitCompare } from 'lucide-react';

// 型定義
interface SlideData {
  id: number;
  title: string;
  image: string;
  details: {
    design: string;
    test: string;
    infrastructure: string;
  };
}

interface NavItem {
  title: string;
  icon: React.ComponentType<{ size?: number }>;
  action: () => void;
}

interface ProfileSection {
  title: string;
  content: string;
}

// 抽象化されたコンポーネント
const SlideCard: React.FC<{
  slide: SlideData;
  index: number;
  currentSlide: number;
  isDarkMode: boolean;
  onModalOpen: (slide: SlideData) => void;
  isTransitioning: boolean;
}> = ({ slide, index, currentSlide, isDarkMode, onModalOpen, isTransitioning }) => {
  const isActive = index === currentSlide;
  const isPrev = index === (currentSlide - 1 + 3) % 3; // slides.lengthの代わりに3を使用
  
  return (
    <div 
      className={`absolute w-full h-full transition-all duration-1000 ease-in-out ${
        isActive 
          ? 'opacity-100 z-10 scale-100' 
          : isPrev
            ? 'opacity-0 z-0 scale-90'
            : 'opacity-0 z-0 scale-90'
      }`}
      style={{
        transform: isActive 
          ? 'rotateY(0deg) scale(1)' 
          : isPrev
            ? 'rotateY(90deg) scale(0.9)'
            : 'rotateY(-90deg) scale(0.9)'
      }}
    >
      <img 
        src={slide.image} 
        alt={slide.title}
        className="w-full h-full object-cover"
      />
      <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/30 to-transparent flex items-center justify-center">
        <div className="text-center transform transition-all duration-1000">
          <h2 className={`text-4xl font-bold text-white mb-8 transition-all duration-1000 ${
            isActive ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'
          }`}>
            {slide.title}
          </h2>
          <button
            onClick={() => onModalOpen(slide)}
            disabled={isTransitioning}
            className={`px-6 py-3 rounded-lg font-semibold transition-all duration-500 hover:scale-105 disabled:opacity-50 ${
              isActive ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'
            } ${
              isDarkMode 
                ? 'bg-blue-600 hover:bg-blue-700 text-white' 
                : 'bg-blue-500 hover:bg-blue-600 text-white'
            }`}
          >
            詳細を見る
          </button>
        </div>
      </div>
    </div>
  );
};

const NavItemComponent: React.FC<{
  item: { title: string };
  isActive: boolean;
  isDarkMode: boolean;
  onClick: () => void;
  icon: React.ComponentType<{ size?: number }>;
}> = ({ item, isActive, isDarkMode, onClick, icon: Icon }) => (
  <button
    onClick={onClick}
    className={`w-full text-left p-3 rounded-lg transition-colors flex items-center space-x-3 ${
      isActive 
        ? isDarkMode ? 'bg-blue-700 text-white' : 'bg-blue-100 text-blue-900'
        : isDarkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
    }`}
  >
    <Icon size={18} />
    <span>{item.title}</span>
  </button>
);

const ProfileCard: React.FC<{
  section: ProfileSection;
  index: number;
  currentProfileSection: number;
  isDarkMode: boolean;
}> = ({ section, index, currentProfileSection, isDarkMode }) => {
  const isActive = currentProfileSection === index;
  
  return (
    <div
      className={`p-6 rounded-lg transition-all duration-300 ${
        isActive
          ? isDarkMode
            ? 'bg-blue-900 border-2 border-blue-400 shadow-lg transform scale-105'
            : 'bg-blue-50 border-2 border-blue-400 shadow-lg transform scale-105'
          : isDarkMode
            ? 'bg-gray-800 border border-gray-700 hover:bg-gray-750'
            : 'bg-white border border-gray-200 hover:bg-gray-50'
      }`}
    >
      <h3 className="text-xl font-semibold mb-4 text-blue-600">{section.title}</h3>
      <p className={`leading-relaxed ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
        {section.content}
      </p>
    </div>
  );
};

const SearchBar: React.FC<{
  isSearchOpen: boolean;
  searchTerm: string;
  onSearchChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  isDarkMode: boolean;
}> = ({ isSearchOpen, searchTerm, onSearchChange, isDarkMode }) => (
  isSearchOpen ? (
    <div className="mt-4">
      <div className="relative">
        <Search size={20} className="absolute left-3 top-3 text-gray-400" />
        <input
          type="text"
          placeholder="内容を検索... (Space to toggle)"
          value={searchTerm}
          onChange={onSearchChange}
          className={`w-full pl-10 pr-4 py-2 rounded-lg border focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
            isDarkMode 
              ? 'bg-gray-700 border-gray-600 text-white' 
              : 'bg-white border-gray-300'
          }`}
          autoFocus
        />
      </div>
    </div>
  ) : null
);

export const PresentationSlider: React.FC = () => {
  const [currentSlide, setCurrentSlide] = useState<number>(0);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [modalContent, setModalContent] = useState<SlideData | null>(null);
  const [isSearchOpen, setIsSearchOpen] = useState<boolean>(false);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [isDarkMode, setIsDarkMode] = useState<boolean>(false);
  const [currentProfileSection, setCurrentProfileSection] = useState<number>(0);
  const [isNavbarOpen, setIsNavbarOpen] = useState<boolean>(true);
  const [isTransitioning, setIsTransitioning] = useState<boolean>(false);
  const [pressedKeys, setPressedKeys] = useState<Set<string>>(new Set());

  // スライド資料データ
  const slides: SlideData[] = [
    {
      id: 1,
      title: "システム概要",
      image: "/api/placeholder/600/400",
      details: {
        design: "マイクロサービスアーキテクチャを採用し、各サービスは独立してデプロイ可能な設計となっています。APIゲートウェイによる統一されたエンドポイント管理を実現しています。",
        test: "単体テスト、結合テスト、E2Eテストを自動化し、コードカバレッジは95%以上を維持しています。CI/CDパイプラインにより品質担保を自動化しています。",
        infrastructure: "AWS EKSクラスターでコンテナ運用、RDSによるデータベース管理、CloudFrontによるCDN配信を実装。Auto Scalingによる負荷対応も実現しています。"
      }
    },
    {
      id: 2,
      title: "データ処理フロー",
      image: "/api/placeholder/600/400",
      details: {
        design: "リアルタイムストリーミング処理とバッチ処理のハイブリッド設計。Apache Kafkaによるメッセージキューイングシステムを中核としています。",
        test: "データ整合性テスト、パフォーマンステスト、障害復旧テストを定期実行。データ品質監視ダッシュボードで継続的な品質管理を実施。",
        infrastructure: "Apache Kafka、Apache Spark、Apache Airflow、Amazon S3によるデータレイク構成。分散処理により大容量データも効率的に処理できます。"
      }
    },
    {
      id: 3,
      title: "セキュリティ対策",
      image: "/api/placeholder/600/400",
      details: {
        design: "多層防御によるセキュリティ設計。OAuth 2.0、JWT認証、Role-based Access Control (RBAC)を実装しています。",
        test: "セキュリティテスト、脆弱性スキャン、ペネトレーションテストを定期実行。OWASP Top 10に対する対策を網羅しています。",
        infrastructure: "AWS WAF、AWS Security Hub、AWS GuardDutyによる包括的なセキュリティ監視体制。SSL/TLS暗号化と不正アクセス検知システムを導入。"
      }
    }
  ];

  // ナビゲーション項目
  const navItems: NavItem[] = [
    { title: "連絡", icon: Mail, action: () => window.open('mailto:contact@example.com') },
    { title: "マッチ度", icon: BarChart, action: () => console.log('マッチ度確認') },
    { title: "比較", icon: GitCompare, action: () => console.log('比較分析') }
  ];

  // プロフィールセクション
  const profileSections: ProfileSection[] = [
    {
      title: "経歴",
      content: "2018年よりソフトウェアエンジニアとしてキャリアをスタート。大手IT企業でWebアプリケーション開発、クラウドインフラ構築、チームリード経験を積む。現在はフルスタック開発者として幅広い技術領域で活動中。"
    },
    {
      title: "スキル",
      content: "フロントエンド: React, Vue.js, TypeScript, Next.js | バックエンド: Node.js, Python, Java, Go | インフラ: AWS, Docker, Kubernetes, Terraform | データベース: PostgreSQL, MongoDB, Redis | その他: GraphQL, マイクロサービス、DevOps"
    },
    {
      title: "ビジネスインパクト",
      content: "月間100万PVのWebサービス構築により売上20%向上に貢献。CI/CDパイプライン導入によりデプロイ時間を80%短縮。マイクロサービス化によりシステム可用性99.9%を実現し、顧客満足度向上に寄与。"
    }
  ];

  // 自動スライド（10秒間隔）
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % slides.length);
    }, 10000);
    return () => clearInterval(interval);
  }, [slides.length]);

  // キーダウンイベント
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    setPressedKeys(prev => new Set(prev).add(e.key));
    
    // Space + Enter の同時押し検出
    if (pressedKeys.has(' ') && e.key === 'Enter') {
      e.preventDefault();
      setIsNavbarOpen(prev => !prev);
    } else if (pressedKeys.has('Enter') && e.key === ' ') {
      e.preventDefault();
      setIsNavbarOpen(prev => !prev);
    }
  }, [pressedKeys]);

  // キーアップイベント
  const handleKeyUp = useCallback((e: KeyboardEvent) => {
    setPressedKeys(prev => {
      const newSet = new Set(prev);
      newSet.delete(e.key);
      return newSet;
    });
  }, []);

  // キーボードイベント（型定義修正）
  const handleKeyPress = useCallback((e: KeyboardEvent) => {
    const { key, shiftKey } = e;
    
    if (key === ' ' && shiftKey) {
      e.preventDefault();
      if (isModalOpen) {
        setIsModalOpen(false);
        setIsSearchOpen(false);
      } else {
        setCurrentProfileSection((prev) => (prev + 1) % profileSections.length);
      }
    } else if (key === ' ' && !shiftKey && isModalOpen) {
      e.preventDefault();
      setIsSearchOpen(!isSearchOpen);
    } else if (key === 'Enter' && shiftKey) {
      e.preventDefault();
      handleDownload();
    }
  }, [isModalOpen, isSearchOpen, profileSections.length]);

  useEffect(() => {
    const keyDownHandler = (e: KeyboardEvent) => handleKeyDown(e);
    const keyUpHandler = (e: KeyboardEvent) => handleKeyUp(e);
    const keyPressHandler = (e: KeyboardEvent) => handleKeyPress(e);
    
    document.addEventListener('keydown', keyDownHandler);
    document.addEventListener('keyup', keyUpHandler);
    document.addEventListener('keypress', keyPressHandler);
    
    return () => {
      document.removeEventListener('keydown', keyDownHandler);
      document.removeEventListener('keyup', keyUpHandler);
      document.removeEventListener('keypress', keyPressHandler);
    };
  }, [handleKeyDown, handleKeyUp, handleKeyPress]);

  // スライド操作
  const handleSlideTransition = useCallback((direction: 'next' | 'prev' | number) => {
    if (isTransitioning) return;
    
    setIsTransitioning(true);
    
    if (typeof direction === 'number') {
      setCurrentSlide(direction);
    } else {
      setCurrentSlide(prev => 
        direction === 'next' 
          ? (prev + 1) % slides.length
          : (prev - 1 + slides.length) % slides.length
      );
    }
    
    setTimeout(() => setIsTransitioning(false), 1000);
  }, [isTransitioning, slides.length]);

  // モーダル操作
  const openModal = useCallback((slideData: SlideData) => {
    setModalContent(slideData);
    setIsModalOpen(true);
  }, []);

  const closeModal = useCallback(() => {
    setIsModalOpen(false);
    setIsSearchOpen(false);
    setSearchTerm('');
  }, []);

  // ダウンロード処理
  const handleDownload = useCallback(() => {
    const element = document.createElement('a');
    element.href = 'data:text/plain;charset=utf-8,' + encodeURIComponent('プレゼンテーション資料');
    element.download = `slide-${currentSlide + 1}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  }, [currentSlide]);

  // 検索フィルタリング
  const filteredContent = modalContent && searchTerm
    ? Object.entries(modalContent.details).filter(([, value]) =>
        value.toLowerCase().includes(searchTerm.toLowerCase())
      )
    : modalContent ? Object.entries(modalContent.details) : [];

  const detailLabels: Record<string, string> = { 
    design: '詳細設計', 
    test: 'テスト結果', 
    infrastructure: 'インフラ構成' 
  };

  return (
    <div className={`min-h-screen transition-colors duration-300 ${isDarkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'}`}>
      {/* ダークモード切り替え */}
      <button
        onClick={() => setIsDarkMode(!isDarkMode)}
        className={`fixed top-4 right-4 z-50 p-2 rounded-full ${isDarkMode ? 'bg-gray-700 text-yellow-400' : 'bg-gray-200 text-gray-700'} hover:scale-110 transition-transform`}
      >
        {isDarkMode ? '☀️' : '🌙'}
      </button>

      {/* ナビゲーションバー */}
      <div className={`fixed left-0 top-0 h-full z-40 transition-transform duration-300 ${
        isNavbarOpen ? 'transform translate-x-0' : 'transform -translate-x-full'
      }`}>
        <div className={`h-full w-64 ${
          isDarkMode ? 'bg-gray-800' : 'bg-white'
        } shadow-2xl border-r-4 border-blue-500 rounded-r-3xl`}>
          <div className="p-6">
            <h2 className="text-xl font-bold mb-6">ナビゲーション</h2>
            <nav className="space-y-4">
              {slides.map((slide, index) => (
                <NavItemComponent
                  key={slide.id}
                  item={slide}
                  isActive={currentSlide === index}
                  isDarkMode={isDarkMode}
                  onClick={() => handleSlideTransition(index)}
                  icon={index === 0 ? BarChart : index === 1 ? GitCompare : Mail}
                />
              ))}
              
              <div className={`pt-4 border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-300'}`}>
                {navItems.map((item) => (
                  <NavItemComponent
                    key={item.title}
                    item={item}
                    isActive={false}
                    isDarkMode={isDarkMode}
                    onClick={item.action}
                    icon={item.icon}
                  />
                ))}
              </div>
              
              <div className="pt-4 border-t border-gray-300">
                <p className="text-sm opacity-70 mb-2">ショートカット:</p>
                <p className="text-xs opacity-60">Space + Enter : ナビ開閉</p>
              </div>
            </nav>
          </div>
        </div>
      </div>

      {/* ナビバー表示切り替えボタン */}
      {!isNavbarOpen && (
        <button
          onClick={() => setIsNavbarOpen(true)}
          className={`fixed left-4 top-4 z-50 p-2 rounded-full ${
            isDarkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-white hover:bg-gray-100'
          } shadow-lg transition-all`}
        >
          <ChevronRight size={20} />
        </button>
      )}

      {/* スライダーセクション */}
      <div className={`relative transition-all duration-300 ${isNavbarOpen ? 'ml-64' : 'ml-0'}`}>
        <div className={`relative h-screen overflow-hidden ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`} style={{ perspective: '1000px' }}>
          <div className="relative h-full" style={{ transformStyle: 'preserve-3d' }}>
            {slides.map((slide, index) => (
              <SlideCard
                key={slide.id}
                slide={slide}
                index={index}
                currentSlide={currentSlide}
                isDarkMode={isDarkMode}
                onModalOpen={openModal}
                isTransitioning={isTransitioning}
              />
            ))}
          </div>

          {/* 左右ナビゲーション */}
          <button
            onClick={() => handleSlideTransition('prev')}
            disabled={isTransitioning}
            className="absolute left-4 top-1/2 transform -translate-y-1/2 p-2 bg-white bg-opacity-20 hover:bg-opacity-30 text-white rounded-full transition-all disabled:opacity-50"
          >
            <ChevronLeft size={24} />
          </button>
          <button
            onClick={() => handleSlideTransition('next')}
            disabled={isTransitioning}
            className="absolute right-4 top-1/2 transform -translate-y-1/2 p-2 bg-white bg-opacity-20 hover:bg-opacity-30 text-white rounded-full transition-all disabled:opacity-50"
          >
            <ChevronRight size={24} />
          </button>

          {/* スライドインジケーター */}
          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-2">
            {slides.map((_, index) => (
              <button
                key={index}
                onClick={() => handleSlideTransition(index)}
                disabled={isTransitioning}
                className={`w-3 h-3 rounded-full transition-all disabled:opacity-50 ${
                  currentSlide === index ? 'bg-white' : 'bg-white bg-opacity-50'
                }`}
              />
            ))}
          </div>
        </div>

        {/* ダウンロードボタン */}
        <div className="absolute bottom-20 left-1/2 transform -translate-x-1/2">
          <button
            onClick={handleDownload}
            className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-semibold transition-all duration-300 hover:scale-105 ${
              isDarkMode 
                ? 'bg-green-600 hover:bg-green-700 text-white' 
                : 'bg-green-500 hover:bg-green-600 text-white'
            }`}
          >
            <Download size={20} />
            <span>資料をダウンロード (Shift+Enter)</span>
          </button>
        </div>
      </div>

      {/* 自己紹介セクション */}
      <div className={`py-20 px-8 transition-all duration-300 ${isNavbarOpen ? 'ml-64' : 'ml-0'} ${isDarkMode ? 'bg-gray-900' : 'bg-gray-50'}`}>
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">自己紹介</h2>
          <div className="grid md:grid-cols-3 gap-8">
            {profileSections.map((section, index) => (
              <ProfileCard
                key={section.title}
                section={section}
                index={index}
                currentProfileSection={currentProfileSection}
                isDarkMode={isDarkMode}
              />
            ))}
          </div>
          <p className="text-center mt-8 text-sm opacity-70">
            Shift+Spaceで大項目を移動できます
          </p>
        </div>
      </div>

      {/* モーダル */}
      {isModalOpen && modalContent && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className={`max-w-4xl w-full max-h-[90vh] overflow-hidden rounded-lg ${
            isDarkMode ? 'bg-gray-800' : 'bg-white'
          }`}>
            <div className={`p-6 border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <div className="flex justify-between items-center">
                <h3 className="text-2xl font-bold">{modalContent.title}</h3>
                <button
                  onClick={closeModal}
                  className={`p-2 rounded-full hover:scale-110 transition-transform ${
                    isDarkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
                  }`}
                >
                  <X size={24} />
                </button>
              </div>

              <SearchBar
                isSearchOpen={isSearchOpen}
                searchTerm={searchTerm}
                onSearchChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchTerm(e.target.value)}
                isDarkMode={isDarkMode}
              />
            </div>

            <div className="p-6 overflow-y-auto max-h-[60vh]">
              {(searchTerm ? filteredContent : Object.entries(modalContent.details || {})).map(([key, value]) => (
                <div key={key} className="mb-6">
                  <h4 className="text-lg font-semibold mb-2 capitalize text-blue-600">
                    {detailLabels[key] || key}
                  </h4>
                  <p className={`leading-relaxed ${
                    isDarkMode ? 'text-gray-300' : 'text-gray-700'
                  }`} style={{
                    animation: 'glow 2s ease-in-out infinite alternate',
                    textShadow: isDarkMode ? '0 0 10px rgba(59, 130, 246, 0.5)' : 'none'
                  }}>
                    {value}
                  </p>
                </div>
              ))}

              {searchTerm && filteredContent.length === 0 && (
                <p className="text-center text-gray-500 py-8">検索結果が見つかりません</p>
              )}
            </div>

            <div className={`p-4 border-t text-sm text-center opacity-70 ${
              isDarkMode ? 'border-gray-700' : 'border-gray-200'
            }`}>
              Spaceで検索 | Shift+Spaceで閉じる
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
