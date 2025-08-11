import React, { useState, useMemo } from 'react';
import { Search, AlertTriangle, Building2, TrendingUp, DollarSign, MessageCircle, ChevronDown, ChevronUp, Filter, X } from 'lucide-react';

// 型定義
interface CautionItem {
  id: string;
  title: string;
  description: string;
  severity: 'high' | 'medium' | 'low';
}

interface CompanyType {
  id: string;
  name: string;
  description: string;
  compatibility: number;
  features: string[];
}

interface Benefit {
  id: string;
  title: string;
  description: string;
  category: 'technical' | 'communication' | 'leadership';
  impact: number;
}

interface PricingTier {
  id: string;
  name: string;
  rate: number;
  currency: string;
  unit: 'hour' | 'day' | 'month';
  description: string;
}

interface ConsultationTopic {
  id: string;
  topic: string;
  description: string;
  urgency: 'high' | 'medium' | 'low';
  category: 'technical' | 'career' | 'business';
}

interface AppState {
  searchTerm: string;
  selectedCategory: string;
  expandedSection: string | null;
  filterSeverity: string;
  filterUrgency: string;
}

type EventHandler<T = HTMLInputElement> = React.ChangeEvent<T>;
type ButtonClickHandler = React.MouseEvent<HTMLButtonElement>;

export const SelfManualComponent: React.FC = () => {
  // Ste管理
  const [state, setState] = useState<AppState>({
    searchTerm: '',
    selectedCategory: 'all',
    expandedSection: null,
    filterSeverity: 'all',
    filterUrgency: 'all'
  });

  // サンプルデータ
  const cautions: CautionItem[] = [
    {
      id: '1',
      title: '過度な完璧主義',
      description: 'デッドラインよりも品質を優先しがちです。適度な妥協点を見つけることが重要です。',
      severity: 'high'
    },
    {
      id: '2',
      title: 'コミュニケーション頻度',
      description: '集中時間が必要なため、定期的な進捗報告のスケジュールを設定してください。',
      severity: 'medium'
    },
    {
      id: '3',
      title: '新技術への関心',
      description: '新しい技術に興味を持ちやすいため、プロジェクトスコープの管理に注意が必要です。',
      severity: 'low'
    }
  ];

  const companyTypes: CompanyType[] = [
    {
      id: '1',
      name: 'スタートアップ企業',
      description: '柔軟性と成長志向の環境',
      compatibility: 95,
      features: ['高い自由度', '幅広い業務経験', '急速な成長']
    },
    {
      id: '2',
      name: 'テック企業',
      description: '技術力重視の開発環境',
      compatibility: 88,
      features: ['最新技術', 'イノベーション', 'エンジニア文化']
    },
    {
      id: '3',
      name: '大手企業',
      description: '安定した環境でのプロダクト開発',
      compatibility: 72,
      features: ['安定性', '体系的な研修', 'チーム開発']
    }
  ];

  const benefits: Benefit[] = [
    {
      id: '1',
      title: 'フルスタック開発',
      description: 'フロントエンドからバックエンドまで一貫した開発が可能',
      category: 'technical',
      impact: 9
    },
    {
      id: '2',
      title: '問題解決能力',
      description: '複雑な課題を分析し、効率的なソリューションを提供',
      category: 'technical',
      impact: 8
    },
    {
      id: '3',
      title: 'チームコラボレーション',
      description: '異なる職種のメンバーと効果的に協働できる',
      category: 'communication',
      impact: 7
    }
  ];

  const pricing: PricingTier[] = [
    {
      id: '1',
      name: 'コンサルティング',
      rate: 8000,
      currency: 'JPY',
      unit: 'hour',
      description: '技術相談・アーキテクチャ設計'
    },
    {
      id: '2',
      name: '開発業務',
      rate: 60000,
      currency: 'JPY',
      unit: 'day',
      description: '実装・テスト・デプロイメント'
    },
    {
      id: '3',
      name: 'プロジェクト管理',
      rate: 500000,
      currency: 'JPY',
      unit: 'month',
      description: 'チームリード・プロジェクト統括'
    }
  ];

  const consultations: ConsultationTopic[] = [
    {
      id: '1',
      topic: 'キャリアパス相談',
      description: '技術者としてのキャリア設計についてアドバイス',
      urgency: 'medium',
      category: 'career'
    },
    {
      id: '2',
      topic: 'アーキテクチャ設計',
      description: 'システム設計やマイクロサービス化の検討',
      urgency: 'high',
      category: 'technical'
    },
    {
      id: '3',
      topic: 'チーム構築',
      description: '開発チームの組織化と効率化',
      urgency: 'low',
      category: 'business'
    }
  ];

  // Event Handlers
  const handleSearchChange = (e: EventHandler): void => {
    setState(prev => ({ ...prev, searchTerm: e.target.value }));
  };

  const handleCategoryChange = (e: EventHandler<HTMLSelectElement>): void => {
    setState(prev => ({ ...prev, selectedCategory: e.target.value }));
  };

  const handleSectionToggle = (section: string) => (e: ButtonClickHandler): void => {
    e.preventDefault();
    setState(prev => ({ 
      ...prev, 
      expandedSection: prev.expandedSection === section ? null : section 
    }));
  };

  const handleFilterChange = (filterType: 'severity' | 'urgency') => 
    (e: EventHandler<HTMLSelectElement>): void => {
      setState(prev => ({ 
        ...prev, 
        [filterType === 'severity' ? 'filterSeverity' : 'filterUrgency']: e.target.value 
      }));
    };

  const clearFilters = (e: ButtonClickHandler): void => {
    e.preventDefault();
    setState(prev => ({
      ...prev,
      searchTerm: '',
      selectedCategory: 'all',
      filterSeverity: 'all',
      filterUrgency: 'all'
    }));
  };

  // フィルタリング処理（map, filter, listを活用）
  const filteredCautions = useMemo(() => {
    return cautions
      .filter(caution => 
        caution.title.toLowerCase().includes(state.searchTerm.toLowerCase()) ||
        caution.description.toLowerCase().includes(state.searchTerm.toLowerCase())
      )
      .filter(caution => 
        state.filterSeverity === 'all' || caution.severity === state.filterSeverity
      );
  }, [state.searchTerm, state.filterSeverity]);

  const filteredBenefits = useMemo(() => {
    return benefits
      .filter(benefit => 
        state.selectedCategory === 'all' || benefit.category === state.selectedCategory
      )
      .filter(benefit =>
        benefit.title.toLowerCase().includes(state.searchTerm.toLowerCase())
      )
      .sort((a, b) => b.impact - a.impact);
  }, [state.selectedCategory, state.searchTerm]);

  const filteredConsultations = useMemo(() => {
    return consultations
      .filter(consultation =>
        consultation.topic.toLowerCase().includes(state.searchTerm.toLowerCase())
      )
      .filter(consultation =>
        state.filterUrgency === 'all' || consultation.urgency === state.filterUrgency
      );
  }, [state.searchTerm, state.filterUrgency]);

  const getSeverityColor = (severity: string): string => {
    const colorMap: Record<string, string> = {
      high: 'bg-red-900 text-red-200 border-red-700',
      medium: 'bg-yellow-900 text-yellow-200 border-yellow-700',
      low: 'bg-green-900 text-green-200 border-green-700'
    };
    return colorMap[severity] || 'bg-gray-700 text-gray-300 border-gray-600';
  };

  const getUrgencyColor = (urgency: string): string => {
    const colorMap: Record<string, string> = {
      high: 'border-l-red-400',
      medium: 'border-l-yellow-400',
      low: 'border-l-blue-400'
    };
    return colorMap[urgency] || 'border-l-gray-400';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-800 to-gray-900 px-8 py-4 sm:px-12 sm:py-6 lg:px-16 lg:py-8">
      <div className="max-w-7xl mx-auto">
        {/* ヘッダー */}
        <div className="text-center mb-6 sm:mb-8 lg:mb-10">
          <h1 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-2 sm:mb-4 px-2">
            自己取り扱い説明書
          </h1>
          <p className="text-sm sm:text-base md:text-lg text-gray-300 max-w-xl lg:max-w-2xl mx-auto px-4 leading-relaxed">
            効果的な協働のためのガイドライン・スキル・料金体系
          </p>
        </div>

        {/* 検索・フィルター */}
        <div className="bg-gray-800 border border-gray-700 rounded-xl shadow-lg p-4 sm:p-6 mb-6 sm:mb-8">
          <div className="flex flex-col gap-3 sm:gap-4">
            <div className="relative w-full">
              <Search className="absolute left-2 sm:left-3 top-2.5 sm:top-3 h-4 w-4 sm:h-5 sm:w-5 text-gray-400 z-10" />
              <input
                type="text"
                placeholder="検索..."
                value={state.searchTerm}
                onChange={handleSearchChange}
                className="w-full pl-8 sm:pl-10 pr-3 sm:pr-4 py-2 sm:py-2.5 bg-gray-700 border border-gray-600 text-white placeholder-gray-400 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm sm:text-base"
                autoComplete="off"
                spellCheck="false"
              />
            </div>
            <div className="flex flex-col sm:flex-row gap-2 sm:gap-3">
              <select
                value={state.selectedCategory}
                onChange={handleCategoryChange}
                className="flex-1 sm:flex-none px-3 sm:px-4 py-2 sm:py-2.5 bg-gray-700 border border-gray-600 text-white rounded-lg focus:ring-2 focus:ring-blue-500 text-sm sm:text-base"
                autoComplete="off"
              >
                <option value="all">全カテゴリ</option>
                <option value="technical">技術</option>
                <option value="communication">コミュニケーション</option>
                <option value="leadership">リーダーシップ</option>
              </select>
              <button
                onClick={clearFilters}
                className="px-3 sm:px-4 py-2 sm:py-2.5 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors flex items-center justify-center gap-2 text-sm sm:text-base font-medium"
              >
                <X className="h-4 w-4" />
                <span>クリア</span>
              </button>
            </div>
          </div>
        </div>

        <div className="space-y-6 sm:space-y-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6 lg:gap-8">
            {/* 注意点セクション */}
            <section className="bg-gray-800 border border-gray-700 rounded-xl shadow-lg overflow-hidden">
              <button
                onClick={handleSectionToggle('cautions')}
                className="w-full p-4 sm:p-6 flex items-center justify-between hover:bg-gray-700 transition-colors text-white"
              >
                <div className="flex items-center gap-2 sm:gap-3">
                  <AlertTriangle className="h-5 w-5 sm:h-6 sm:w-6 text-red-400 flex-shrink-0" />
                  <h2 className="text-lg sm:text-xl font-semibold text-white">注意点</h2>
                </div>
                {state.expandedSection === 'cautions' ? 
                  <ChevronUp className="h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" /> : 
                  <ChevronDown className="h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" />
                }
              </button>
              
              {state.expandedSection === 'cautions' && (
                <div className="px-4 sm:px-6 pb-4 sm:pb-6">
                  <div className="mb-3 sm:mb-4">
                    <select
                      value={state.filterSeverity}
                      onChange={handleFilterChange('severity')}
                      className="px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700 border border-gray-600 text-white rounded-lg text-xs sm:text-sm w-full sm:w-auto"
                      autoComplete="off"
                    >
                      <option value="all">全重要度</option>
                      <option value="high">高</option>
                      <option value="medium">中</option>
                      <option value="low">低</option>
                    </select>
                  </div>
                  <div className="space-y-3 sm:space-y-4">
                    {filteredCautions.map(caution => (
                      <div key={caution.id} className="border border-gray-600 bg-gray-700 rounded-lg p-3 sm:p-4">
                        <div className="flex items-start gap-2 sm:gap-3 mb-2">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium border flex-shrink-0 ${getSeverityColor(caution.severity)}`}>
                            {caution.severity === 'high' ? '高' : caution.severity === 'medium' ? '中' : '低'}
                          </span>
                          <h3 className="font-semibold text-white text-sm sm:text-base leading-tight">{caution.title}</h3>
                        </div>
                        <p className="text-gray-300 text-xs sm:text-sm leading-relaxed">{caution.description}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>

            {/* おすすめ企業型セクション */}
            <section className="bg-gray-800 border border-gray-700 rounded-xl shadow-lg overflow-hidden">
              <button
                onClick={handleSectionToggle('companies')}
                className="w-full p-4 sm:p-6 flex items-center justify-between hover:bg-gray-700 transition-colors text-white"
              >
                <div className="flex items-center gap-2 sm:gap-3">
                  <Building2 className="h-5 w-5 sm:h-6 sm:w-6 text-blue-400 flex-shrink-0" />
                  <h2 className="text-lg sm:text-xl font-semibold text-white">おすすめ企業型</h2>
                </div>
                {state.expandedSection === 'companies' ? 
                  <ChevronUp className="h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" /> : 
                  <ChevronDown className="h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" />
                }
              </button>
              
              {state.expandedSection === 'companies' && (
                <div className="px-4 sm:px-6 pb-4 sm:pb-6 space-y-3 sm:space-y-4">
                  {companyTypes.map(company => (
                    <div key={company.id} className="border border-gray-600 bg-gray-700 rounded-lg p-3 sm:p-4">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-3 gap-2">
                        <h3 className="font-semibold text-white text-sm sm:text-base">{company.name}</h3>
                        <div className="flex items-center gap-2">
                          <div className="w-12 sm:w-16 bg-gray-600 rounded-full h-1.5 sm:h-2">
                            <div 
                              className="bg-blue-500 h-1.5 sm:h-2 rounded-full transition-all duration-300" 
                              style={{width: `${company.compatibility}%`}}
                            />
                          </div>
                          <span className="text-xs sm:text-sm font-medium text-blue-400 min-w-[3rem]">
                            {company.compatibility}%
                          </span>
                        </div>
                      </div>
                      <p className="text-gray-300 text-xs sm:text-sm mb-3 leading-relaxed">{company.description}</p>
                      <div className="flex flex-wrap gap-1.5 sm:gap-2">
                        {company.features.map((feature, index) => (
                          <span key={index} className="px-2 py-1 bg-blue-900 text-blue-200 rounded-full text-xs">
                            {feature}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6 lg:gap-8">
            {/* 利益セクション */}
            <section className="bg-gray-800 border border-gray-700 rounded-xl shadow-lg overflow-hidden">
              <button
                onClick={handleSectionToggle('benefits')}
                className="w-full p-4 sm:p-6 flex items-center justify-between hover:bg-gray-700 transition-colors text-white"
              >
                <div className="flex items-center gap-2 sm:gap-3">
                  <TrendingUp className="h-5 w-5 sm:h-6 sm:w-6 text-green-400 flex-shrink-0" />
                  <h2 className="text-lg sm:text-xl font-semibold text-white">提供できる価値</h2>
                </div>
                {state.expandedSection === 'benefits' ? 
                  <ChevronUp className="h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" /> : 
                  <ChevronDown className="h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" />
                }
              </button>
              
              {state.expandedSection === 'benefits' && (
                <div className="px-4 sm:px-6 pb-4 sm:pb-6 space-y-3 sm:space-y-4">
                  {filteredBenefits.map(benefit => (
                    <div key={benefit.id} className="border border-gray-600 bg-gray-700 rounded-lg p-3 sm:p-4">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-2 gap-2">
                        <h3 className="font-semibold text-white text-sm sm:text-base">{benefit.title}</h3>
                        <div className="flex items-center gap-1">
                          {[...Array(10)].map((_, i) => (
                            <div
                              key={i}
                              className={`w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-full transition-all duration-200 ${
                                i < benefit.impact ? 'bg-green-400' : 'bg-gray-600'
                              }`}
                            />
                          ))}
                        </div>
                      </div>
                      <p className="text-gray-300 text-xs sm:text-sm leading-relaxed">{benefit.description}</p>
                    </div>
                  ))}
                </div>
              )}
            </section>

            {/* 単価セクション */}
            <section className="bg-gray-800 border border-gray-700 rounded-xl shadow-lg overflow-hidden">
              <button
                onClick={handleSectionToggle('pricing')}
                className="w-full p-4 sm:p-6 flex items-center justify-between hover:bg-gray-700 transition-colors text-white"
              >
                <div className="flex items-center gap-2 sm:gap-3">
                  <DollarSign className="h-5 w-5 sm:h-6 sm:w-6 text-yellow-400 flex-shrink-0" />
                  <h2 className="text-lg sm:text-xl font-semibold text-white">料金体系</h2>
                </div>
                {state.expandedSection === 'pricing' ? 
                  <ChevronUp className="h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" /> : 
                  <ChevronDown className="h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" />
                }
              </button>
              
              {state.expandedSection === 'pricing' && (
                <div className="px-4 sm:px-6 pb-4 sm:pb-6 space-y-3 sm:space-y-4">
                  {pricing.map(tier => (
                    <div key={tier.id} className="border border-gray-600 bg-gray-700 rounded-lg p-3 sm:p-4">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-2">
                        <h3 className="font-semibold text-white text-sm sm:text-base">{tier.name}</h3>
                        <div className="text-left sm:text-right">
                          <div className="text-xl sm:text-2xl font-bold text-green-400">
                            ¥{tier.rate.toLocaleString()}
                          </div>
                          <div className="text-xs sm:text-sm text-gray-400">
                            / {tier.unit === 'hour' ? '時間' : tier.unit === 'day' ? '日' : '月'}
                          </div>
                        </div>
                      </div>
                      <p className="text-gray-300 text-xs sm:text-sm leading-relaxed">{tier.description}</p>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </div>
        </div>

        {/* 相談事項セクション */}
        <section className="mt-6 sm:mt-8 bg-gray-800 border border-gray-700 rounded-xl shadow-lg overflow-hidden">
          <button
            onClick={handleSectionToggle('consultations')}
            className="w-full p-4 sm:p-6 flex items-center justify-between hover:bg-gray-700 transition-colors text-white"
          >
            <div className="flex items-center gap-2 sm:gap-3">
              <MessageCircle className="h-5 w-5 sm:h-6 sm:w-6 text-purple-400 flex-shrink-0" />
              <h2 className="text-lg sm:text-xl font-semibold text-white">相談可能な事項</h2>
            </div>
            {state.expandedSection === 'consultations' ? 
              <ChevronUp className="h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" /> : 
              <ChevronDown className="h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" />
            }
          </button>
          
          {state.expandedSection === 'consultations' && (
            <div className="px-4 sm:px-6 pb-4 sm:pb-6">
              <div className="mb-3 sm:mb-4">
                <select
                  value={state.filterUrgency}
                  onChange={handleFilterChange('urgency')}
                  className="px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700 border border-gray-600 text-white rounded-lg text-xs sm:text-sm w-full sm:w-auto"
                  autoComplete="off"
                >
                  <option value="all">全緊急度</option>
                  <option value="high">高</option>
                  <option value="medium">中</option>
                  <option value="low">低</option>
                </select>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 2xl:grid-cols-3 gap-3 sm:gap-4">
                {filteredConsultations.map(consultation => (
                  <div key={consultation.id} className={`border-l-4 bg-gray-700 border border-gray-600 rounded-lg p-3 sm:p-4 ${getUrgencyColor(consultation.urgency)}`}>
                    <h3 className="font-semibold text-white mb-2 text-sm sm:text-base">{consultation.topic}</h3>
                    <p className="text-gray-300 text-xs sm:text-sm mb-3 leading-relaxed">{consultation.description}</p>
                    <div className="flex flex-wrap gap-1.5 sm:gap-2">
                      <span className="px-2 py-1 bg-gray-600 text-gray-300 rounded-full text-xs font-medium">
                        {consultation.category}
                      </span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        consultation.urgency === 'high' ? 'bg-red-900 text-red-200' :
                        consultation.urgency === 'medium' ? 'bg-yellow-900 text-yellow-200' :
                        'bg-blue-900 text-blue-200'
                      }`}>
                        {consultation.urgency === 'high' ? '急' : consultation.urgency === 'medium' ? '中' : '低'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
};
