import { useState, useEffect } from 'react';
import { Link } from 'react-router';

export function Mains(){
  const [visibleChars, setVisibleChars] = useState(0);
  const [phase, setPhase] = useState('appearing'); // 'appearing', 'disappearing', 'buttons'
  const [visibleButtons, setVisibleButtons] = useState(0);
  const text = "Happy Welcome";
  
  const questions = [
    { id: 1, text: "Annual income?", icon: "🌤️" },
    { id: 2, text: "Skill up?", icon: "🎬" },
    { id: 3, text: "Business Consulting?", icon: "🍳" },
    { id: 4, text: "Other contact information", icon: "✈️" }
  ];

  useEffect(() => {
    if (phase === 'appearing') {
      const timer = setInterval(() => {
        setVisibleChars(prev => {
          if (prev < text.length) {
            return prev + 1;
          }
          clearInterval(timer);
          // 文字表示完了後、1.5秒待ってから消去開始
          setTimeout(() => setPhase('disappearing'), 1500);
          return prev;
        });
      }, 150);
      return () => clearInterval(timer);
    }

    if (phase === 'disappearing') {
      const timer = setInterval(() => {
        setVisibleChars(prev => {
          if (prev > 0) {
            return prev - 1;
          }
          clearInterval(timer);
          // 文字消去完了後、ボタン表示開始
          setTimeout(() => setPhase('buttons'), 300);
          return prev;
        });
      }, 100);
      return () => clearInterval(timer);
    }

    if (phase === 'buttons') {
      const timer = setInterval(() => {
        setVisibleButtons(prev => {
          if (prev < questions.length) {
            return prev + 1;
          }
          clearInterval(timer);
          return prev;
        });
      }, 200);
      return () => clearInterval(timer);
    }
  }, [phase, text.length, questions.length]);

  const handleQuestionClick = (question:string) => {
    console.log('質問が選択されました:', question);
    // ここで質問に対する処理を追加
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-white dark:bg-gray-900 transition-colors duration-300">
      {phase !== 'buttons' ? (
        // ウェルカムテキスト表示フェーズ
        <div className="text-center">
          <div 
            className="text-6xl font-bold transform -rotate-1 hover:rotate-0 transition-transform duration-300"
            style={{ fontFamily: 'cursive' }}
          >
            {text.split('').map((char, index) => (
              <span
                key={index}
                className={`inline-block transition-all duration-700 ease-out text-gray-800 dark:text-white ${
                  index < visibleChars 
                    ? 'opacity-100 transform translate-y-0' 
                    : 'opacity-0 transform translate-y-8'
                }`}
                style={{
                  transitionDelay: `${index * 50}ms`,
                  textShadow: '2px 2px 4px rgba(0, 0, 0, 0.1), 0 0 10px rgba(255, 255, 255, 0.1)'
                }}
              >
                {char === ' ' ? '\u00A0' : char}
              </span>
            ))}
          </div>
        </div>
      ) : (
        // ボタン表示フェーズ
        <div className="w-full max-w-4xl px-8">
          <div className="grid grid-cols-2 gap-8">
            {questions.map((question, index) => (
              <button
                key={question.id}
                onClick={() => handleQuestionClick(question.text)}
                className={`
                  group relative p-8 rounded-2xl border-2 border-gray-200 dark:border-gray-700
                  bg-white dark:bg-gray-800 shadow-lg hover:shadow-xl
                  transform transition-all duration-700 ease-out hover:scale-105
                  ${index < visibleButtons 
                    ? 'opacity-100 translate-y-0 rotate-0' 
                    : 'opacity-0 translate-y-12 rotate-3'
                  }
                `}
                style={{
                  transitionDelay: `${index * 150}ms`
                }}
              >
                <div className="text-center space-y-3">
                  <div className="text-4xl group-hover:scale-110 transition-transform duration-300">
                    {question.icon}
                  </div>
                  <div className="text-lg font-semibold text-gray-800 dark:text-white">
                    {question.text}
                  </div>
                </div>
                
                {/* ホバー時の装飾 */}
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-blue-500/10 to-purple-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}