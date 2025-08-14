import { jsx, jsxs, Fragment } from "react/jsx-runtime";
import { PassThrough } from "node:stream";
import { createReadableStreamFromReadable } from "@react-router/node";
import { ServerRouter, UNSAFE_withComponentProps, Outlet, UNSAFE_withErrorBoundaryProps, isRouteErrorResponse, Meta, Links, ScrollRestoration, Scripts, useNavigate } from "react-router";
import { isbot } from "isbot";
import { renderToPipeableStream } from "react-dom/server";
import { Link } from "react-router-dom";
import { useRef, useState, useCallback, useEffect, useMemo } from "react";
import Webcam from "react-webcam";
import { Camera, MicOff, Mic, Download, Pause, Play, Search, X, AlertTriangle, ChevronUp, ChevronDown, Building2, TrendingUp, DollarSign, MessageCircle, BarChart, GitCompare, Mail, ChevronRight, ChevronLeft, Brain, Bot, Sparkles, Zap, Youtube, Film, Tv, Video, Paperclip, Send } from "lucide-react";
const streamTimeout = 5e3;
function handleRequest(request, responseStatusCode, responseHeaders, routerContext, loadContext) {
  return new Promise((resolve, reject) => {
    let shellRendered = false;
    let userAgent = request.headers.get("user-agent");
    let readyOption = userAgent && isbot(userAgent) || routerContext.isSpaMode ? "onAllReady" : "onShellReady";
    const { pipe, abort } = renderToPipeableStream(
      /* @__PURE__ */ jsx(ServerRouter, { context: routerContext, url: request.url }),
      {
        [readyOption]() {
          shellRendered = true;
          const body = new PassThrough();
          const stream = createReadableStreamFromReadable(body);
          responseHeaders.set("Content-Type", "text/html");
          resolve(
            new Response(stream, {
              headers: responseHeaders,
              status: responseStatusCode
            })
          );
          pipe(body);
        },
        onShellError(error) {
          reject(error);
        },
        onError(error) {
          responseStatusCode = 500;
          if (shellRendered) {
            console.error(error);
          }
        }
      }
    );
    setTimeout(abort, streamTimeout + 1e3);
  });
}
const entryServer = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: handleRequest,
  streamTimeout
}, Symbol.toStringTag, { value: "Module" }));
const links = () => [{
  rel: "preconnect",
  href: "https://fonts.googleapis.com"
}, {
  rel: "preconnect",
  href: "https://fonts.gstatic.com",
  crossOrigin: "anonymous"
}, {
  rel: "stylesheet",
  href: "https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap"
}];
function Layout({
  children
}) {
  return /* @__PURE__ */ jsxs("html", {
    lang: "en",
    children: [/* @__PURE__ */ jsxs("head", {
      children: [/* @__PURE__ */ jsx("meta", {
        charSet: "utf-8"
      }), /* @__PURE__ */ jsx("meta", {
        name: "viewport",
        content: "width=device-width, initial-scale=1"
      }), /* @__PURE__ */ jsx(Meta, {}), /* @__PURE__ */ jsx(Links, {})]
    }), /* @__PURE__ */ jsxs("body", {
      children: [children, /* @__PURE__ */ jsx(ScrollRestoration, {}), /* @__PURE__ */ jsx(Scripts, {})]
    })]
  });
}
const root = UNSAFE_withComponentProps(function App() {
  return /* @__PURE__ */ jsx(Outlet, {});
});
const ErrorBoundary = UNSAFE_withErrorBoundaryProps(function ErrorBoundary2({
  error
}) {
  let message = "Oops!";
  let details = "An unexpected error occurred.";
  let stack;
  if (isRouteErrorResponse(error)) {
    message = error.status === 404 ? "404" : "Error";
    details = error.status === 404 ? "The requested page could not be found." : error.statusText || details;
  }
  return /* @__PURE__ */ jsxs("main", {
    className: "pt-16 p-4 container mx-auto",
    children: [/* @__PURE__ */ jsx("h1", {
      children: message
    }), /* @__PURE__ */ jsx("p", {
      children: details
    }), stack]
  });
});
const route0 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  ErrorBoundary,
  Layout,
  default: root,
  links
}, Symbol.toStringTag, { value: "Module" }));
const logoOS = "/assets/logo-os-DcdGc2XL.jpg";
const logoOS2 = "/assets/logo-os-light-gJsW2AC5.jpg";
function Welcome() {
  return /* @__PURE__ */ jsx("main", { className: "flex items-center justify-center pt-16 pb-4", children: /* @__PURE__ */ jsxs("div", { className: "flex-1 flex flex-col items-center gap-16 min-h-0", children: [
    /* @__PURE__ */ jsx("header", { className: "flex flex-col items-center gap-9", children: /* @__PURE__ */ jsxs("div", { className: "w-[500px] max-w-[100vw] p-4", children: [
      /* @__PURE__ */ jsx(
        "img",
        {
          src: logoOS2,
          alt: "OS project2",
          className: "block w-full dark:hidden"
        }
      ),
      /* @__PURE__ */ jsx(
        "img",
        {
          src: logoOS,
          alt: "OS project",
          className: "hidden w-full dark:block"
        }
      )
    ] }) }),
    /* @__PURE__ */ jsx("div", { className: "max-w-[300px] w-full space-y-6 px-4", children: /* @__PURE__ */ jsxs("nav", { className: "rounded-3xl border border-gray-200 p-6 dark:border-gray-700 space-y-4", children: [
      /* @__PURE__ */ jsx("p", { className: "leading-6 text-gray-700 dark:text-gray-200 text-center", children: "What's next?" }),
      /* @__PURE__ */ jsx("ul", { children: resources.map(({ href: href2, text, icon }) => /* @__PURE__ */ jsx("li", { children: /* @__PURE__ */ jsxs(
        Link,
        {
          className: "group flex items-center gap-3 self-stretch p-3 leading-normal text-blue-700 hover:underline dark:text-blue-500",
          to: href2,
          target: "",
          rel: "noreferrer",
          children: [
            icon,
            text
          ]
        }
      ) }, href2)) })
    ] }) })
  ] }) });
}
const resources = [
  {
    href: "/page",
    text: "MultiModalAI fusion OS project",
    icon: /* @__PURE__ */ jsx(
      "svg",
      {
        xmlns: "http://www.w3.org/2000/svg",
        width: "24",
        height: "20",
        viewBox: "0 0 20 20",
        fill: "none",
        className: "stroke-gray-600 group-hover:stroke-current dark:stroke-gray-300",
        children: /* @__PURE__ */ jsx(
          "path",
          {
            d: "M9.99981 10.0751V9.99992M17.4688 17.4688C15.889 19.0485 11.2645 16.9853 7.13958 12.8604C3.01467 8.73546 0.951405 4.11091 2.53116 2.53116C4.11091 0.951405 8.73546 3.01467 12.8604 7.13958C16.9853 11.2645 19.0485 15.889 17.4688 17.4688ZM2.53132 17.4688C0.951566 15.8891 3.01483 11.2645 7.13974 7.13963C11.2647 3.01471 15.8892 0.951453 17.469 2.53121C19.0487 4.11096 16.9854 8.73551 12.8605 12.8604C8.73562 16.9853 4.11107 19.0486 2.53132 17.4688Z",
            strokeWidth: "1.5",
            strokeLinecap: "round"
          }
        )
      }
    )
  },
  {
    href: "/chat",
    text: "Go AIchat",
    icon: /* @__PURE__ */ jsx(
      "svg",
      {
        xmlns: "http://www.w3.org/2000/svg",
        width: "24",
        height: "20",
        viewBox: "0 0 20 20",
        fill: "none",
        className: "stroke-gray-600 group-hover:stroke-current dark:stroke-gray-300",
        children: /* @__PURE__ */ jsx(
          "path",
          {
            d: "M9.99981 10.0751V9.99992M17.4688 17.4688C15.889 19.0485 11.2645 16.9853 7.13958 12.8604C3.01467 8.73546 0.951405 4.11091 2.53116 2.53116C4.11091 0.951405 8.73546 3.01467 12.8604 7.13958C16.9853 11.2645 19.0485 15.889 17.4688 17.4688ZM2.53132 17.4688C0.951566 15.8891 3.01483 11.2645 7.13974 7.13963C11.2647 3.01471 15.8892 0.951453 17.469 2.53121C19.0487 4.11096 16.9854 8.73551 12.8605 12.8604C8.73562 16.9853 4.11107 19.0486 2.53132 17.4688Z",
            strokeWidth: "1.5",
            strokeLinecap: "round"
          }
        )
      }
    )
  },
  {
    href: "https://github.com/yuugiouduele/math",
    text: "Join OS projects",
    icon: /* @__PURE__ */ jsx(
      "svg",
      {
        xmlns: "http://www.w3.org/2000/svg",
        width: "24",
        height: "20",
        viewBox: "0 0 24 20",
        fill: "none",
        className: "stroke-gray-600 group-hover:stroke-current dark:stroke-gray-300",
        children: /* @__PURE__ */ jsx(
          "path",
          {
            d: "M15.0686 1.25995L14.5477 1.17423L14.2913 1.63578C14.1754 1.84439 14.0545 2.08275 13.9422 2.31963C12.6461 2.16488 11.3406 2.16505 10.0445 2.32014C9.92822 2.08178 9.80478 1.84975 9.67412 1.62413L9.41449 1.17584L8.90333 1.25995C7.33547 1.51794 5.80717 1.99419 4.37748 2.66939L4.19 2.75793L4.07461 2.93019C1.23864 7.16437 0.46302 11.3053 0.838165 15.3924L0.868838 15.7266L1.13844 15.9264C2.81818 17.1714 4.68053 18.1233 6.68582 18.719L7.18892 18.8684L7.50166 18.4469C7.96179 17.8268 8.36504 17.1824 8.709 16.4944L8.71099 16.4904C10.8645 17.0471 13.128 17.0485 15.2821 16.4947C15.6261 17.1826 16.0293 17.8269 16.4892 18.4469L16.805 18.8725L17.3116 18.717C19.3056 18.105 21.1876 17.1751 22.8559 15.9238L23.1224 15.724L23.1528 15.3923C23.5873 10.6524 22.3579 6.53306 19.8947 2.90714L19.7759 2.73227L19.5833 2.64518C18.1437 1.99439 16.6386 1.51826 15.0686 1.25995ZM16.6074 10.7755L16.6074 10.7756C16.5934 11.6409 16.0212 12.1444 15.4783 12.1444C14.9297 12.1444 14.3493 11.6173 14.3493 10.7877C14.3493 9.94885 14.9378 9.41192 15.4783 9.41192C16.0471 9.41192 16.6209 9.93851 16.6074 10.7755ZM8.49373 12.1444C7.94513 12.1444 7.36471 11.6173 7.36471 10.7877C7.36471 9.94885 7.95323 9.41192 8.49373 9.41192C9.06038 9.41192 9.63892 9.93712 9.6417 10.7815C9.62517 11.6239 9.05462 12.1444 8.49373 12.1444Z",
            strokeWidth: "1.5"
          }
        )
      }
    )
  }
];
function meta$5({}) {
  return [{
    title: "New React Router App"
  }, {
    name: "description",
    content: "Welcome to React Router!"
  }];
}
const home = UNSAFE_withComponentProps(function Home() {
  return /* @__PURE__ */ jsx(Welcome, {});
});
const route1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: home,
  meta: meta$5
}, Symbol.toStringTag, { value: "Module" }));
const ReactCameraComponent = () => {
  const webcamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioRef = useRef(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [audioChunks, setAudioChunks] = useState([]);
  const [audioURL, setAudioURL] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [stream, setStream] = useState(null);
  const [ready, setReady] = useState(false);
  const handleGetUserMedia = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: { facingMode: "user" }
      });
      setStream(mediaStream);
      setReady(true);
    } catch (error) {
      console.error("getUserMedia error:", error);
      setReady(false);
    }
  }, []);
  useEffect(() => {
    handleGetUserMedia();
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [handleGetUserMedia, stream]);
  useEffect(() => {
    if (stream) {
      const options = { mimeType: "audio/webm" };
      const recorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
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
  const startRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "inactive") {
      mediaRecorderRef.current.start();
      setIsRecording(true);
      setAudioChunks([]);
      setAudioURL(null);
    }
  };
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };
  const capture = useCallback(() => {
    var _a;
    const image = (_a = webcamRef.current) == null ? void 0 : _a.getScreenshot();
    if (image) {
      setImageSrc(image);
    }
  }, []);
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
  return /* @__PURE__ */ jsx("div", { className: "min-h-screen bg-gray-900 text-white p-6", children: /* @__PURE__ */ jsxs("div", { className: "max-w-4xl mx-auto", children: [
    /* @__PURE__ */ jsx("h1", { className: "text-3xl font-bold text-center mb-8 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent", children: "React Camera & Audio Recorder" }),
    /* @__PURE__ */ jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-2 gap-8", children: [
      /* @__PURE__ */ jsxs("div", { className: "bg-gray-800 rounded-xl p-6 shadow-2xl", children: [
        /* @__PURE__ */ jsxs("h2", { className: "text-xl font-semibold mb-4 flex items-center gap-2", children: [
          /* @__PURE__ */ jsx(Camera, { className: "w-5 h-5" }),
          "Camera"
        ] }),
        /* @__PURE__ */ jsx("div", { className: "mb-4", children: /* @__PURE__ */ jsx(
          Webcam,
          {
            audio: false,
            ref: webcamRef,
            screenshotFormat: "image/jpeg",
            width: 400,
            height: 300,
            videoConstraints: { facingMode: "user" },
            className: "rounded-lg border border-gray-600"
          }
        ) }),
        /* @__PURE__ */ jsxs(
          "button",
          {
            onClick: capture,
            disabled: !ready,
            className: "w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-4 py-2 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2",
            children: [
              /* @__PURE__ */ jsx(Camera, { className: "w-4 h-4" }),
              "Take Photo"
            ]
          }
        )
      ] }),
      /* @__PURE__ */ jsxs("div", { className: "bg-gray-800 rounded-xl p-6 shadow-2xl", children: [
        /* @__PURE__ */ jsxs("h2", { className: "text-xl font-semibold mb-4 flex items-center gap-2", children: [
          isRecording ? /* @__PURE__ */ jsx(MicOff, { className: "w-5 h-5 text-red-500" }) : /* @__PURE__ */ jsx(Mic, { className: "w-5 h-5" }),
          "Audio Recording"
        ] }),
        /* @__PURE__ */ jsx("div", { className: "mb-4 text-center", children: isRecording ? /* @__PURE__ */ jsxs("div", { className: "text-red-500", children: [
          /* @__PURE__ */ jsx("div", { className: "animate-pulse text-lg font-semibold", children: "● Recording..." }),
          /* @__PURE__ */ jsx("div", { className: "text-sm mt-2", children: "Click stop to finish recording" })
        ] }) : /* @__PURE__ */ jsxs("div", { className: "text-gray-400", children: [
          /* @__PURE__ */ jsx("div", { className: "text-lg", children: "Ready to record" }),
          /* @__PURE__ */ jsx("div", { className: "text-sm mt-2", children: "Click start to begin recording" })
        ] }) }),
        /* @__PURE__ */ jsx("div", { className: "space-y-3", children: !isRecording ? /* @__PURE__ */ jsxs(
          "button",
          {
            onClick: startRecording,
            disabled: !ready,
            className: "w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-4 py-2 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2",
            children: [
              /* @__PURE__ */ jsx(Mic, { className: "w-4 h-4" }),
              "Start Recording"
            ]
          }
        ) : /* @__PURE__ */ jsxs(
          "button",
          {
            onClick: stopRecording,
            className: "w-full bg-gray-600 hover:bg-gray-700 px-4 py-2 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2",
            children: [
              /* @__PURE__ */ jsx(MicOff, { className: "w-4 h-4" }),
              "Stop Recording"
            ]
          }
        ) })
      ] })
    ] }),
    /* @__PURE__ */ jsxs("div", { className: "mt-8 grid grid-cols-1 lg:grid-cols-2 gap-8", children: [
      imageSrc && /* @__PURE__ */ jsxs("div", { className: "bg-gray-800 rounded-xl p-6 shadow-2xl", children: [
        /* @__PURE__ */ jsx("h3", { className: "text-xl font-semibold mb-4", children: "Captured Photo" }),
        /* @__PURE__ */ jsx("div", { className: "mb-4", children: /* @__PURE__ */ jsx("img", { src: imageSrc, alt: "Captured", className: "w-full rounded-lg border border-gray-600" }) }),
        /* @__PURE__ */ jsxs(
          "button",
          {
            onClick: downloadImage,
            className: "w-full bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2",
            children: [
              /* @__PURE__ */ jsx(Download, { className: "w-4 h-4" }),
              "Download Image"
            ]
          }
        )
      ] }),
      audioURL && /* @__PURE__ */ jsxs("div", { className: "bg-gray-800 rounded-xl p-6 shadow-2xl", children: [
        /* @__PURE__ */ jsx("h3", { className: "text-xl font-semibold mb-4", children: "Recorded Audio" }),
        /* @__PURE__ */ jsx("div", { className: "mb-4", children: /* @__PURE__ */ jsx(
          "audio",
          {
            ref: audioRef,
            src: audioURL,
            onPlay: () => setIsPlaying(true),
            onPause: () => setIsPlaying(false),
            onEnded: () => setIsPlaying(false),
            className: "w-full",
            controls: true
          }
        ) }),
        /* @__PURE__ */ jsxs("div", { className: "space-y-2", children: [
          /* @__PURE__ */ jsxs(
            "button",
            {
              onClick: toggleAudioPlayback,
              className: "w-full bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2",
              children: [
                isPlaying ? /* @__PURE__ */ jsx(Pause, { className: "w-4 h-4" }) : /* @__PURE__ */ jsx(Play, { className: "w-4 h-4" }),
                isPlaying ? "Pause" : "Play",
                " Audio"
              ]
            }
          ),
          /* @__PURE__ */ jsxs(
            "button",
            {
              onClick: downloadAudio,
              className: "w-full bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2",
              children: [
                /* @__PURE__ */ jsx(Download, { className: "w-4 h-4" }),
                "Download Audio"
              ]
            }
          )
        ] })
      ] })
    ] }),
    /* @__PURE__ */ jsx("div", { className: "mt-8 text-center", children: /* @__PURE__ */ jsxs(
      "div",
      {
        className: `inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm ${ready ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"}`,
        children: [
          /* @__PURE__ */ jsx("div", { className: `w-2 h-2 rounded-full ${ready ? "bg-green-400" : "bg-red-400"}` }),
          ready ? "Camera & Microphone Ready" : "Waiting for permissions..."
        ]
      }
    ) })
  ] }) });
};
function meta$4({}) {
  return [{
    title: "New React Router App"
  }, {
    name: "description",
    content: "Welcome to React Router!"
  }];
}
const camera = UNSAFE_withComponentProps(function Home2() {
  return /* @__PURE__ */ jsx(ReactCameraComponent, {});
});
const route2 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: camera,
  meta: meta$4
}, Symbol.toStringTag, { value: "Module" }));
const SelfManualComponent = () => {
  const [state, setState] = useState({
    searchTerm: "",
    selectedCategory: "all",
    expandedSection: null,
    filterSeverity: "all",
    filterUrgency: "all"
  });
  const cautions = [
    {
      id: "1",
      title: "過度な完璧主義",
      description: "デッドラインよりも品質を優先しがちです。適度な妥協点を見つけることが重要です。",
      severity: "high"
    },
    {
      id: "2",
      title: "コミュニケーション頻度",
      description: "集中時間が必要なため、定期的な進捗報告のスケジュールを設定してください。",
      severity: "medium"
    },
    {
      id: "3",
      title: "新技術への関心",
      description: "新しい技術に興味を持ちやすいため、プロジェクトスコープの管理に注意が必要です。",
      severity: "low"
    }
  ];
  const companyTypes = [
    {
      id: "1",
      name: "スタートアップ企業",
      description: "柔軟性と成長志向の環境",
      compatibility: 95,
      features: ["高い自由度", "幅広い業務経験", "急速な成長"]
    },
    {
      id: "2",
      name: "テック企業",
      description: "技術力重視の開発環境",
      compatibility: 88,
      features: ["最新技術", "イノベーション", "エンジニア文化"]
    },
    {
      id: "3",
      name: "大手企業",
      description: "安定した環境でのプロダクト開発",
      compatibility: 72,
      features: ["安定性", "体系的な研修", "チーム開発"]
    }
  ];
  const benefits = [
    {
      id: "1",
      title: "フルスタック開発",
      description: "フロントエンドからバックエンドまで一貫した開発が可能",
      category: "technical",
      impact: 9
    },
    {
      id: "2",
      title: "問題解決能力",
      description: "複雑な課題を分析し、効率的なソリューションを提供",
      category: "technical",
      impact: 8
    },
    {
      id: "3",
      title: "チームコラボレーション",
      description: "異なる職種のメンバーと効果的に協働できる",
      category: "communication",
      impact: 7
    }
  ];
  const pricing = [
    {
      id: "1",
      name: "コンサルティング",
      rate: 8e3,
      currency: "JPY",
      unit: "hour",
      description: "技術相談・アーキテクチャ設計"
    },
    {
      id: "2",
      name: "開発業務",
      rate: 6e4,
      currency: "JPY",
      unit: "day",
      description: "実装・テスト・デプロイメント"
    },
    {
      id: "3",
      name: "プロジェクト管理",
      rate: 5e5,
      currency: "JPY",
      unit: "month",
      description: "チームリード・プロジェクト統括"
    }
  ];
  const consultations = [
    {
      id: "1",
      topic: "キャリアパス相談",
      description: "技術者としてのキャリア設計についてアドバイス",
      urgency: "medium",
      category: "career"
    },
    {
      id: "2",
      topic: "アーキテクチャ設計",
      description: "システム設計やマイクロサービス化の検討",
      urgency: "high",
      category: "technical"
    },
    {
      id: "3",
      topic: "チーム構築",
      description: "開発チームの組織化と効率化",
      urgency: "low",
      category: "business"
    }
  ];
  const handleSearchChange = (e) => {
    setState((prev) => ({ ...prev, searchTerm: e.target.value }));
  };
  const handleCategoryChange = (e) => {
    setState((prev) => ({ ...prev, selectedCategory: e.target.value }));
  };
  const handleSectionToggle = (section) => (e) => {
    e.preventDefault();
    setState((prev) => ({
      ...prev,
      expandedSection: prev.expandedSection === section ? null : section
    }));
  };
  const handleFilterChange = (filterType) => (e) => {
    setState((prev) => ({
      ...prev,
      [filterType === "severity" ? "filterSeverity" : "filterUrgency"]: e.target.value
    }));
  };
  const clearFilters = (e) => {
    e.preventDefault();
    setState((prev) => ({
      ...prev,
      searchTerm: "",
      selectedCategory: "all",
      filterSeverity: "all",
      filterUrgency: "all"
    }));
  };
  const filteredCautions = useMemo(() => {
    return cautions.filter(
      (caution) => caution.title.toLowerCase().includes(state.searchTerm.toLowerCase()) || caution.description.toLowerCase().includes(state.searchTerm.toLowerCase())
    ).filter(
      (caution) => state.filterSeverity === "all" || caution.severity === state.filterSeverity
    );
  }, [state.searchTerm, state.filterSeverity]);
  const filteredBenefits = useMemo(() => {
    return benefits.filter(
      (benefit) => state.selectedCategory === "all" || benefit.category === state.selectedCategory
    ).filter(
      (benefit) => benefit.title.toLowerCase().includes(state.searchTerm.toLowerCase())
    ).sort((a, b) => b.impact - a.impact);
  }, [state.selectedCategory, state.searchTerm]);
  const filteredConsultations = useMemo(() => {
    return consultations.filter(
      (consultation) => consultation.topic.toLowerCase().includes(state.searchTerm.toLowerCase())
    ).filter(
      (consultation) => state.filterUrgency === "all" || consultation.urgency === state.filterUrgency
    );
  }, [state.searchTerm, state.filterUrgency]);
  const getSeverityColor = (severity) => {
    const colorMap = {
      high: "bg-red-900 text-red-200 border-red-700",
      medium: "bg-yellow-900 text-yellow-200 border-yellow-700",
      low: "bg-green-900 text-green-200 border-green-700"
    };
    return colorMap[severity] || "bg-gray-700 text-gray-300 border-gray-600";
  };
  const getUrgencyColor = (urgency) => {
    const colorMap = {
      high: "border-l-red-400",
      medium: "border-l-yellow-400",
      low: "border-l-blue-400"
    };
    return colorMap[urgency] || "border-l-gray-400";
  };
  return /* @__PURE__ */ jsx("div", { className: "min-h-screen bg-gradient-to-br from-gray-900 via-slate-800 to-gray-900 px-8 py-4 sm:px-12 sm:py-6 lg:px-16 lg:py-8", children: /* @__PURE__ */ jsxs("div", { className: "max-w-7xl mx-auto", children: [
    /* @__PURE__ */ jsxs("div", { className: "text-center mb-6 sm:mb-8 lg:mb-10", children: [
      /* @__PURE__ */ jsx("h1", { className: "text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-2 sm:mb-4 px-2", children: "自己取り扱い説明書" }),
      /* @__PURE__ */ jsx("p", { className: "text-sm sm:text-base md:text-lg text-gray-300 max-w-xl lg:max-w-2xl mx-auto px-4 leading-relaxed", children: "効果的な協働のためのガイドライン・スキル・料金体系" })
    ] }),
    /* @__PURE__ */ jsx("div", { className: "bg-gray-800 border border-gray-700 rounded-xl shadow-lg p-4 sm:p-6 mb-6 sm:mb-8", children: /* @__PURE__ */ jsxs("div", { className: "flex flex-col gap-3 sm:gap-4", children: [
      /* @__PURE__ */ jsxs("div", { className: "relative w-full", children: [
        /* @__PURE__ */ jsx(Search, { className: "absolute left-2 sm:left-3 top-2.5 sm:top-3 h-4 w-4 sm:h-5 sm:w-5 text-gray-400 z-10" }),
        /* @__PURE__ */ jsx(
          "input",
          {
            type: "text",
            placeholder: "検索...",
            value: state.searchTerm,
            onChange: handleSearchChange,
            className: "w-full pl-8 sm:pl-10 pr-3 sm:pr-4 py-2 sm:py-2.5 bg-gray-700 border border-gray-600 text-white placeholder-gray-400 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm sm:text-base",
            autoComplete: "off",
            spellCheck: "false"
          }
        )
      ] }),
      /* @__PURE__ */ jsxs("div", { className: "flex flex-col sm:flex-row gap-2 sm:gap-3", children: [
        /* @__PURE__ */ jsxs(
          "select",
          {
            value: state.selectedCategory,
            onChange: handleCategoryChange,
            className: "flex-1 sm:flex-none px-3 sm:px-4 py-2 sm:py-2.5 bg-gray-700 border border-gray-600 text-white rounded-lg focus:ring-2 focus:ring-blue-500 text-sm sm:text-base",
            autoComplete: "off",
            children: [
              /* @__PURE__ */ jsx("option", { value: "all", children: "全カテゴリ" }),
              /* @__PURE__ */ jsx("option", { value: "technical", children: "技術" }),
              /* @__PURE__ */ jsx("option", { value: "communication", children: "コミュニケーション" }),
              /* @__PURE__ */ jsx("option", { value: "leadership", children: "リーダーシップ" })
            ]
          }
        ),
        /* @__PURE__ */ jsxs(
          "button",
          {
            onClick: clearFilters,
            className: "px-3 sm:px-4 py-2 sm:py-2.5 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors flex items-center justify-center gap-2 text-sm sm:text-base font-medium",
            children: [
              /* @__PURE__ */ jsx(X, { className: "h-4 w-4" }),
              /* @__PURE__ */ jsx("span", { children: "クリア" })
            ]
          }
        )
      ] })
    ] }) }),
    /* @__PURE__ */ jsxs("div", { className: "space-y-6 sm:space-y-8", children: [
      /* @__PURE__ */ jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6 lg:gap-8", children: [
        /* @__PURE__ */ jsxs("section", { className: "bg-gray-800 border border-gray-700 rounded-xl shadow-lg overflow-hidden", children: [
          /* @__PURE__ */ jsxs(
            "button",
            {
              onClick: handleSectionToggle("cautions"),
              className: "w-full p-4 sm:p-6 flex items-center justify-between hover:bg-gray-700 transition-colors text-white",
              children: [
                /* @__PURE__ */ jsxs("div", { className: "flex items-center gap-2 sm:gap-3", children: [
                  /* @__PURE__ */ jsx(AlertTriangle, { className: "h-5 w-5 sm:h-6 sm:w-6 text-red-400 flex-shrink-0" }),
                  /* @__PURE__ */ jsx("h2", { className: "text-lg sm:text-xl font-semibold text-white", children: "注意点" })
                ] }),
                state.expandedSection === "cautions" ? /* @__PURE__ */ jsx(ChevronUp, { className: "h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" }) : /* @__PURE__ */ jsx(ChevronDown, { className: "h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" })
              ]
            }
          ),
          state.expandedSection === "cautions" && /* @__PURE__ */ jsxs("div", { className: "px-4 sm:px-6 pb-4 sm:pb-6", children: [
            /* @__PURE__ */ jsx("div", { className: "mb-3 sm:mb-4", children: /* @__PURE__ */ jsxs(
              "select",
              {
                value: state.filterSeverity,
                onChange: handleFilterChange("severity"),
                className: "px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700 border border-gray-600 text-white rounded-lg text-xs sm:text-sm w-full sm:w-auto",
                autoComplete: "off",
                children: [
                  /* @__PURE__ */ jsx("option", { value: "all", children: "全重要度" }),
                  /* @__PURE__ */ jsx("option", { value: "high", children: "高" }),
                  /* @__PURE__ */ jsx("option", { value: "medium", children: "中" }),
                  /* @__PURE__ */ jsx("option", { value: "low", children: "低" })
                ]
              }
            ) }),
            /* @__PURE__ */ jsx("div", { className: "space-y-3 sm:space-y-4", children: filteredCautions.map((caution) => /* @__PURE__ */ jsxs("div", { className: "border border-gray-600 bg-gray-700 rounded-lg p-3 sm:p-4", children: [
              /* @__PURE__ */ jsxs("div", { className: "flex items-start gap-2 sm:gap-3 mb-2", children: [
                /* @__PURE__ */ jsx("span", { className: `px-2 py-1 rounded-full text-xs font-medium border flex-shrink-0 ${getSeverityColor(caution.severity)}`, children: caution.severity === "high" ? "高" : caution.severity === "medium" ? "中" : "低" }),
                /* @__PURE__ */ jsx("h3", { className: "font-semibold text-white text-sm sm:text-base leading-tight", children: caution.title })
              ] }),
              /* @__PURE__ */ jsx("p", { className: "text-gray-300 text-xs sm:text-sm leading-relaxed", children: caution.description })
            ] }, caution.id)) })
          ] })
        ] }),
        /* @__PURE__ */ jsxs("section", { className: "bg-gray-800 border border-gray-700 rounded-xl shadow-lg overflow-hidden", children: [
          /* @__PURE__ */ jsxs(
            "button",
            {
              onClick: handleSectionToggle("companies"),
              className: "w-full p-4 sm:p-6 flex items-center justify-between hover:bg-gray-700 transition-colors text-white",
              children: [
                /* @__PURE__ */ jsxs("div", { className: "flex items-center gap-2 sm:gap-3", children: [
                  /* @__PURE__ */ jsx(Building2, { className: "h-5 w-5 sm:h-6 sm:w-6 text-blue-400 flex-shrink-0" }),
                  /* @__PURE__ */ jsx("h2", { className: "text-lg sm:text-xl font-semibold text-white", children: "おすすめ企業型" })
                ] }),
                state.expandedSection === "companies" ? /* @__PURE__ */ jsx(ChevronUp, { className: "h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" }) : /* @__PURE__ */ jsx(ChevronDown, { className: "h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" })
              ]
            }
          ),
          state.expandedSection === "companies" && /* @__PURE__ */ jsx("div", { className: "px-4 sm:px-6 pb-4 sm:pb-6 space-y-3 sm:space-y-4", children: companyTypes.map((company) => /* @__PURE__ */ jsxs("div", { className: "border border-gray-600 bg-gray-700 rounded-lg p-3 sm:p-4", children: [
            /* @__PURE__ */ jsxs("div", { className: "flex flex-col sm:flex-row sm:items-center justify-between mb-3 gap-2", children: [
              /* @__PURE__ */ jsx("h3", { className: "font-semibold text-white text-sm sm:text-base", children: company.name }),
              /* @__PURE__ */ jsxs("div", { className: "flex items-center gap-2", children: [
                /* @__PURE__ */ jsx("div", { className: "w-12 sm:w-16 bg-gray-600 rounded-full h-1.5 sm:h-2", children: /* @__PURE__ */ jsx(
                  "div",
                  {
                    className: "bg-blue-500 h-1.5 sm:h-2 rounded-full transition-all duration-300",
                    style: { width: `${company.compatibility}%` }
                  }
                ) }),
                /* @__PURE__ */ jsxs("span", { className: "text-xs sm:text-sm font-medium text-blue-400 min-w-[3rem]", children: [
                  company.compatibility,
                  "%"
                ] })
              ] })
            ] }),
            /* @__PURE__ */ jsx("p", { className: "text-gray-300 text-xs sm:text-sm mb-3 leading-relaxed", children: company.description }),
            /* @__PURE__ */ jsx("div", { className: "flex flex-wrap gap-1.5 sm:gap-2", children: company.features.map((feature, index) => /* @__PURE__ */ jsx("span", { className: "px-2 py-1 bg-blue-900 text-blue-200 rounded-full text-xs", children: feature }, index)) })
          ] }, company.id)) })
        ] })
      ] }),
      /* @__PURE__ */ jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6 lg:gap-8", children: [
        /* @__PURE__ */ jsxs("section", { className: "bg-gray-800 border border-gray-700 rounded-xl shadow-lg overflow-hidden", children: [
          /* @__PURE__ */ jsxs(
            "button",
            {
              onClick: handleSectionToggle("benefits"),
              className: "w-full p-4 sm:p-6 flex items-center justify-between hover:bg-gray-700 transition-colors text-white",
              children: [
                /* @__PURE__ */ jsxs("div", { className: "flex items-center gap-2 sm:gap-3", children: [
                  /* @__PURE__ */ jsx(TrendingUp, { className: "h-5 w-5 sm:h-6 sm:w-6 text-green-400 flex-shrink-0" }),
                  /* @__PURE__ */ jsx("h2", { className: "text-lg sm:text-xl font-semibold text-white", children: "提供できる価値" })
                ] }),
                state.expandedSection === "benefits" ? /* @__PURE__ */ jsx(ChevronUp, { className: "h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" }) : /* @__PURE__ */ jsx(ChevronDown, { className: "h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" })
              ]
            }
          ),
          state.expandedSection === "benefits" && /* @__PURE__ */ jsx("div", { className: "px-4 sm:px-6 pb-4 sm:pb-6 space-y-3 sm:space-y-4", children: filteredBenefits.map((benefit) => /* @__PURE__ */ jsxs("div", { className: "border border-gray-600 bg-gray-700 rounded-lg p-3 sm:p-4", children: [
            /* @__PURE__ */ jsxs("div", { className: "flex flex-col sm:flex-row sm:items-center justify-between mb-2 gap-2", children: [
              /* @__PURE__ */ jsx("h3", { className: "font-semibold text-white text-sm sm:text-base", children: benefit.title }),
              /* @__PURE__ */ jsx("div", { className: "flex items-center gap-1", children: [...Array(10)].map((_, i) => /* @__PURE__ */ jsx(
                "div",
                {
                  className: `w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-full transition-all duration-200 ${i < benefit.impact ? "bg-green-400" : "bg-gray-600"}`
                },
                i
              )) })
            ] }),
            /* @__PURE__ */ jsx("p", { className: "text-gray-300 text-xs sm:text-sm leading-relaxed", children: benefit.description })
          ] }, benefit.id)) })
        ] }),
        /* @__PURE__ */ jsxs("section", { className: "bg-gray-800 border border-gray-700 rounded-xl shadow-lg overflow-hidden", children: [
          /* @__PURE__ */ jsxs(
            "button",
            {
              onClick: handleSectionToggle("pricing"),
              className: "w-full p-4 sm:p-6 flex items-center justify-between hover:bg-gray-700 transition-colors text-white",
              children: [
                /* @__PURE__ */ jsxs("div", { className: "flex items-center gap-2 sm:gap-3", children: [
                  /* @__PURE__ */ jsx(DollarSign, { className: "h-5 w-5 sm:h-6 sm:w-6 text-yellow-400 flex-shrink-0" }),
                  /* @__PURE__ */ jsx("h2", { className: "text-lg sm:text-xl font-semibold text-white", children: "料金体系" })
                ] }),
                state.expandedSection === "pricing" ? /* @__PURE__ */ jsx(ChevronUp, { className: "h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" }) : /* @__PURE__ */ jsx(ChevronDown, { className: "h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" })
              ]
            }
          ),
          state.expandedSection === "pricing" && /* @__PURE__ */ jsx("div", { className: "px-4 sm:px-6 pb-4 sm:pb-6 space-y-3 sm:space-y-4", children: pricing.map((tier) => /* @__PURE__ */ jsxs("div", { className: "border border-gray-600 bg-gray-700 rounded-lg p-3 sm:p-4", children: [
            /* @__PURE__ */ jsxs("div", { className: "flex flex-col sm:flex-row sm:items-center justify-between mb-2", children: [
              /* @__PURE__ */ jsx("h3", { className: "font-semibold text-white text-sm sm:text-base", children: tier.name }),
              /* @__PURE__ */ jsxs("div", { className: "text-left sm:text-right", children: [
                /* @__PURE__ */ jsxs("div", { className: "text-xl sm:text-2xl font-bold text-green-400", children: [
                  "¥",
                  tier.rate.toLocaleString()
                ] }),
                /* @__PURE__ */ jsxs("div", { className: "text-xs sm:text-sm text-gray-400", children: [
                  "/ ",
                  tier.unit === "hour" ? "時間" : tier.unit === "day" ? "日" : "月"
                ] })
              ] })
            ] }),
            /* @__PURE__ */ jsx("p", { className: "text-gray-300 text-xs sm:text-sm leading-relaxed", children: tier.description })
          ] }, tier.id)) })
        ] })
      ] })
    ] }),
    /* @__PURE__ */ jsxs("section", { className: "mt-6 sm:mt-8 bg-gray-800 border border-gray-700 rounded-xl shadow-lg overflow-hidden", children: [
      /* @__PURE__ */ jsxs(
        "button",
        {
          onClick: handleSectionToggle("consultations"),
          className: "w-full p-4 sm:p-6 flex items-center justify-between hover:bg-gray-700 transition-colors text-white",
          children: [
            /* @__PURE__ */ jsxs("div", { className: "flex items-center gap-2 sm:gap-3", children: [
              /* @__PURE__ */ jsx(MessageCircle, { className: "h-5 w-5 sm:h-6 sm:w-6 text-purple-400 flex-shrink-0" }),
              /* @__PURE__ */ jsx("h2", { className: "text-lg sm:text-xl font-semibold text-white", children: "相談可能な事項" })
            ] }),
            state.expandedSection === "consultations" ? /* @__PURE__ */ jsx(ChevronUp, { className: "h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" }) : /* @__PURE__ */ jsx(ChevronDown, { className: "h-4 w-4 sm:h-5 sm:w-5 text-gray-300 flex-shrink-0" })
          ]
        }
      ),
      state.expandedSection === "consultations" && /* @__PURE__ */ jsxs("div", { className: "px-4 sm:px-6 pb-4 sm:pb-6", children: [
        /* @__PURE__ */ jsx("div", { className: "mb-3 sm:mb-4", children: /* @__PURE__ */ jsxs(
          "select",
          {
            value: state.filterUrgency,
            onChange: handleFilterChange("urgency"),
            className: "px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-700 border border-gray-600 text-white rounded-lg text-xs sm:text-sm w-full sm:w-auto",
            autoComplete: "off",
            children: [
              /* @__PURE__ */ jsx("option", { value: "all", children: "全緊急度" }),
              /* @__PURE__ */ jsx("option", { value: "high", children: "高" }),
              /* @__PURE__ */ jsx("option", { value: "medium", children: "中" }),
              /* @__PURE__ */ jsx("option", { value: "low", children: "低" })
            ]
          }
        ) }),
        /* @__PURE__ */ jsx("div", { className: "grid grid-cols-1 lg:grid-cols-2 2xl:grid-cols-3 gap-3 sm:gap-4", children: filteredConsultations.map((consultation) => /* @__PURE__ */ jsxs("div", { className: `border-l-4 bg-gray-700 border border-gray-600 rounded-lg p-3 sm:p-4 ${getUrgencyColor(consultation.urgency)}`, children: [
          /* @__PURE__ */ jsx("h3", { className: "font-semibold text-white mb-2 text-sm sm:text-base", children: consultation.topic }),
          /* @__PURE__ */ jsx("p", { className: "text-gray-300 text-xs sm:text-sm mb-3 leading-relaxed", children: consultation.description }),
          /* @__PURE__ */ jsxs("div", { className: "flex flex-wrap gap-1.5 sm:gap-2", children: [
            /* @__PURE__ */ jsx("span", { className: "px-2 py-1 bg-gray-600 text-gray-300 rounded-full text-xs font-medium", children: consultation.category }),
            /* @__PURE__ */ jsx("span", { className: `px-2 py-1 rounded-full text-xs font-medium ${consultation.urgency === "high" ? "bg-red-900 text-red-200" : consultation.urgency === "medium" ? "bg-yellow-900 text-yellow-200" : "bg-blue-900 text-blue-200"}`, children: consultation.urgency === "high" ? "急" : consultation.urgency === "medium" ? "中" : "低" })
          ] })
        ] }, consultation.id)) })
      ] })
    ] })
  ] }) });
};
function meta$3({}) {
  return [{
    title: "New React Router App"
  }, {
    name: "description",
    content: "Welcome to React Router!"
  }];
}
const manual = UNSAFE_withComponentProps(function Home3() {
  return /* @__PURE__ */ jsx(SelfManualComponent, {});
});
const route3 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: manual,
  meta: meta$3
}, Symbol.toStringTag, { value: "Module" }));
const SlideCard = ({ slide, index, currentSlide, isDarkMode, onModalOpen, isTransitioning }) => {
  const isActive = index === currentSlide;
  const isPrev = index === (currentSlide - 1 + 3) % 3;
  return /* @__PURE__ */ jsxs(
    "div",
    {
      className: `absolute w-full h-full transition-all duration-1000 ease-in-out ${isActive ? "opacity-100 z-10 scale-100" : isPrev ? "opacity-0 z-0 scale-90" : "opacity-0 z-0 scale-90"}`,
      style: {
        transform: isActive ? "rotateY(0deg) scale(1)" : isPrev ? "rotateY(90deg) scale(0.9)" : "rotateY(-90deg) scale(0.9)"
      },
      children: [
        /* @__PURE__ */ jsx(
          "img",
          {
            src: slide.image,
            alt: slide.title,
            className: "w-full h-full object-cover"
          }
        ),
        /* @__PURE__ */ jsx("div", { className: "absolute inset-0 bg-gradient-to-t from-black/70 via-black/30 to-transparent flex items-center justify-center", children: /* @__PURE__ */ jsxs("div", { className: "text-center transform transition-all duration-1000", children: [
          /* @__PURE__ */ jsx("h2", { className: `text-4xl font-bold text-white mb-8 transition-all duration-1000 ${isActive ? "translate-y-0 opacity-100" : "translate-y-10 opacity-0"}`, children: slide.title }),
          /* @__PURE__ */ jsx(
            "button",
            {
              onClick: () => onModalOpen(slide),
              disabled: isTransitioning,
              className: `px-6 py-3 rounded-lg font-semibold transition-all duration-500 hover:scale-105 disabled:opacity-50 ${isActive ? "translate-y-0 opacity-100" : "translate-y-10 opacity-0"} ${isDarkMode ? "bg-blue-600 hover:bg-blue-700 text-white" : "bg-blue-500 hover:bg-blue-600 text-white"}`,
              children: "詳細を見る"
            }
          )
        ] }) })
      ]
    }
  );
};
const NavItemComponent = ({ item, isActive, isDarkMode, onClick, icon: Icon }) => /* @__PURE__ */ jsxs(
  "button",
  {
    onClick,
    className: `w-full text-left p-3 rounded-lg transition-colors flex items-center space-x-3 ${isActive ? isDarkMode ? "bg-blue-700 text-white" : "bg-blue-100 text-blue-900" : isDarkMode ? "hover:bg-gray-700" : "hover:bg-gray-100"}`,
    children: [
      /* @__PURE__ */ jsx(Icon, { size: 18 }),
      /* @__PURE__ */ jsx("span", { children: item.title })
    ]
  }
);
const ProfileCard = ({ section, index, currentProfileSection, isDarkMode }) => {
  const isActive = currentProfileSection === index;
  return /* @__PURE__ */ jsxs(
    "div",
    {
      className: `p-6 rounded-lg transition-all duration-300 ${isActive ? isDarkMode ? "bg-blue-900 border-2 border-blue-400 shadow-lg transform scale-105" : "bg-blue-50 border-2 border-blue-400 shadow-lg transform scale-105" : isDarkMode ? "bg-gray-800 border border-gray-700 hover:bg-gray-750" : "bg-white border border-gray-200 hover:bg-gray-50"}`,
      children: [
        /* @__PURE__ */ jsx("h3", { className: "text-xl font-semibold mb-4 text-blue-600", children: section.title }),
        /* @__PURE__ */ jsx("p", { className: `leading-relaxed ${isDarkMode ? "text-gray-300" : "text-gray-700"}`, children: section.content })
      ]
    }
  );
};
const SearchBar$1 = ({ isSearchOpen, searchTerm, onSearchChange, isDarkMode }) => isSearchOpen ? /* @__PURE__ */ jsx("div", { className: "mt-4", children: /* @__PURE__ */ jsxs("div", { className: "relative", children: [
  /* @__PURE__ */ jsx(Search, { size: 20, className: "absolute left-3 top-3 text-gray-400" }),
  /* @__PURE__ */ jsx(
    "input",
    {
      type: "text",
      placeholder: "内容を検索... (Space to toggle)",
      value: searchTerm,
      onChange: onSearchChange,
      className: `w-full pl-10 pr-4 py-2 rounded-lg border focus:ring-2 focus:ring-blue-500 focus:border-transparent ${isDarkMode ? "bg-gray-700 border-gray-600 text-white" : "bg-white border-gray-300"}`,
      autoFocus: true
    }
  )
] }) }) : null;
const PresentationSlider = () => {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalContent, setModalContent] = useState(null);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [currentProfileSection, setCurrentProfileSection] = useState(0);
  const [isNavbarOpen, setIsNavbarOpen] = useState(true);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [pressedKeys, setPressedKeys] = useState(/* @__PURE__ */ new Set());
  const slides = [
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
  const navItems = [
    { title: "連絡", icon: Mail, action: () => window.open("mailto:contact@example.com") },
    { title: "マッチ度", icon: BarChart, action: () => console.log("マッチ度確認") },
    { title: "比較", icon: GitCompare, action: () => console.log("比較分析") }
  ];
  const profileSections = [
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
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % slides.length);
    }, 1e4);
    return () => clearInterval(interval);
  }, [slides.length]);
  const handleKeyDown = useCallback((e) => {
    setPressedKeys((prev) => new Set(prev).add(e.key));
    if (pressedKeys.has(" ") && e.key === "Enter") {
      e.preventDefault();
      setIsNavbarOpen((prev) => !prev);
    } else if (pressedKeys.has("Enter") && e.key === " ") {
      e.preventDefault();
      setIsNavbarOpen((prev) => !prev);
    }
  }, [pressedKeys]);
  const handleKeyUp = useCallback((e) => {
    setPressedKeys((prev) => {
      const newSet = new Set(prev);
      newSet.delete(e.key);
      return newSet;
    });
  }, []);
  const handleKeyPress = useCallback((e) => {
    const { key, shiftKey } = e;
    if (key === " " && shiftKey) {
      e.preventDefault();
      if (isModalOpen) {
        setIsModalOpen(false);
        setIsSearchOpen(false);
      } else {
        setCurrentProfileSection((prev) => (prev + 1) % profileSections.length);
      }
    } else if (key === " " && !shiftKey && isModalOpen) {
      e.preventDefault();
      setIsSearchOpen(!isSearchOpen);
    } else if (key === "Enter" && shiftKey) {
      e.preventDefault();
      handleDownload();
    }
  }, [isModalOpen, isSearchOpen, profileSections.length]);
  useEffect(() => {
    const keyDownHandler = (e) => handleKeyDown(e);
    const keyUpHandler = (e) => handleKeyUp(e);
    const keyPressHandler = (e) => handleKeyPress(e);
    document.addEventListener("keydown", keyDownHandler);
    document.addEventListener("keyup", keyUpHandler);
    document.addEventListener("keypress", keyPressHandler);
    return () => {
      document.removeEventListener("keydown", keyDownHandler);
      document.removeEventListener("keyup", keyUpHandler);
      document.removeEventListener("keypress", keyPressHandler);
    };
  }, [handleKeyDown, handleKeyUp, handleKeyPress]);
  const handleSlideTransition = useCallback((direction) => {
    if (isTransitioning) return;
    setIsTransitioning(true);
    if (typeof direction === "number") {
      setCurrentSlide(direction);
    } else {
      setCurrentSlide(
        (prev) => direction === "next" ? (prev + 1) % slides.length : (prev - 1 + slides.length) % slides.length
      );
    }
    setTimeout(() => setIsTransitioning(false), 1e3);
  }, [isTransitioning, slides.length]);
  const openModal = useCallback((slideData) => {
    setModalContent(slideData);
    setIsModalOpen(true);
  }, []);
  const closeModal = useCallback(() => {
    setIsModalOpen(false);
    setIsSearchOpen(false);
    setSearchTerm("");
  }, []);
  const handleDownload = useCallback(() => {
    const element = document.createElement("a");
    element.href = "data:text/plain;charset=utf-8," + encodeURIComponent("プレゼンテーション資料");
    element.download = `slide-${currentSlide + 1}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  }, [currentSlide]);
  const filteredContent = modalContent && searchTerm ? Object.entries(modalContent.details).filter(
    ([, value]) => value.toLowerCase().includes(searchTerm.toLowerCase())
  ) : modalContent ? Object.entries(modalContent.details) : [];
  const detailLabels = {
    design: "詳細設計",
    test: "テスト結果",
    infrastructure: "インフラ構成"
  };
  return /* @__PURE__ */ jsxs("div", { className: `min-h-screen transition-colors duration-300 ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-gray-900"}`, children: [
    /* @__PURE__ */ jsx(
      "button",
      {
        onClick: () => setIsDarkMode(!isDarkMode),
        className: `fixed top-4 right-4 z-50 p-2 rounded-full ${isDarkMode ? "bg-gray-700 text-yellow-400" : "bg-gray-200 text-gray-700"} hover:scale-110 transition-transform`,
        children: isDarkMode ? "☀️" : "🌙"
      }
    ),
    /* @__PURE__ */ jsx("div", { className: `fixed left-0 top-0 h-full z-40 transition-transform duration-300 ${isNavbarOpen ? "transform translate-x-0" : "transform -translate-x-full"}`, children: /* @__PURE__ */ jsx("div", { className: `h-full w-64 ${isDarkMode ? "bg-gray-800" : "bg-white"} shadow-2xl border-r-4 border-blue-500 rounded-r-3xl`, children: /* @__PURE__ */ jsxs("div", { className: "p-6", children: [
      /* @__PURE__ */ jsx("h2", { className: "text-xl font-bold mb-6", children: "ナビゲーション" }),
      /* @__PURE__ */ jsxs("nav", { className: "space-y-4", children: [
        slides.map((slide, index) => /* @__PURE__ */ jsx(
          NavItemComponent,
          {
            item: slide,
            isActive: currentSlide === index,
            isDarkMode,
            onClick: () => handleSlideTransition(index),
            icon: index === 0 ? BarChart : index === 1 ? GitCompare : Mail
          },
          slide.id
        )),
        /* @__PURE__ */ jsx("div", { className: `pt-4 border-t ${isDarkMode ? "border-gray-700" : "border-gray-300"}`, children: navItems.map((item) => /* @__PURE__ */ jsx(
          NavItemComponent,
          {
            item,
            isActive: false,
            isDarkMode,
            onClick: item.action,
            icon: item.icon
          },
          item.title
        )) }),
        /* @__PURE__ */ jsxs("div", { className: "pt-4 border-t border-gray-300", children: [
          /* @__PURE__ */ jsx("p", { className: "text-sm opacity-70 mb-2", children: "ショートカット:" }),
          /* @__PURE__ */ jsx("p", { className: "text-xs opacity-60", children: "Space + Enter : ナビ開閉" })
        ] })
      ] })
    ] }) }) }),
    !isNavbarOpen && /* @__PURE__ */ jsx(
      "button",
      {
        onClick: () => setIsNavbarOpen(true),
        className: `fixed left-4 top-4 z-50 p-2 rounded-full ${isDarkMode ? "bg-gray-700 hover:bg-gray-600" : "bg-white hover:bg-gray-100"} shadow-lg transition-all`,
        children: /* @__PURE__ */ jsx(ChevronRight, { size: 20 })
      }
    ),
    /* @__PURE__ */ jsxs("div", { className: `relative transition-all duration-300 ${isNavbarOpen ? "ml-64" : "ml-0"}`, children: [
      /* @__PURE__ */ jsxs("div", { className: `relative h-screen overflow-hidden ${isDarkMode ? "bg-gray-800" : "bg-white"}`, style: { perspective: "1000px" }, children: [
        /* @__PURE__ */ jsx("div", { className: "relative h-full", style: { transformStyle: "preserve-3d" }, children: slides.map((slide, index) => /* @__PURE__ */ jsx(
          SlideCard,
          {
            slide,
            index,
            currentSlide,
            isDarkMode,
            onModalOpen: openModal,
            isTransitioning
          },
          slide.id
        )) }),
        /* @__PURE__ */ jsx(
          "button",
          {
            onClick: () => handleSlideTransition("prev"),
            disabled: isTransitioning,
            className: "absolute left-4 top-1/2 transform -translate-y-1/2 p-2 bg-white bg-opacity-20 hover:bg-opacity-30 text-white rounded-full transition-all disabled:opacity-50",
            children: /* @__PURE__ */ jsx(ChevronLeft, { size: 24 })
          }
        ),
        /* @__PURE__ */ jsx(
          "button",
          {
            onClick: () => handleSlideTransition("next"),
            disabled: isTransitioning,
            className: "absolute right-4 top-1/2 transform -translate-y-1/2 p-2 bg-white bg-opacity-20 hover:bg-opacity-30 text-white rounded-full transition-all disabled:opacity-50",
            children: /* @__PURE__ */ jsx(ChevronRight, { size: 24 })
          }
        ),
        /* @__PURE__ */ jsx("div", { className: "absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-2", children: slides.map((_, index) => /* @__PURE__ */ jsx(
          "button",
          {
            onClick: () => handleSlideTransition(index),
            disabled: isTransitioning,
            className: `w-3 h-3 rounded-full transition-all disabled:opacity-50 ${currentSlide === index ? "bg-white" : "bg-white bg-opacity-50"}`
          },
          index
        )) })
      ] }),
      /* @__PURE__ */ jsx("div", { className: "absolute bottom-20 left-1/2 transform -translate-x-1/2", children: /* @__PURE__ */ jsxs(
        "button",
        {
          onClick: handleDownload,
          className: `flex items-center space-x-2 px-6 py-3 rounded-lg font-semibold transition-all duration-300 hover:scale-105 ${isDarkMode ? "bg-green-600 hover:bg-green-700 text-white" : "bg-green-500 hover:bg-green-600 text-white"}`,
          children: [
            /* @__PURE__ */ jsx(Download, { size: 20 }),
            /* @__PURE__ */ jsx("span", { children: "資料をダウンロード (Shift+Enter)" })
          ]
        }
      ) })
    ] }),
    /* @__PURE__ */ jsx("div", { className: `py-20 px-8 transition-all duration-300 ${isNavbarOpen ? "ml-64" : "ml-0"} ${isDarkMode ? "bg-gray-900" : "bg-gray-50"}`, children: /* @__PURE__ */ jsxs("div", { className: "max-w-4xl mx-auto", children: [
      /* @__PURE__ */ jsx("h2", { className: "text-3xl font-bold text-center mb-12", children: "自己紹介" }),
      /* @__PURE__ */ jsx("div", { className: "grid md:grid-cols-3 gap-8", children: profileSections.map((section, index) => /* @__PURE__ */ jsx(
        ProfileCard,
        {
          section,
          index,
          currentProfileSection,
          isDarkMode
        },
        section.title
      )) }),
      /* @__PURE__ */ jsx("p", { className: "text-center mt-8 text-sm opacity-70", children: "Shift+Spaceで大項目を移動できます" })
    ] }) }),
    isModalOpen && modalContent && /* @__PURE__ */ jsx("div", { className: "fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4", children: /* @__PURE__ */ jsxs("div", { className: `max-w-4xl w-full max-h-[90vh] overflow-hidden rounded-lg ${isDarkMode ? "bg-gray-800" : "bg-white"}`, children: [
      /* @__PURE__ */ jsxs("div", { className: `p-6 border-b ${isDarkMode ? "border-gray-700" : "border-gray-200"}`, children: [
        /* @__PURE__ */ jsxs("div", { className: "flex justify-between items-center", children: [
          /* @__PURE__ */ jsx("h3", { className: "text-2xl font-bold", children: modalContent.title }),
          /* @__PURE__ */ jsx(
            "button",
            {
              onClick: closeModal,
              className: `p-2 rounded-full hover:scale-110 transition-transform ${isDarkMode ? "hover:bg-gray-700" : "hover:bg-gray-100"}`,
              children: /* @__PURE__ */ jsx(X, { size: 24 })
            }
          )
        ] }),
        /* @__PURE__ */ jsx(
          SearchBar$1,
          {
            isSearchOpen,
            searchTerm,
            onSearchChange: (e) => setSearchTerm(e.target.value),
            isDarkMode
          }
        )
      ] }),
      /* @__PURE__ */ jsxs("div", { className: "p-6 overflow-y-auto max-h-[60vh]", children: [
        (searchTerm ? filteredContent : Object.entries(modalContent.details || {})).map(([key, value]) => /* @__PURE__ */ jsxs("div", { className: "mb-6", children: [
          /* @__PURE__ */ jsx("h4", { className: "text-lg font-semibold mb-2 capitalize text-blue-600", children: detailLabels[key] || key }),
          /* @__PURE__ */ jsx("p", { className: `leading-relaxed ${isDarkMode ? "text-gray-300" : "text-gray-700"}`, style: {
            animation: "glow 2s ease-in-out infinite alternate",
            textShadow: isDarkMode ? "0 0 10px rgba(59, 130, 246, 0.5)" : "none"
          }, children: value })
        ] }, key)),
        searchTerm && filteredContent.length === 0 && /* @__PURE__ */ jsx("p", { className: "text-center text-gray-500 py-8", children: "検索結果が見つかりません" })
      ] }),
      /* @__PURE__ */ jsx("div", { className: `p-4 border-t text-sm text-center opacity-70 ${isDarkMode ? "border-gray-700" : "border-gray-200"}`, children: "Spaceで検索 | Shift+Spaceで閉じる" })
    ] }) })
  ] });
};
function meta$2({}) {
  return [{
    title: "New React Router App"
  }, {
    name: "description",
    content: "Welcome to React Router!"
  }];
}
const intro = UNSAFE_withComponentProps(function Intro() {
  return /* @__PURE__ */ jsx(PresentationSlider, {});
});
const route4 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: intro,
  meta: meta$2
}, Symbol.toStringTag, { value: "Module" }));
const SearchBar = () => {
  const [query, setQuery] = useState("");
  const [selectedModel, setSelectedModel] = useState("gpt-4");
  const [selectedVideoEngine, setSelectedVideoEngine] = useState("youtube");
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [currentEngine, setCurrentEngine] = useState("ai");
  const [isTransitioning, setIsTransitioning] = useState(false);
  const containerRef = useRef(null);
  const lastWheelTime = useRef(0);
  const wheelDeltaX = useRef(0);
  const aiModels = [
    { id: "gpt-4", name: "GPT-4", icon: Brain, color: "text-purple-400" },
    { id: "claude", name: "Claude", icon: Bot, color: "text-blue-400" },
    { id: "gemini", name: "Gemini", icon: Sparkles, color: "text-green-400" },
    { id: "gpt-3.5", name: "GPT-3.5", icon: Zap, color: "text-yellow-400" }
  ];
  const videoEngines = [
    { id: "youtube", name: "YouTube", icon: Youtube, color: "text-red-400" },
    { id: "vimeo", name: "Vimeo", icon: Play, color: "text-blue-400" },
    { id: "dailymotion", name: "Dailymotion", icon: Film, color: "text-orange-400" },
    { id: "twitch", name: "Twitch", icon: Tv, color: "text-purple-400" },
    { id: "tiktok", name: "TikTok", icon: Video, color: "text-pink-400" }
  ];
  const handleSubmit = () => {
    if (query.trim()) {
      if (currentEngine === "ai") {
        console.log("AI検索クエリ:", query);
        console.log("選択モデル:", selectedModel);
      } else {
        console.log("動画検索クエリ:", query);
        console.log("選択動画エンジン:", selectedVideoEngine);
      }
      console.log("添付ファイル:", attachedFiles);
    }
  };
  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };
  const handleFileAttach = (e) => {
    const files = Array.from(e.target.files ?? []);
    setAttachedFiles((prev) => [...prev, ...files]);
  };
  const removeFile = (index) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
  };
  const switchEngine = () => {
    if (isTransitioning) return;
    setIsTransitioning(true);
    setTimeout(() => {
      setCurrentEngine((prev) => prev === "ai" ? "video" : "ai");
      setQuery("");
    }, 500);
    setTimeout(() => {
      setIsTransitioning(false);
    }, 1e3);
  };
  const handleWheel = (e) => {
    if (isTransitioning) return;
    const now = Date.now();
    const timeDelta = now - lastWheelTime.current;
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      e.preventDefault();
      wheelDeltaX.current += e.deltaX;
      const threshold = 150;
      if (Math.abs(wheelDeltaX.current) > threshold) {
        switchEngine();
        wheelDeltaX.current = 0;
        lastWheelTime.current = now;
        return;
      }
    }
    if (timeDelta > 200) {
      wheelDeltaX.current = e.deltaX || 0;
    }
    lastWheelTime.current = now;
    setTimeout(() => {
      if (Date.now() - lastWheelTime.current > 250) {
        wheelDeltaX.current *= 0.5;
        if (Math.abs(wheelDeltaX.current) < 10) {
          wheelDeltaX.current = 0;
        }
      }
    }, 300);
  };
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      container.removeEventListener("wheel", handleWheel);
    };
  }, [isTransitioning]);
  const selectedModelData = aiModels.find((model) => model.id === selectedModel);
  const selectedVideoData = videoEngines.find((engine) => engine.id === selectedVideoEngine);
  const ModelIcon = (selectedModelData == null ? void 0 : selectedModelData.icon) || Bot;
  const VideoIcon = (selectedVideoData == null ? void 0 : selectedVideoData.icon) || Youtube;
  const getEngineStyles = () => {
    if (currentEngine === "ai") {
      return {
        gradient: "from-blue-600 to-purple-600",
        ring: "ring-blue-500",
        button: "bg-blue-600 hover:bg-blue-700",
        title: "AI検索アシスタント",
        subtitle: "お好みのAIモデルで検索してください",
        placeholder: "何でも聞いてください..."
      };
    } else {
      return {
        gradient: "from-red-600 to-pink-600",
        ring: "ring-red-500",
        button: "bg-red-600 hover:bg-red-700",
        title: "動画検索エンジン",
        subtitle: "お好みのプラットフォームで動画を検索",
        placeholder: "動画を検索してください..."
      };
    }
  };
  const styles = getEngineStyles();
  return /* @__PURE__ */ jsx("div", { className: `min-h-screen transition-colors duration-300 ${isDarkMode ? "bg-gray-900" : "bg-gray-50"}`, children: /* @__PURE__ */ jsxs("div", { className: "container mx-auto px-4 py-8", children: [
    /* @__PURE__ */ jsx("div", { className: "flex justify-end mb-6", children: /* @__PURE__ */ jsx(
      "button",
      {
        onClick: () => setIsDarkMode(!isDarkMode),
        className: `px-4 py-2 rounded-lg transition-colors ${isDarkMode ? "bg-gray-800 text-gray-200 hover:bg-gray-700" : "bg-white text-gray-700 hover:bg-gray-100 border"}`,
        children: isDarkMode ? "🌙 ダーク" : "☀️ ライト"
      }
    ) }),
    /* @__PURE__ */ jsxs(
      "div",
      {
        ref: containerRef,
        className: `transition-all duration-500 ease-in-out ${isTransitioning ? "opacity-0 transform translate-y-4 scale-95" : "opacity-100 transform translate-y-0 scale-100"}`,
        style: {
          userSelect: "none"
        },
        children: [
          /* @__PURE__ */ jsxs("div", { className: "text-center mb-8", children: [
            /* @__PURE__ */ jsx("h1", { className: `text-3xl font-bold mb-2 bg-gradient-to-r ${styles.gradient} bg-clip-text text-transparent`, children: styles.title }),
            /* @__PURE__ */ jsx("p", { className: `text-lg ${isDarkMode ? "text-gray-400" : "text-gray-600"}`, children: styles.subtitle }),
            /* @__PURE__ */ jsx("div", { className: "mt-4 text-sm text-gray-500", children: "← マウスパッドでスワイプして切り替え →" })
          ] }),
          /* @__PURE__ */ jsx("div", { className: "max-w-4xl mx-auto", children: /* @__PURE__ */ jsxs("div", { className: "space-y-4", children: [
            /* @__PURE__ */ jsx("div", { className: "flex flex-wrap gap-3 justify-center", children: (currentEngine === "ai" ? aiModels : videoEngines).map((item) => {
              const Icon = item.icon;
              const isSelected = currentEngine === "ai" ? selectedModel === item.id : selectedVideoEngine === item.id;
              return /* @__PURE__ */ jsxs(
                "button",
                {
                  type: "button",
                  onClick: () => {
                    if (currentEngine === "ai") {
                      setSelectedModel(item.id);
                    } else {
                      setSelectedVideoEngine(item.id);
                    }
                  },
                  className: `flex items-center gap-2 px-4 py-2 rounded-full transition-all duration-200 ${isSelected ? isDarkMode ? `bg-gray-700 text-white ring-2 ${styles.ring}` : `bg-blue-100 text-blue-700 ring-2 ring-blue-300` : isDarkMode ? "bg-gray-800 text-gray-300 hover:bg-gray-700" : "bg-white text-gray-600 hover:bg-gray-50 border"}`,
                  children: [
                    /* @__PURE__ */ jsx(Icon, { size: 18, className: isSelected ? item.color : "" }),
                    /* @__PURE__ */ jsx("span", { className: "font-medium", children: item.name })
                  ]
                },
                item.id
              );
            }) }),
            /* @__PURE__ */ jsxs("div", { className: `relative rounded-2xl shadow-lg transition-all duration-300 ${isDarkMode ? "bg-gray-800 border border-gray-700" : "bg-white border-2 border-gray-200"} focus-within:ring-4 ${isDarkMode ? `focus-within:${styles.ring}/20` : "focus-within:ring-blue-200"}`, children: [
              /* @__PURE__ */ jsxs("div", { className: "flex items-center p-4", children: [
                /* @__PURE__ */ jsx("div", { className: "flex-shrink-0 mr-3", children: /* @__PURE__ */ jsx(Search, { className: `w-5 h-5 ${isDarkMode ? "text-gray-400" : "text-gray-500"}` }) }),
                /* @__PURE__ */ jsx("div", { className: `flex items-center gap-2 px-3 py-1 rounded-full mr-3 ${isDarkMode ? "bg-gray-700" : "bg-gray-100"}`, children: currentEngine === "ai" ? /* @__PURE__ */ jsxs(Fragment, { children: [
                  /* @__PURE__ */ jsx(ModelIcon, { size: 16, className: selectedModelData == null ? void 0 : selectedModelData.color }),
                  /* @__PURE__ */ jsx("span", { className: `text-sm font-medium ${isDarkMode ? "text-gray-300" : "text-gray-600"}`, children: selectedModelData == null ? void 0 : selectedModelData.name })
                ] }) : /* @__PURE__ */ jsxs(Fragment, { children: [
                  /* @__PURE__ */ jsx(VideoIcon, { size: 16, className: selectedVideoData == null ? void 0 : selectedVideoData.color }),
                  /* @__PURE__ */ jsx("span", { className: `text-sm font-medium ${isDarkMode ? "text-gray-300" : "text-gray-600"}`, children: selectedVideoData == null ? void 0 : selectedVideoData.name })
                ] }) }),
                /* @__PURE__ */ jsx(
                  "textarea",
                  {
                    value: query,
                    onChange: (e) => setQuery(e.target.value),
                    onKeyPress: handleKeyPress,
                    placeholder: styles.placeholder,
                    rows: 1,
                    className: `flex-1 resize-none outline-none text-lg ${isDarkMode ? "bg-transparent text-white placeholder-gray-400" : "bg-transparent text-gray-900 placeholder-gray-500"}`,
                    style: {
                      minHeight: "28px",
                      maxHeight: "120px"
                    }
                  }
                ),
                /* @__PURE__ */ jsxs("div", { className: "flex items-center gap-2 ml-3", children: [
                  /* @__PURE__ */ jsxs("label", { className: `p-2 rounded-lg cursor-pointer transition-colors ${isDarkMode ? "hover:bg-gray-700 text-gray-400 hover:text-gray-300" : "hover:bg-gray-100 text-gray-500 hover:text-gray-600"}`, children: [
                    /* @__PURE__ */ jsx(Paperclip, { size: 20 }),
                    /* @__PURE__ */ jsx(
                      "input",
                      {
                        type: "file",
                        multiple: true,
                        onChange: handleFileAttach,
                        className: "hidden"
                      }
                    )
                  ] }),
                  /* @__PURE__ */ jsx(
                    "button",
                    {
                      type: "submit",
                      disabled: !query.trim(),
                      onClick: handleSubmit,
                      className: `p-2 rounded-lg transition-all duration-200 ${query.trim() ? `${styles.button} text-white shadow-md hover:shadow-lg` : isDarkMode ? "bg-gray-700 text-gray-500 cursor-not-allowed" : "bg-gray-200 text-gray-400 cursor-not-allowed"}`,
                      children: /* @__PURE__ */ jsx(Send, { size: 20 })
                    }
                  )
                ] })
              ] }),
              attachedFiles.length > 0 && /* @__PURE__ */ jsx("div", { className: `px-4 pb-4 border-t ${isDarkMode ? "border-gray-700" : "border-gray-200"}`, children: /* @__PURE__ */ jsx("div", { className: "flex flex-wrap gap-2 mt-3", children: attachedFiles.map((file, index) => /* @__PURE__ */ jsxs(
                "div",
                {
                  className: `flex items-center gap-2 px-3 py-1 rounded-full text-sm ${isDarkMode ? "bg-gray-700 text-gray-300" : "bg-gray-100 text-gray-600"}`,
                  children: [
                    /* @__PURE__ */ jsx(Paperclip, { size: 14 }),
                    /* @__PURE__ */ jsx("span", { className: "truncate max-w-32", children: file.name }),
                    /* @__PURE__ */ jsx(
                      "button",
                      {
                        type: "button",
                        onClick: () => removeFile(index),
                        className: `ml-1 rounded-full w-4 h-4 flex items-center justify-center text-xs ${isDarkMode ? "hover:bg-gray-600 text-gray-400" : "hover:bg-gray-200 text-gray-500"}`,
                        children: "×"
                      }
                    )
                  ]
                },
                index
              )) }) })
            ] }),
            /* @__PURE__ */ jsx("div", { className: "text-center", children: /* @__PURE__ */ jsx("p", { className: `text-sm ${isDarkMode ? "text-gray-500" : "text-gray-400"}`, children: "Enter で送信 • Shift + Enter で改行 • ファイルをドラッグ&ドロップも可能" }) })
          ] }) })
        ]
      }
    )
  ] }) });
};
function meta$1({}) {
  return [{
    title: "New React Router App"
  }, {
    name: "description",
    content: "Welcome to React Router!"
  }];
}
const chat = UNSAFE_withComponentProps(function Home4() {
  return /* @__PURE__ */ jsx(SearchBar, {});
});
const route5 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: chat,
  meta: meta$1
}, Symbol.toStringTag, { value: "Module" }));
function Mains() {
  const [visibleChars, setVisibleChars] = useState(0);
  const [phase, setPhase] = useState("appearing");
  const [visibleButtons, setVisibleButtons] = useState(0);
  const text = "Happy Welcome";
  const questions = [
    { id: 1, text: "Annual income?", icon: "🌤️", href: "/intro" },
    { id: 2, text: "Skill up?", icon: "🎬", href: "/chat" },
    { id: 3, text: "Business Consulting?", icon: "🍳", href: "/manual" },
    { id: 4, text: "Other contact information", icon: "✈️", href: "/camera" }
  ];
  useEffect(() => {
    if (phase === "appearing") {
      const timer = setInterval(() => {
        setVisibleChars((prev) => {
          if (prev < text.length) {
            return prev + 1;
          }
          clearInterval(timer);
          setTimeout(() => setPhase("disappearing"), 1500);
          return prev;
        });
      }, 150);
      return () => clearInterval(timer);
    }
    if (phase === "disappearing") {
      const timer = setInterval(() => {
        setVisibleChars((prev) => {
          if (prev > 0) {
            return prev - 1;
          }
          clearInterval(timer);
          setTimeout(() => setPhase("buttons"), 300);
          return prev;
        });
      }, 100);
      return () => clearInterval(timer);
    }
    if (phase === "buttons") {
      const timer = setInterval(() => {
        setVisibleButtons((prev) => {
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
  const handleQuestionClick = (question) => {
    console.log("質問が選択されました:", question);
  };
  const navigate = useNavigate();
  return /* @__PURE__ */ jsx("div", { className: "flex items-center justify-center min-h-screen bg-white dark:bg-gray-900 transition-colors duration-300", children: phase !== "buttons" ? (
    // ウェルカムテキスト表示フェーズ
    /* @__PURE__ */ jsx("div", { className: "text-center", children: /* @__PURE__ */ jsx(
      "div",
      {
        className: "text-6xl font-bold transform -rotate-1 hover:rotate-0 transition-transform duration-300",
        style: { fontFamily: "cursive" },
        children: text.split("").map((char, index) => /* @__PURE__ */ jsx(
          "span",
          {
            className: `inline-block transition-all duration-700 ease-out text-gray-800 dark:text-white ${index < visibleChars ? "opacity-100 transform translate-y-0" : "opacity-0 transform translate-y-8"}`,
            style: {
              transitionDelay: `${index * 50}ms`,
              textShadow: "2px 2px 4px rgba(0, 0, 0, 0.1), 0 0 10px rgba(255, 255, 255, 0.1)"
            },
            children: char === " " ? " " : char
          },
          index
        ))
      }
    ) })
  ) : (
    // ボタン表示フェーズ
    /* @__PURE__ */ jsx("div", { className: "w-full max-w-4xl px-8", children: /* @__PURE__ */ jsx("div", { className: "grid grid-cols-2 gap-8", children: questions.map((question, index) => /* @__PURE__ */ jsxs(
      "button",
      {
        onClick: () => {
          handleQuestionClick(question.text);
          navigate(question.href);
        },
        className: `
                  group relative p-8 rounded-2xl border-2 border-gray-200 dark:border-gray-700
                  bg-white dark:bg-gray-800 shadow-lg hover:shadow-xl
                  transform transition-all duration-700 ease-out hover:scale-105
                  ${index < visibleButtons ? "opacity-100 translate-y-0 rotate-0" : "opacity-0 translate-y-12 rotate-3"}
                `,
        style: {
          transitionDelay: `${index * 150}ms`
        },
        children: [
          /* @__PURE__ */ jsxs("div", { className: "text-center space-y-3", children: [
            /* @__PURE__ */ jsx("div", { className: "text-4xl group-hover:scale-110 transition-transform duration-300", children: question.icon }),
            /* @__PURE__ */ jsx("div", { className: "text-lg font-semibold text-gray-800 dark:text-white", children: question.text })
          ] }),
          /* @__PURE__ */ jsx("div", { className: "absolute inset-0 rounded-2xl bg-gradient-to-r from-blue-500/10 to-purple-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" })
        ]
      },
      question.id
    )) }) })
  ) });
}
function meta({}) {
  return [{
    title: "New React Router App"
  }, {
    name: "description",
    content: "Welcome to React Router!"
  }];
}
const page = UNSAFE_withComponentProps(function Home5() {
  return /* @__PURE__ */ jsx(Mains, {});
});
const route6 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: page,
  meta
}, Symbol.toStringTag, { value: "Module" }));
const serverManifest = { "entry": { "module": "/assets/entry.client-B6r0kRvB.js", "imports": ["/assets/chunk-ZYFC6VSF-up4aUAKg.js"], "css": [] }, "routes": { "root": { "id": "root", "parentId": void 0, "path": "", "index": void 0, "caseSensitive": void 0, "hasAction": false, "hasLoader": false, "hasClientAction": false, "hasClientLoader": false, "hasClientMiddleware": false, "hasErrorBoundary": true, "module": "/assets/root-DKVD1_jx.js", "imports": ["/assets/chunk-ZYFC6VSF-up4aUAKg.js"], "css": ["/assets/root-BZP5sSad.css"], "clientActionModule": void 0, "clientLoaderModule": void 0, "clientMiddlewareModule": void 0, "hydrateFallbackModule": void 0 }, "routes/home": { "id": "routes/home", "parentId": "root", "path": void 0, "index": true, "caseSensitive": void 0, "hasAction": false, "hasLoader": false, "hasClientAction": false, "hasClientLoader": false, "hasClientMiddleware": false, "hasErrorBoundary": false, "module": "/assets/home-DPpkYF3A.js", "imports": ["/assets/chunk-ZYFC6VSF-up4aUAKg.js"], "css": [], "clientActionModule": void 0, "clientLoaderModule": void 0, "clientMiddlewareModule": void 0, "hydrateFallbackModule": void 0 }, "routes/camera": { "id": "routes/camera", "parentId": "root", "path": "camera", "index": void 0, "caseSensitive": void 0, "hasAction": false, "hasLoader": false, "hasClientAction": false, "hasClientLoader": false, "hasClientMiddleware": false, "hasErrorBoundary": false, "module": "/assets/camera-mLoGhbL1.js", "imports": ["/assets/chunk-ZYFC6VSF-up4aUAKg.js", "/assets/createLucideIcon-CxbA2obt.js", "/assets/download-CXfEqW-z.js", "/assets/play-DRvFFLT-.js"], "css": [], "clientActionModule": void 0, "clientLoaderModule": void 0, "clientMiddlewareModule": void 0, "hydrateFallbackModule": void 0 }, "routes/manual": { "id": "routes/manual", "parentId": "root", "path": "manual", "index": void 0, "caseSensitive": void 0, "hasAction": false, "hasLoader": false, "hasClientAction": false, "hasClientLoader": false, "hasClientMiddleware": false, "hasErrorBoundary": false, "module": "/assets/manual-BebqYOY0.js", "imports": ["/assets/chunk-ZYFC6VSF-up4aUAKg.js", "/assets/search-CyeCip0u.js", "/assets/x-BJxeLrof.js", "/assets/createLucideIcon-CxbA2obt.js"], "css": [], "clientActionModule": void 0, "clientLoaderModule": void 0, "clientMiddlewareModule": void 0, "hydrateFallbackModule": void 0 }, "routes/intro": { "id": "routes/intro", "parentId": "root", "path": "intro", "index": void 0, "caseSensitive": void 0, "hasAction": false, "hasLoader": false, "hasClientAction": false, "hasClientLoader": false, "hasClientMiddleware": false, "hasErrorBoundary": false, "module": "/assets/intro-yTJSWrtv.js", "imports": ["/assets/chunk-ZYFC6VSF-up4aUAKg.js", "/assets/createLucideIcon-CxbA2obt.js", "/assets/download-CXfEqW-z.js", "/assets/x-BJxeLrof.js", "/assets/search-CyeCip0u.js"], "css": [], "clientActionModule": void 0, "clientLoaderModule": void 0, "clientMiddlewareModule": void 0, "hydrateFallbackModule": void 0 }, "routes/chat": { "id": "routes/chat", "parentId": "root", "path": "chat", "index": void 0, "caseSensitive": void 0, "hasAction": false, "hasLoader": false, "hasClientAction": false, "hasClientLoader": false, "hasClientMiddleware": false, "hasErrorBoundary": false, "module": "/assets/chat-D3YwZJ5X.js", "imports": ["/assets/chunk-ZYFC6VSF-up4aUAKg.js", "/assets/createLucideIcon-CxbA2obt.js", "/assets/play-DRvFFLT-.js", "/assets/search-CyeCip0u.js"], "css": [], "clientActionModule": void 0, "clientLoaderModule": void 0, "clientMiddlewareModule": void 0, "hydrateFallbackModule": void 0 }, "routes/page": { "id": "routes/page", "parentId": "root", "path": "page", "index": void 0, "caseSensitive": void 0, "hasAction": false, "hasLoader": false, "hasClientAction": false, "hasClientLoader": false, "hasClientMiddleware": false, "hasErrorBoundary": false, "module": "/assets/page-BSGa762w.js", "imports": ["/assets/chunk-ZYFC6VSF-up4aUAKg.js"], "css": [], "clientActionModule": void 0, "clientLoaderModule": void 0, "clientMiddlewareModule": void 0, "hydrateFallbackModule": void 0 } }, "url": "/assets/manifest-7dfec1bc.js", "version": "7dfec1bc", "sri": void 0 };
const assetsBuildDirectory = "build/client";
const basename = "/";
const future = { "unstable_middleware": false, "unstable_optimizeDeps": false, "unstable_splitRouteModules": false, "unstable_subResourceIntegrity": false, "unstable_viteEnvironmentApi": false };
const ssr = true;
const isSpaMode = false;
const prerender = [];
const routeDiscovery = { "mode": "lazy", "manifestPath": "/__manifest" };
const publicPath = "/";
const entry = { module: entryServer };
const routes = {
  "root": {
    id: "root",
    parentId: void 0,
    path: "",
    index: void 0,
    caseSensitive: void 0,
    module: route0
  },
  "routes/home": {
    id: "routes/home",
    parentId: "root",
    path: void 0,
    index: true,
    caseSensitive: void 0,
    module: route1
  },
  "routes/camera": {
    id: "routes/camera",
    parentId: "root",
    path: "camera",
    index: void 0,
    caseSensitive: void 0,
    module: route2
  },
  "routes/manual": {
    id: "routes/manual",
    parentId: "root",
    path: "manual",
    index: void 0,
    caseSensitive: void 0,
    module: route3
  },
  "routes/intro": {
    id: "routes/intro",
    parentId: "root",
    path: "intro",
    index: void 0,
    caseSensitive: void 0,
    module: route4
  },
  "routes/chat": {
    id: "routes/chat",
    parentId: "root",
    path: "chat",
    index: void 0,
    caseSensitive: void 0,
    module: route5
  },
  "routes/page": {
    id: "routes/page",
    parentId: "root",
    path: "page",
    index: void 0,
    caseSensitive: void 0,
    module: route6
  }
};
export {
  serverManifest as assets,
  assetsBuildDirectory,
  basename,
  entry,
  future,
  isSpaMode,
  prerender,
  publicPath,
  routeDiscovery,
  routes,
  ssr
};
