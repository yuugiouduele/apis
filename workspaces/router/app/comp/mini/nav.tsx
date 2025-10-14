import React from "react";

const navLinks = [
  {
    href: "#home",
    label: "Agent",
    icon: (
      <svg
        className="w-6 h-6 mb-1"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        viewBox="0 0 24 24"
      >
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 14l9-5-9-5-9 5 9 5z" />
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M12 14l6.16-3.422A12.083 12.083 0 0118 19.882M12 14v7m0 0l-6.16-3.422A12.083 12.083 0 006 19.882"
        />
      </svg>
    ),
  },
  {
    href: "#about",
    label: "Doc",
    icon: (
      <svg
        className="w-6 h-6 mb-1"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        viewBox="0 0 24 24"
      >
        <path strokeLinecap="round" strokeLinejoin="round" d="M8 16h8M8 12h8M8 8h8M4 6h16v12H4z" />
      </svg>
    ),
  },
  {
    href: "#services",
    label: "Services",
    icon: (
      <svg
        className="w-6 h-6 mb-1"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        viewBox="0 0 24 24"
      >
        <circle cx="12" cy="12" r="3" strokeLinecap="round" strokeLinejoin="round" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M19.4 15a7.962 7.962 0 000-6M4.6 15a7.962 7.962 0 010-6" />
      </svg>
    ),
  },
  {
    href: "#contact",
    label: "Contact",
    icon: (
      <svg
        className="w-6 h-6 mb-1"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        viewBox="0 0 24 24"
      >
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 8l7.89 5.26a3 3 0 003.22 0L21 8m-9 12v-8" />
      </svg>
    ),
  },
  {
    href: "#setting",
    label: "Setting",
    icon: (
      <svg
        className="w-6 h-6 mb-1"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        viewBox="0 0 24 24"
      >
        <circle cx="12" cy="12" r="3" strokeLinecap="round" strokeLinejoin="round" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M19.4 15a7.962 7.962 0 000-6M4.6 15a7.962 7.962 0 010-6" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v2M12 19v2M3 12h2M19 12h2" />
      </svg>
    ),
  },
  {
    href: "#parmater",
    label: "Parameter",
    icon: (
      <svg
        className="w-6 h-6 mb-1"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        viewBox="0 0 24 24"
      >
        <rect width="18" height="12" x="3" y="6" strokeLinecap="round" strokeLinejoin="round" rx="2" ry="2" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 6v12M18 6v12" />
      </svg>
    ),
  },
];

export function LeftAlignedVerticalNavBar() {
  return (
    <nav className="fixed top-0 left-0 bg-gray-800 text-white w-36 min-h-full p-4 flex flex-col space-y-3 text-xs items-center">
      {navLinks.map(({ href, label, icon }) => (
        <a
          key={href}
          href={href}
          className="flex flex-col items-center hover:bg-gray-700 px-3 py-2 rounded text-center"
        >
          {icon}
          <span className="text-xs">{label}</span>
        </a>
      ))}
    </nav>
  );
}
