import type { Route } from "./+types/home";
import { GeminiSearch } from "~/comp/gemini";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "New React Router App" },
    { name: "description", content: "Welcome to React Router!" },
  ];
}

export default function Home() {
  return<GeminiSearch/>;
}
