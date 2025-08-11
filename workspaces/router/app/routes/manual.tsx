import type { Route } from "./+types/home";
import { SelfManualComponent } from "../comp/recommend";
import { Router } from "react-router";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "New React Router App" },
    { name: "description", content: "Welcome to React Router!" },
  ];
}

export default function Home() {
  return < SelfManualComponent/>;
}
