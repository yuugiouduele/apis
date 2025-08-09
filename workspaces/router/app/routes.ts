import { type RouteConfig, index } from "@react-router/dev/routes";
import { flatRoutes } from "@react-router/fs-routes";

export default (async (): Promise<RouteConfig> => {
  const autoRoutes = await flatRoutes({
    rootDirectory: "routes",
    ignoredRouteFiles: ["home.tsx"],
  });

  return [
    index("routes/home.tsx"),
    ...autoRoutes.filter(r => r.file !== "routes/home.tsx"),
  ];
})();
