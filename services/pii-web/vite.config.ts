import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Dev only. In the container, nginx serves the build and proxies /api,
    // so the app is same-origin and the session cookie needs no CORS dance.
    proxy: { "/api": { target: "http://127.0.0.1:8080", changeOrigin: true } },
  },
  build: { outDir: "dist", sourcemap: false },
});
