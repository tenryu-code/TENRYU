import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@tenryu-common": new URL("../gui-common/src", import.meta.url).pathname,
      "react/jsx-runtime": new URL(
        "./node_modules/react/jsx-runtime",
        import.meta.url,
      ).pathname,
      react: new URL("./node_modules/react", import.meta.url).pathname,
      h5wasm: new URL("./node_modules/h5wasm", import.meta.url).pathname,
      "@tauri-apps/api/core": new URL(
        "./node_modules/@tauri-apps/api/core",
        import.meta.url,
      ).pathname,
      "@tauri-apps/plugin-dialog": new URL(
        "./node_modules/@tauri-apps/plugin-dialog",
        import.meta.url,
      ).pathname,
      "@tauri-apps/plugin-fs": new URL(
        "./node_modules/@tauri-apps/plugin-fs",
        import.meta.url,
      ).pathname,
      "@tauri-apps/plugin-shell": new URL(
        "./node_modules/@tauri-apps/plugin-shell",
        import.meta.url,
      ).pathname,
      "@tauri-apps/plugin-store": new URL(
        "./node_modules/@tauri-apps/plugin-store",
        import.meta.url,
      ).pathname,
    },
  },
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:5175", changeOrigin: false },
    },
  },
  build: { target: "es2021", outDir: "dist" },
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
  },
});
