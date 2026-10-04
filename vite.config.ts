import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" is required so the built files load inside the Android WebView
export default defineConfig({
  base: "./",
  plugins: [react()],
});
