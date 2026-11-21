/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#f4f7f8",
        ink: { DEFAULT: "#0f1f2a", soft: "#3d5361", muted: "#6b8290" },
        line: "#dde6ea",
        brand: { 50: "#ecfdf8", 100: "#d1f5ec", 500: "#14a08c", 600: "#0f766e", 700: "#0b5c57", 900: "#06302d" },
        signal: { info: "#2563eb", warn: "#b45309", warnbg: "#fef6e7", alert: "#be123c", alertbg: "#fdecef", ok: "#047857", okbg: "#e7f7f0" },
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "-apple-system", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
        mono: ["JetBrains Mono", "Cascadia Code", "Consolas", "monospace"],
      },
      boxShadow: { card: "0 1px 2px rgba(15,31,42,0.04), 0 1px 3px rgba(15,31,42,0.06)" },
    },
  },
  plugins: [],
};
