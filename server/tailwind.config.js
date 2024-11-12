/** @type {import("tailwindcss").Config} */
module.exports = {
  content: ["./data/**/*{html,js}"],
  plugins: [require("daisyui")],
  safelist: ["highlighted-note"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Jost"', "sans-serif"],
        title: ['"FingerPaint"', "sans-serif"],
      },
    },
  },
  daisyui: {
    themes: [
      {
        drummerscore: {
          ...require("daisyui/src/theming/themes")["dark"],
          "base-100": "#1A051E",
          "base-200": "#460B50",
          "base-300": "#571164",
          primary: "#FF7A00",
          "primary-content": "white",
          accent: "#773482",
          "accent-content": "white",
          "base-content": "white",
          neutral: "#34083b",
          "neutral-content": "#FF7A00",
        },
      },
    ],
  },
};
