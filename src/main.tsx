import React from "react";
import ReactDOM from "react-dom/client";
import App from "./app/App";
import "./styles/global.css";

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("#root element not found");
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/sw.js")
      .then((registration) => {
        registration.addEventListener("updatefound", () => {
          const installing = registration.installing;
          if (!installing || !navigator.serviceWorker.controller) return;
          installing.addEventListener("statechange", () => {
            if (installing.state === "installed") {
              window.dispatchEvent(new CustomEvent("app-update-available"));
            }
          });
        });
      })
      .catch(() => {
        // Offline support is a progressive enhancement; ignore registration failures.
      });
  });
}
