import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { installGlobalCapture } from "@tenryu-common/core/applog";
import "./styles.css";
import ErrorBoundary from "./ui/ErrorBoundary";

installGlobalCapture();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
);
