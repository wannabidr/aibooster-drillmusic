import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";

// eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- root element guaranteed by index.html
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
