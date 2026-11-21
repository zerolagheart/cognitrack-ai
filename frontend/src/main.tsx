import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Explainability } from "./pages/Explainability";
import { Longitudinal } from "./pages/Longitudinal";
import { ModelIntelligence } from "./pages/ModelIntelligence";
import { Monitoring } from "./pages/Monitoring";
import { Overview } from "./pages/Overview";
import { ParticipantDetail, Participants } from "./pages/Participants";
import { ResponsibleAI } from "./pages/ResponsibleAI";
import { SpeechAnalysis } from "./pages/SpeechAnalysis";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/participants" element={<Participants />} />
          <Route path="/participants/:id" element={<ParticipantDetail />} />
          <Route path="/speech" element={<SpeechAnalysis />} />
          <Route path="/speech/:key" element={<SpeechAnalysis />} />
          <Route path="/models" element={<ModelIntelligence />} />
          <Route path="/longitudinal" element={<Longitudinal />} />
          <Route path="/longitudinal/:id" element={<Longitudinal />} />
          <Route path="/explainability" element={<Explainability />} />
          <Route path="/explainability/:key" element={<Explainability />} />
          <Route path="/responsible-ai" element={<ResponsibleAI />} />
          <Route path="/monitoring" element={<Monitoring />} />
          <Route path="*" element={<div className="py-20 text-center text-ink-muted">Page not found.</div>} />
        </Routes>
      </Layout>
    </BrowserRouter>
  </React.StrictMode>,
);
