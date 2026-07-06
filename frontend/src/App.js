import React, { useState, useEffect, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
import MessageInput from "./components/MessageInput";
import Dashboard from "./components/Dashboard";
import { fetchModels, sendQuery, sendChat, sendEvaluate } from "./api";

function Toast({ toast }) {
  if (!toast) return null;
  return <div className={`status-toast ${toast.type}`}>{toast.message}</div>;
}

export default function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("gpt-oss-120b");
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeAgent, setActiveAgent] = useState("");
  const [activeSubAgent, setActiveSubAgent] = useState("none");
  const [toast, setToast] = useState(null);
  const [view, setView] = useState("chat"); // 'chat' or 'dashboard'
  const [evaluations, setEvaluations] = useState([]);

  const showToast = useCallback((type, message) => {
    setToast({ type, message });
    setTimeout(() => setToast(null), 3500);
  }, []);

  // Fetch available models on mount
  useEffect(() => {
    fetchModels()
      .then((data) => {
        setModels(data.models || []);
        if (data.models && data.models.length > 0) {
          setSelectedModel(data.models[0].key);
        }
      })
      .catch(() => {
        // Fallback models if backend is not ready
        setModels([
          { key: "gpt-oss-120b", display_name: "GPT-OSS 120B" },
          { key: "gpt-oss-20b", display_name: "GPT-OSS 20B" },
        ]);
      });
  }, []);

  const handleSend = useCallback(
    async (text) => {
      // Add user message
      const userMsg = { role: "user", text };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);
      setActiveAgent("");
      setActiveSubAgent("none");

      try {
        let result;
        const hasFiles = uploadedFiles.length > 0;

        if (hasFiles) {
          // Use the RAG pipeline
          result = await sendQuery(text, selectedModel);
          
          // Trigger evaluation asynchronously (in background)
          const contexts = result.contexts || [];
          sendEvaluate(text, result.response, contexts)
            .then(metrics => {
               setEvaluations(prev => [...prev, { query: text, ...metrics }]);
               showToast("success", "Evaluation complete");
            })
            .catch(e => console.error("Eval error", e));
            
        } else {
          // No files uploaded — use chat agent
          result = await sendChat(text, selectedModel);
        }

        const assistantMsg = {
          role: "assistant",
          text: result.response || "No response generated.",
          agent: result.agent || "unknown",
          sub_agent: result.sub_agent || "none",
          sources: result.sources || [],
        };

        setMessages((prev) => [...prev, assistantMsg]);
        setActiveAgent(result.agent || "");
        setActiveSubAgent(result.sub_agent || "none");
      } catch (err) {
        const errMsg = {
          role: "assistant",
          text: `⚠️ **Error:** ${err.message}`,
          agent: "error",
          sub_agent: "none",
          sources: [],
        };
        setMessages((prev) => [...prev, errMsg]);
        showToast("error", err.message);
      } finally {
        setIsLoading(false);
      }
    },
    [selectedModel, uploadedFiles, showToast]
  );

  return (
    <div className="app-container">
      <Toast toast={toast} />

      <Sidebar
        models={models}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
        uploadedFiles={uploadedFiles}
        setUploadedFiles={setUploadedFiles}
        activeAgent={activeAgent}
        activeSubAgent={activeSubAgent}
        onToast={showToast}
      />

      <div className="main-content">
        <div className="chat-header">
          <h2>{view === "chat" ? "💬 Study Chat" : "📊 Evaluation Dashboard"}</h2>
          <div className="agent-badge-group">
            <button 
              onClick={() => setView("chat")}
              style={{ padding: '6px 12px', marginRight: '8px', background: view === "chat" ? '#3b82f6' : 'transparent', border: '1px solid #3b82f6', color: view === "chat" ? '#fff' : '#3b82f6', borderRadius: '4px', cursor: 'pointer', fontSize: '14px', fontWeight: 'bold' }}
            >
              Chat
            </button>
            <button 
              onClick={() => setView("dashboard")}
              style={{ padding: '6px 12px', background: view === "dashboard" ? '#3b82f6' : 'transparent', border: '1px solid #3b82f6', color: view === "dashboard" ? '#fff' : '#3b82f6', borderRadius: '4px', cursor: 'pointer', fontSize: '14px', fontWeight: 'bold' }}
            >
              Dashboard
            </button>
            
            {view === "chat" && activeAgent && activeAgent !== "error" && (
              <span className={`agent-badge ${activeAgent}`}>
                🤖 {activeAgent.replace(/_/g, " ")}
              </span>
            )}
            {activeSubAgent && activeSubAgent !== "none" && (
              <span className={`agent-badge ${activeSubAgent}`}>
                ⚡ Sub: {activeSubAgent.replace(/_/g, " ")}
              </span>
            )}
            {!activeAgent && (
              <span style={{ fontSize: "0.78rem", color: "#64748b" }}>
                {uploadedFiles.length > 0
                  ? "Ready — ask a question"
                  : "Upload PDFs or chat freely"}
              </span>
            )}
          </div>
        </div>

        {view === "chat" ? (
          <>
            <ChatArea messages={messages} isLoading={isLoading} />
            <MessageInput onSend={handleSend} disabled={isLoading} />
          </>
        ) : (
          <Dashboard evaluations={evaluations} />
        )}
      </div>
    </div>
  );
}
