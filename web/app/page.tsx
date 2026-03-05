"use client";

import { useState } from "react";
import ChatInterface from "@/components/ChatInterface";
import Sidebar from "@/components/Sidebar";

export default function Home() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const handleQuickAction = (message: string) => {
    // This will be passed down to ChatInterface
    setChatInput(message);
  };

  const [chatInput, setChatInput] = useState("");

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        onQuickAction={handleQuickAction}
      />
      <main className="flex-1 flex flex-col min-w-0">
        <ChatInterface
          externalInput={chatInput}
          onExternalInputConsumed={() => setChatInput("")}
        />
      </main>
    </div>
  );
}
