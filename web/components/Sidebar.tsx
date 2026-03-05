"use client";

import { useState, useEffect } from "react";

interface Campaign {
  id: number;
  campaign_name: string;
  client_name: string;
  vertical: string;
  status: string;
}

interface HealthStatus {
  status: string;
  database: string;
  vector_store: string;
  llm: string;
}

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  onQuickAction: (message: string) => void;
}

export default function Sidebar({
  collapsed,
  onToggle,
  onQuickAction,
}: SidebarProps) {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [showCampaigns, setShowCampaigns] = useState(true);
  const [showActions, setShowActions] = useState(true);

  useEffect(() => {
    // Fetch health
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));

    // Fetch campaigns
    fetch("/api/campaigns?limit=20")
      .then((r) => r.json())
      .then((data) => setCampaigns(data.campaigns || []))
      .catch(() => setCampaigns([]));
  }, []);

  if (collapsed) {
    return (
      <div className="w-12 bg-zinc-900 flex flex-col items-center py-4">
        <button
          onClick={onToggle}
          className="text-zinc-400 hover:text-white transition-colors"
          aria-label="Expand sidebar"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
          </svg>
        </button>
      </div>
    );
  }

  const healthDot = health
    ? health.status === "healthy"
      ? "bg-green-500"
      : "bg-yellow-500"
    : "bg-red-500";

  const quickActions = [
    {
      label: "Generate Report",
      message: "Generate an LCI report for the best performing campaign",
      icon: "&#x1f4cb;",
    },
    {
      label: "Compare Campaigns",
      message: "Compare the top two QSR campaigns by ROAS",
      icon: "&#x2696;&#xfe0f;",
    },
    {
      label: "Recommend Audience",
      message: "Recommend audience segments for a new QSR lunch campaign",
      icon: "&#x1f3af;",
    },
  ];

  return (
    <div className="w-72 bg-zinc-900 text-zinc-300 flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-4 border-b border-zinc-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`w-2.5 h-2.5 rounded-full ${healthDot}`} />
          <span className="text-sm font-medium text-zinc-200">
            System {health?.status || "loading..."}
          </span>
        </div>
        <button
          onClick={onToggle}
          className="text-zinc-500 hover:text-zinc-300 transition-colors"
          aria-label="Collapse sidebar"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
          </svg>
        </button>
      </div>

      {/* Quick Actions */}
      <div className="px-4 py-3">
        <button
          onClick={() => setShowActions(!showActions)}
          className="flex items-center justify-between w-full text-xs font-semibold uppercase tracking-wider text-zinc-500 hover:text-zinc-300"
        >
          Quick Actions
          <svg
            className={`w-3 h-3 transition-transform ${showActions ? "rotate-180" : ""}`}
            fill="none" stroke="currentColor" viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {showActions && (
          <div className="mt-2 space-y-1">
            {quickActions.map((action, i) => (
              <button
                key={i}
                onClick={() => onQuickAction(action.message)}
                className="w-full text-left px-3 py-2 rounded-lg text-sm hover:bg-zinc-800 transition-colors flex items-center gap-2"
              >
                <span dangerouslySetInnerHTML={{ __html: action.icon }} />
                {action.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Campaign List */}
      <div className="flex-1 overflow-y-auto px-4 py-3 border-t border-zinc-800">
        <button
          onClick={() => setShowCampaigns(!showCampaigns)}
          className="flex items-center justify-between w-full text-xs font-semibold uppercase tracking-wider text-zinc-500 hover:text-zinc-300 mb-2"
        >
          Campaigns ({campaigns.length})
          <svg
            className={`w-3 h-3 transition-transform ${showCampaigns ? "rotate-180" : ""}`}
            fill="none" stroke="currentColor" viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {showCampaigns && (
          <div className="space-y-1">
            {campaigns.map((campaign) => (
              <button
                key={campaign.id}
                onClick={() =>
                  onQuickAction(
                    `Tell me about the ${campaign.campaign_name} campaign`
                  )
                }
                className="w-full text-left px-3 py-2 rounded-lg text-sm hover:bg-zinc-800 transition-colors"
              >
                <div className="font-medium text-zinc-200 truncate">
                  {campaign.campaign_name}
                </div>
                <div className="text-xs text-zinc-500 flex items-center gap-2">
                  <span>{campaign.client_name}</span>
                  <span className="text-zinc-700">|</span>
                  <span>{campaign.vertical}</span>
                  <span
                    className={`inline-block w-1.5 h-1.5 rounded-full ${
                      campaign.status === "completed"
                        ? "bg-green-500"
                        : campaign.status === "active"
                        ? "bg-blue-500"
                        : "bg-zinc-500"
                    }`}
                  />
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
