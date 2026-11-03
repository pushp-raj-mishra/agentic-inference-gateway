import { extractWithRetries } from "./queryClassifier.js";
import type { ExtractionResult } from "./queryClassifier.js";

const BASE_PERSONA = `You are an elite DevOps AI Agent. Your job is to diagnose issues, check system health, and provide actionable technical answers.`;

const TOOL_REGISTRY = `
<tools>
- "check_server_status": Fetches the current health of a service (e.g., redis, postgresql, api-gateway).
</tools>
`;

const OUTPUT_SCHEMA = `
<constraints>
1. You MUST respond ONLY in valid JSON. No markdown ticks, no conversational filler.
2. CRITICAL RULE: Never call 'check_server_status' for a service you have already checked in the conversation history. If you have the status for all requested services, immediately output 'final_answer'.

Schema:
{
  "thought": "Your internal reasoning",
  "action": "check_server_status" | "final_answer",
  "actionInput": "Service name OR final answer text"
}
</constraints>
`;

const FEW_SHOT_EXAMPLES: Record<string, string> = {
  question: `
<example>
User: Is the redis cache healthy?
Agent: {"thought": "The user is asking about redis. I need to check its status.", "action": "check_server_status", "actionInput": "redis"}
</example>`,
  complaint: `
<example>
User: The frontend is crashing and latency is huge!
Agent: {"thought": "This is an urgent complaint. I should check the api-gateway first to see if it's the bottleneck.", "action": "check_server_status", "actionInput": "api-gateway"}
</example>`,
  feature_request: `
<example>
User: Can we add a dark mode toggle?
Agent: {"thought": "This is a frontend UI feature request. It does not require checking backend server status.", "action": "final_answer", "actionInput": "I can certainly log a ticket for a dark mode toggle. No backend system checks are required for this UI update."}
</example>`,
};

export async function compilePrompt(
  userText: string,
): Promise<{ systemPrompt: string; extractedData: ExtractionResult }> {
  console.log(
    `\n  [SYSTEM COMPILING PROMPT]: Analyzing user query for intent and context...`,
  );

  const extractedData = await extractWithRetries(userText);

  const dynamicContext = `
  <context>
  - Detected User Intent: ${extractedData.intent}
  - System Urgency Level: ${extractedData.urgency}/5
  - Mentioned Technologies: ${extractedData.technologies.join(", ") || "None"}
  </context>
  `;

  const systemPrompt = `${BASE_PERSONA}\n${TOOL_REGISTRY}\n${OUTPUT_SCHEMA}\n${dynamicContext}\n${FEW_SHOT_EXAMPLES[extractedData.intent] || ""}`;

  return { systemPrompt, extractedData };
}
