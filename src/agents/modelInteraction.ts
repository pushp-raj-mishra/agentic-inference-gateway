import { z } from "zod";
import { streamLocalOllama } from "../inference/engine.js";

const MOCK_DATABASE: Record<string, string> = {
  redis: "STATUS: OOM (Out of Memory). Cache size exceeded 4GB",
  postgresql: "STATUS: Healthy. Connection pool at 12%",
  "api-gateway": "STATUS: Degraded. Latency spiked to 4000ms",
};

function checkServerStatus(service: string): string {
  console.log(
    `\n  [SYSTEM EXECUTING TOOL]: Checking database for '${service}'...`,
  );
  return (
    MOCK_DATABASE[service] ||
    `STATUS: Unknown service${service}. No data available.`
  );
}

const AgentResponseSchema = z.object({
  thought: z.string().describe("Your internal reasoning about what to do next"),
  action: z.enum(["check_server_status", "final_answer"]),
  actionInput: z
    .string()
    .describe("The service name if checking status, or final answer to user"),
});

export async function runAgent(userText: string, maxSteps = 3) {
  let step = 1;

  const systemPrompt = `You are a DevOps AI Agent. You have access to the following tool:
        - "check_server_status": Fetches the current health of a service (e.g., redis, postgresql,api-gateway).
        You must respond to the user query in the following JSON format:
        {
          "thought": "Your internal reasoning about what to do next",
          "action": "check_server_status" | "final_answer",
          "actionInput": "The service name if checking status, or final answer to user"
        }
        CRITICAL RULE: Never call 'check_server_status' for a service you have already checked in the conversation history. If you have the status for all requested services, immediately output 'final_answer'.`;

  let conversationHistory = `User: ${userText}\n`;

  while (step <= maxSteps) {
    console.log(`\n[AGENT STEP ${step} of ${maxSteps}]`);
    let rawOutput = "";

    const stream = streamLocalOllama("llama3.1:8b", {
      systemPrompt,
      userPrompt: conversationHistory,
      sampling: {
        temperature: 0.0,
        topP: 1.0,
        topK: 1,
        repetitionPenalty: 1.0,
        maxTokens: 150,
      },
    });

    for await (const chunk of stream) {
      rawOutput += chunk.token;
      process.stdout.write(chunk.token);
    }
    console.log("\n");

    try {
      const cleanedJSON = rawOutput
        .replace(/```json/g, "")
        .replace(/```/g, "")
        .trim();

      const parsedObject = JSON.parse(cleanedJSON);
      const validatedResult = AgentResponseSchema.parse(parsedObject);

      if (validatedResult.action === "check_server_status") {
        const status = checkServerStatus(validatedResult.actionInput);
        conversationHistory += `Agent: ${JSON.stringify(validatedResult)}\nTool Output: ${status}\n`;
      } else if (validatedResult.action === "final_answer") {
        console.log(`\n[AGENT FINAL ANSWER]: ${validatedResult.actionInput}`);
        return validatedResult.actionInput;
      }
    } catch (error: any) {
      console.error(`Error parsing or validating JSON: ${error.message}`);
      conversationHistory += `Agent: Error parsing response. Please provide a valid JSON.\n`;
    }

    step++;
  }

  console.log("\n[AGENT MAX STEPS REACHED]: Unable to provide a final answer.");
  return "Unable to provide a final answer after maximum steps.";
}

const testInput =
  "The frontend is timing out. Can you check the status of the api-gateway and redis to tell me which one is causing it?";
runAgent(testInput, 5).then(console.log).catch(console.error);
