import { streamLocalOllama } from "./engine.js";
import type { SamplingParameters } from "./types.js";

async function runExperiment(label: string, sampling: SamplingParameters) {
  console.log(`\n========================================`);
  console.log(`EXPERIMENT: ${label}`);
  console.log(`Params:`, JSON.stringify(sampling));
  console.log(`========================================\n`);

  const stream = streamLocalOllama("qwen2.5:0.5b", {
    systemPrompt: "You are a concise backend architecture assistant.",
    userPrompt:
      "Explain why Node.js event loop can get blocked during CPU-heavy tasks in 3 bullet points.",
    sampling,
  });

  for await (const chunk of stream) {
    if (chunk.entropy !== undefined && chunk.entropy > 1.2) {
      process.stdout.write(`\x1b[33m${chunk.token}[H:${chunk.entropy}]\x1b[0m`);
    } else {
      process.stdout.write(chunk.token);
    }
  }
  console.log("\n");
}

async function main() {
  await runExperiment("1. Deterministic (Greedy Decoding)", {
    temperature: 0.0,
    topP: 1.0,
    topK: 1,
    repetitionPenalty: 1.0,
    maxTokens: 150,
  });

  await runExperiment("2. High Chaos (Temp 1.6, Top-K 100)", {
    temperature: 1.6,
    topP: 0.95,
    topK: 100,
    repetitionPenalty: 1.0,
    maxTokens: 150,
  });

  await runExperiment("3. Calibrated Balanced Sampling", {
    temperature: 0.3,
    topP: 0.85,
    topK: 20,
    repetitionPenalty: 1.15,
    maxTokens: 150,
  });
}

main().catch(console.error);
