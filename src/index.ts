import { bufferedCascadeStream } from "./cascade.js";

async function main() {
  console.log(
    `\nTesting Buffered Cascade Router (Local llama3.1:8b -> Frontier)...\n`,
  );

  const stream = bufferedCascadeStream(
    {
      systemPrompt: "You are a principal distributed systems engineer.",
      userPrompt:
        "Explain the exact difference between Raft leader election split-vote recovery and Paxos Phase 1a/1b prepare promises, including edge cases under network partitions.",
      sampling: {
        temperature: 0.2,
        topP: 0.9,
        topK: 20,
        repetitionPenalty: 1.1,
        maxTokens: 250,
      },
    },
    0.78,
    3,
  );

  for await (const chunk of stream) {
    if (chunk.provider === "ollama-local") {
      const color = (chunk.entropy ?? 0) > 0.7 ? "\x1b[33m" : "\x1b[32m";
      const score = chunk.entropy !== undefined ? `[U:${chunk.entropy}]` : "";
      process.stdout.write(`${color}${chunk.token}${score}\x1b[0m`);
    } else {
      process.stdout.write(`\x1b[36m${chunk.token}\x1b[0m`);
    }
  }
  console.log("\n");
}

main().catch(console.error);
