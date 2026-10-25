import type { LLMRequest, TokenChunk } from "./types.js";

export function calculateTokenEntropy(
  topLogProbs: Array<{ token: string; logprob: number }>,
): number {
  if (!topLogProbs || topLogProbs.length === 0) {
    return 0;
  }
  const probs = topLogProbs.map((item) => Math.exp(item.logprob));
  const sumProbs = probs.reduce((a, b) => a + b, 0);
  const normalizedProbs = probs.map((p) => p / sumProbs);
  const entropy = -normalizedProbs.reduce(
    (sum, p) => sum + p * Math.log2(p),
    0,
  );
  return Number(entropy.toFixed(4));
}

export async function* streamLocalOllama(
  model: string,
  req: LLMRequest,
  abortSignal?: AbortSignal,
): AsyncGenerator<TokenChunk> {
  const response = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    signal: abortSignal,
    body: JSON.stringify({
      model,
      stream: true,
      logprobs: true,
      top_logprobs: 5,
      messages: [
        { role: "system", content: req.systemPrompt },
        { role: "user", content: req.userPrompt },
      ],
      options: {
        temperature: req.sampling.temperature,
        top_p: req.sampling.topP,
        top_k: req.sampling.topK ?? 40,
        repeat_penalty: req.sampling.repetitionPenalty,
        num_predict: req.sampling.maxTokens,
        num_ctx: 2048,
        num_threads: 3,
      },
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Ollama Error: ${response.status} ${response.statusText}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.trim() === "") continue;
      const data = JSON.parse(line);
      if (data.message?.content) {
        const logprobEntry = data.logprobs?.[0];
        const entropy = logprobEntry?.top_logprobs
          ? calculateTokenEntropy(logprobEntry.top_logprobs)
          : undefined;
        yield {
          token: data.message.content,
          logprob: logprobEntry?.logprob,
          entropy,
          provider: "ollama-local",
        };
      }
    }
  }
}
