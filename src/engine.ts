import { partial } from "zod/mini";
import type { LLMRequest, TokenChunk } from "./types.js";
import "dotenv/config";

export function calculateUncertainty(
  chosenLogprob: number,
  topLogprobs?: Array<{ token: string; logprob: number }>,
): number {
  const p1 = Math.exp(chosenLogprob);
  if (!topLogprobs || topLogprobs.length < 2) {
    return Number((1 - p1).toFixed(4));
  }

  const p2 = Math.exp(topLogprobs[1].logprob);
  const margin = p1 - p2;
  const uncertainty = 1 - margin;
  return Number(Math.max(0, Math.min(1, uncertainty)).toFixed(4));
}

export async function* streamLocalOllama(
  model: string,
  req: LLMRequest,
  abortSignal?: AbortSignal,
): AsyncGenerator<TokenChunk> {
  const KAGGLE_PROXY_URL = process.env.KAGGLE_PROXY_URL;
  const response = await fetch(`${KAGGLE_PROXY_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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
        num_thread: 3,
      },
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Ollama Error: ${response.status} ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.trim()) continue;
        const data = JSON.parse(line);

        if (data.message?.content) {
          const logprobEntry = data.logprobs?.[0];
          const uncertainty =
            logprobEntry?.logprob !== undefined
              ? calculateUncertainty(
                  logprobEntry.logprob,
                  logprobEntry.top_logprobs,
                )
              : undefined;

          yield {
            token: data.message.content,
            logprob: logprobEntry?.logprob,
            entropy: uncertainty,
            provider: "ollama-local",
          };
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export async function* streamFrontierHandoff(
  req: LLMRequest,
  partialAssistantResponse: string,
): AsyncGenerator<TokenChunk> {
  const apiKey = process.env.GEMINI_API_KEY || process.env.OPENROUTER_API_KEY;

  if (!apiKey) {
    const continuationPrompt = `${req.userPrompt}\n\n[Continue this exact partial response seamlessly without repeating it]:\n${partialAssistantResponse}`;
    for await (const chunk of streamLocalOllama("deepseek-r1:1.5b", {
      ...req,
      userPrompt: continuationPrompt,
    })) {
      yield { ...chunk, provider: "cloud-frontier" };
    }
    return;
  }

  const isGemini = Boolean(process.env.GEMINI_API_KEY);
  const url = isGemini
    ? "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    : "https://openrouter.ai/api/v1/chat/completions";
  const model = isGemini ? "gemini-3.8-flash" : "recursive/flint-v1.0:free";

  const messages = [
    { role: "system", content: req.systemPrompt },
    {
      role: "user",
      content: `${req.userPrompt}\n\n=== CONTINUATION INSTRUCTIONS ===\nI have already started answering this, but got cut off. Here is the exact prefix of my response:\n\n${partialAssistantResponse}\n\nContinue the response immediately from the next word. DO NOT repeat the prefix. DO NOT add introductory text (like "Here is the rest"). Just seamlessly complete the sentence.`,
    },
  ];

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      stream: true,
      temperature: 0.2,
      messages: messages,
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error(
      `Frontier API Error: ${response.status} ${await response.text()}`,
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.replace(/^data:\s*/, "").trim();
      if (!trimmed || trimmed === "[DONE]") continue;
      try {
        const parsed = JSON.parse(trimmed);
        const content = parsed.choices?.[0]?.delta?.content;
        if (content) {
          yield { token: content, provider: "cloud-frontier" };
        }
      } catch {}
    }
  }
}
