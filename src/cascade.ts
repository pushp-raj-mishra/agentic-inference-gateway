import { streamLocalOllama, streamFrontierHandoff } from "./engine.js";
import type { LLMRequest, TokenChunk } from "./types.js";

export async function* bufferedCascadeStream(
  req: LLMRequest,
  uncertaintyThreshold = 0.78,
  bufferSize = 5,
): AsyncGenerator<TokenChunk> {
  const abortController = new AbortController();
  const lookaheadBuffer: TokenChunk[] = [];
  let safelyYieldedText = "";
  let handedOff = false;

  try {
    const localStream = streamLocalOllama(
      "llama3.1:8b",
      req,
      abortController.signal,
    );

    const stopwords = new Set([
      "a",
      "an",
      "the",
      "and",
      "or",
      "but",
      "is",
      "are",
      "was",
      "were",
      "to",
      "in",
      "on",
      "at",
      "with",
      "for",
      "of",
      "that",
      "this",
      "it",
    ]);
    let consecutiveSpikes = 0;
    let tokenCount = 0;

    for await (const chunk of localStream) {
      lookaheadBuffer.push(chunk);
      tokenCount++;

      const cleanToken = chunk.token.trim().toLowerCase();
      const isWord = /[a-zA-Z0-9]{2,}/.test(chunk.token);
      const isStopword = stopwords.has(cleanToken);
      if (
        isWord &&
        isStopword &&
        chunk.entropy !== undefined &&
        chunk.entropy > uncertaintyThreshold
      ) {
        consecutiveSpikes++;
      } else {
        consecutiveSpikes = 0;
      }
      if (tokenCount > 10 && consecutiveSpikes >= 2) {
        handedOff = true;
        abortController.abort();
        break;
      }

      if (lookaheadBuffer.length >= bufferSize) {
        const tokenToYield = lookaheadBuffer.shift()!;
        safelyYieldedText += tokenToYield.token;
        yield tokenToYield;
      }
    }
  } catch (err: any) {
    if (err.name !== "AbortError") throw err;
  }

  if (!handedOff) {
    while (lookaheadBuffer.length > 0) {
      const tokenToYield = lookaheadBuffer.shift()!;
      safelyYieldedText += tokenToYield.token;
      yield tokenToYield;
    }
  }

  if (handedOff) {
    yield {
      token: `\nsafely yielded text: "${safelyYieldedText}"\n\n\x1b\x1b[0m `,
      provider: "cloud-frontier",
    };
    yield {
      token: `\n\x1b[36m⚡ [SPIKE DETECTED: DROPPING CORRUPTED BUFFER & SWITCHING TO FRONTIER]\x1b[0m `,
      provider: "cloud-frontier",
    };
    for await (const frontierChunk of streamFrontierHandoff(
      req,
      safelyYieldedText,
    )) {
      yield frontierChunk;
    }
  }
}
