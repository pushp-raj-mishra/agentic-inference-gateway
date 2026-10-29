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
    // const recentUncertainties: number[] = [];
    // const recentUncertaintyWindow = 5;
    // let rollingAverage = 0;
    let tokenCount = 0;
    for await (const chunk of localStream) {
      lookaheadBuffer.push(chunk);
      // recentUncertainties.push(chunk.entropy ?? 0);
      // if (recentUncertainties.length > recentUncertaintyWindow) {
      //   recentUncertainties.shift();
      // }
      // rollingAverage =
      //   recentUncertainties.reduce((sum, val) => sum + val, 0) /
      //   recentUncertainties.length;
      tokenCount++;
      const isWord = /[a-zA-Z0-9]{2,}/.test(chunk.token);
      if (
        tokenCount > 10 &&
        isWord &&
        chunk.entropy !== undefined &&
        chunk.entropy > uncertaintyThreshold
      ) {
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
      token: `safely yielded text: "${safelyYieldedText}"\n\n\x1b\x1b[0m `,
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
