import { streamLocalOllama, streamFrontierHandoff } from "./engine.js";
import type { LLMRequest, TokenChunk } from "./types.js";

export async function* speculativeCascadeStream(
  req: LLMRequest,
  uncertaintyThreshold = 0.78,
): AsyncGenerator<TokenChunk> {
  const abortController = new AbortController();
  let generatedPrefix = "";
  const recentUncertainties: number[] = [];
  let handedOff = false;

  try {
    const localStream = streamLocalOllama(
      "qwen2.5:0.5b",
      req,
      abortController.signal,
    );

    for await (const chunk of localStream) {
      const isWord = /[a-zA-Z0-9]{2,}/.test(chunk.token);
      if (isWord && chunk.entropy !== undefined) {
        recentUncertainties.push(chunk.entropy);
        if (recentUncertainties.length > 4) recentUncertainties.shift();
      }

      const rollingAvg =
        recentUncertainties.length >= 4
          ? recentUncertainties.reduce((a, b) => a + b, 0) / 4
          : 0;

      if (rollingAvg > uncertaintyThreshold && generatedPrefix.length > 25) {
        handedOff = true;
        abortController.abort();
        break;
      }

      generatedPrefix += chunk.token;
      yield chunk;
    }
  } catch (err: any) {
    if (err.name !== "AbortError") throw err;
  }

  if (handedOff) {
    yield {
      token: `\n\x1b[36m⚡ [CASCADE HANDOFF TRIGGERED -> SWITCHING TO FRONTIER MODEL]\x1b[0m `,
      provider: "cloud-frontier",
    };
    for await (const frontierChunk of streamFrontierHandoff(
      req,
      generatedPrefix,
    )) {
      yield frontierChunk;
    }
  }
}
