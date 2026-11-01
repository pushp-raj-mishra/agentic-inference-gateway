export interface SamplingParameters {
  temperature: number;
  topP: number;
  topK?: number;
  repetitionPenalty: number;
  maxTokens: number;
}

export interface TokenChunk {
  token: string;
  logprob?: number;
  entropy?: number;
  provider: "ollama-local" | "cloud-frontier";
}

export interface LLMRequest {
  systemPrompt: string;
  userPrompt: string;
  sampling: SamplingParameters;
}
