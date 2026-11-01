import { z } from "zod";
import { streamLocalOllama } from "../inference/engine.js";

export const ExtractionSchema = z.object({
  technologies: z.array(z.string()),
  intent: z.enum(["question", "complaint", "feature_request"]),
  urgency: z.number().min(1).max(5),
});

export type ExtractionResult = z.infer<typeof ExtractionSchema>;

export async function extractWithRetries(
  userText: string,
  maxRetries = 3,
): Promise<ExtractionResult> {
  let attempt = 0;
  let currentPrompt = userText;

  const systemPrompt = `You are a strict data extraction API. Output ONLY valid JSON matching this structure:
    {
    "technologies": ["string", "string"],
    "intent": "question" | "complaint" | "feature_request",
    "urgency": 1-5 (5 is critical)
    }
    Do not include any introductory text or markdown formatting.`;

  while (attempt < maxRetries) {
    console.log(`Attempt ${attempt + 1} of ${maxRetries} for extraction.`);
    let rawOutput = "";

    const stream = streamLocalOllama("llama3.1:8b", {
      systemPrompt,
      userPrompt: currentPrompt,
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
      const validatedResult = ExtractionSchema.parse(parsedObject);

      console.log("Success");
      return validatedResult;
    } catch (error: any) {
      console.error(`Error parsing or validating JSON: ${error.message}`);
      attempt++;
      currentPrompt = `${userText}\n\nERROR: Your previous output failed validation with this error: ${error.message}. Please output strictly formatted JSON that fixes this error.`;
    }
  }
  throw new Error(`Failed to extract valid JSON after ${maxRetries} attempts.`);
}

const testInput =
  "Hey, my production server is crashing because the Redis cache is overflowing. Node.js event loop is blocked. Fix this immediately!";

extractWithRetries(testInput)
  .then((data) => console.log("FINAL DATA:", data))
  .catch(console.error);
