import { HuggingFaceInference } from "@langchain/community/llms/hf";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const imageSchema = z
    .object({
        name: z.string().describe("Give a name of the photo based on the context"),
        tags: z
            .array(z.string())
            .describe(
                "Tags that might be relevant to the photo for example: 'beach', 'sunset', 'people' etc. Give at lease 5 tags"
            ),
        location: z
            .string()
            .describe(
                "Location of the photo based on the title of the image or the context. Try to include the state and the country if possible"
            ),
    })
    .describe("Information about a person.");

const extractJsonFromOutput = (message: string) => {
    const pattern = /```json\s*((.|\n)*?)\s*```/gs;

    const matches = pattern.exec(message);

    if (matches && matches[1]) {
        try {
            return JSON.parse(matches[1].trim());
        } catch (error) {
            throw new Error(`Failed to parse: ${matches[1]}`);
        }
    } else {
        throw new Error(`No JSON found in: ${message}`);
    }
};

const imageUrl = "https://github.com/Suv4o/wallpaper-images/blob/master/41-Image-On-The-Edge-Together.jpg?raw=true";

const SYSTEM_PROMPT_TEMPLATE = `### Image URL:
${imageUrl}

You must return your answer as JSON that matches the given schema:
{schema}
Make sure to wrap the answer in \`\`\`json and \`\`\` tags.`;

const prompt = ChatPromptTemplate.fromMessages([
    ["system", SYSTEM_PROMPT_TEMPLATE],
    ["human", "{query}"],
]);

const model = new HuggingFaceInference({
    model: "meta-llama/Llama-3.2-11B-Vision-Instruct",
    apiKey: process.env.HF_API_KEY,
});

const query = "Extract information from the image";

const promptValue = await prompt.invoke({
    schema: zodToJsonSchema(imageSchema),
    query,
});

promptValue.toString();

const chain = prompt.pipe(model).pipe(extractJsonFromOutput);

const res = await chain.invoke({
    schema: zodToJsonSchema(imageSchema),
    query,
});

console.log(res);
