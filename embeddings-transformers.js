import * as fs from 'fs';
import { pipeline } from '@xenova/transformers';

// Load the embeddings model
const extractor = await pipeline(
    'feature-extraction',
    'Xenova/bge-small-en-v1.5'
);

const fullOutput = [];

(async () => {
    // Scan transcripts directory for all json files
    const files = fs.readdirSync('transcripts');

    // Iterate through each file and calculate the embeddings
    for (const file of files) {
        const rawContents = fs.readFileSync(`transcripts/${file}`, 'utf-8');
        const json = JSON.parse(rawContents);

        const text = json.text;

        // Calculate chunks based on this text
        const chunks = calculateChunks(text);

        // Extract embeddings for each chunk
        const output = [];

        for (const chunk of chunks) {
            const embeddingOutput = await extractor(chunk, {
                pooling: 'mean',
                normalize: true,
            });

            const embedding = embeddingOutput.tolist()[0];
            output.push({ text: chunk, embedding });
            fullOutput.push({ text: chunk, embedding });
        }

        // Save the embeddings to a file
        const fileOut = `embeddings/${file}`;
        fs.writeFileSync(fileOut, JSON.stringify(output));

        console.log(
            `Embeddings saved for ${file} to ${fileOut} (${
                output.length
            } chunks) (${files.indexOf(file) + 1}/${files.length})`
        );
    }

    // Save the full output to a single file
    const fileOut = `embeddings.json`;
    fs.writeFileSync(fileOut, JSON.stringify(fullOutput));
    console.log(`Complete embeddings saved to ${fileOut}`);
})();

function calculateChunks(text) {
    // We want to split the text into chunks of at least 100 characters, after this we will keep adding to the chunk until we find a sentence boundary
    const chunks = [];
    let chunk = '';
    for (let i = 0; i < text.length; i++) {
        chunk += text[i];

        // If our current character is a punctuation mark, we will split the chunk here
        if (
            chunk.length >= 100 &&
            (text[i] === '.' || text[i] === '?' || text[i] === '!')
        ) {
            chunks.push(chunk.trim());
            chunk = '';
        }

        // If we are exceeding 150 characters and we haven't found a punctuation mark, we will split the chunk at the last space
        if (chunk.length >= 150) {
            let lastSpace = chunk.lastIndexOf(' ');
            if (lastSpace === -1) {
                lastSpace = chunk.length;
            }
            chunks.push(chunk.slice(0, lastSpace).trim());
            chunk = chunk.slice(lastSpace).trim();
        }
    }

    return chunks;
}
