import * as fs from 'fs';
import { pipeline } from '@xenova/transformers';

// Load the embeddings model
const extractor = await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5');

const fullOutput = [];

(async () => {
  // Scan transcripts directory for all json files
  const files = fs.readdirSync('transcripts/markdown');

  // Iterate through each file and calculate the embeddings
  for (const file of files) {
    const text = fs.readFileSync(`transcripts/markdown/${file}`, 'utf-8');
    // const json = JSON.parse(rawContents);

    // Calculate chunks based on this text
    const chunks = calculateMarkdownChunks(text);

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
      `Embeddings saved for ${file} to ${fileOut} (${output.length} chunks) (${
        files.indexOf(file) + 1
      }/${files.length})`
    );
  }

  // Save the full output to a single file
  const fileOut = `embeddings.json`;
  fs.writeFileSync(fileOut, JSON.stringify(fullOutput));
  console.log(`Complete embeddings saved to ${fileOut}`);
})();

function calculateMarkdownChunks(text) {
  const chunks = [];
  const lines = text.split('\n');
  let chunk = '';

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    // Check if the line is a header (starts with #)
    if (line.startsWith('#')) {
      // If we have accumulated a chunk, push it before starting a new one
      if (chunk) {
        chunks.push(chunk.trim());
        chunk = '';
      }
    }

    // Add the line to the current chunk
    chunk += line + '\n';
  }

  // Push the last chunk if any
  if (chunk) {
    chunks.push(chunk.trim());
  }

  return chunks;
}
