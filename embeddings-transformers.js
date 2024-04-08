import * as fs from 'fs';
import { pipeline } from '@xenova/transformers';

// Load the embeddings model
const extractor = await pipeline(
  'feature-extraction',
  'Xenova/bge-small-en-v1.5'
);

const raw = fs.readFileSync('transcripts/_-AfhLQfb6w.json', 'utf-8');
const json = JSON.parse(raw);
const txt = json.text;
// console.log(txt);

const chunks = txt.split(/[.?!]/);

let outputJSON = { embeddings: [] };

for (let chunk of chunks) {
  let output = await extractor(chunk, {
    pooling: 'mean',
    normalize: true,
  });
  const embedding = output.tolist()[0];
  outputJSON.embeddings.push({ text: chunk, embedding });
}

const fileOut = `embeddings.json`;
fs.writeFileSync(fileOut, JSON.stringify(outputJSON));
console.log(`Embeddings saved to ${fileOut}`);
