import * as fs from 'fs';
import { pipeline } from '@xenova/transformers';

// Load the embeddings model
const extractor = await pipeline(
  'feature-extraction',
  'Xenova/bge-small-en-v1.5'
);

async function getEmbeddings(sentences) {
  let embeddings = [];
  for (let sentence of sentences) {
    let output = await extractor(sentence, {
      pooling: 'mean',
      normalize: true,
    });
    embeddings.push(output.data);
  }
  return embeddings;
}

const raw = fs.readFileSync('transcript-embeddings.json', 'utf-8');
const { embeddings } = JSON.parse(raw);

query('What is a five letter word?');

async function query(prompt) {
  const queryEmbedding = await getEmbeddings([prompt]);
  const similarities = embeddings.map((embedding) =>
    cosineSimilarity(embedding.embedding, queryEmbedding[0])
  );
  const sortedIndices = similarities
    .map((_, i) => i)
    .sort((a, b) => similarities[b] - similarities[a]);
  const topResults = sortedIndices.slice(0, 5).map((i) => embeddings[i]);

  for (let result of topResults) {
    console.log(result.text);
    console.log('-----------');
  }
}

// Function to calculate dot product of two vectors
function dotProduct(vecA, vecB) {
  return vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
}

// Function to calculate the magnitude of a vector
function magnitude(vec) {
  return Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
}

// Function to calculate cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
  const numerator = dotProduct(vecA, vecB);
  const denominator = magnitude(vecA) * magnitude(vecB);
  // console.log(numerator, denominator);
  return numerator / denominator;
}
