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

const raw = fs.readFileSync('embeddings.json', 'utf-8');
const embeddings = JSON.parse(raw);

async function query(prompt) {
    const queryEmbedding = await getEmbeddings([prompt]);
    const similarities = embeddings.map((embedding) =>
        cosineSimilarity(embedding.embedding, queryEmbedding[0])
    );
    const sortedIndices = similarities
        .map((_, i) => i)
        .sort((a, b) => similarities[b] - similarities[a]);
    const topResults = sortedIndices.slice(0, 5).map((i) => embeddings[i]);
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

// Benchmarking code, these are the prompts we will query
// these are mostly just random ones to test the system. The actual output is discarded
const prompts = [
    'What is a five letter word?',
    'What is the largest prime number?',
    'What is the smallest composite number?',
    'What is the sum of the first 100 natural numbers?',
    'What is the probability of rolling a prime number on a six-sided die?',
    'What is the probability of flipping a coin and getting heads?',
    'What is the probability of drawing a red card from a standard deck of cards?',
    'What is the probability of drawing a face card from a standard deck of cards?',
    'What is the probability of drawing a spade from a standard deck of cards?',
    'What is the probability of drawing a red card or a face card from a standard deck of cards?',
    'What is the probability of rolling a 6 on a 6-sided die?',
    'What is the sum of the first 10 prime numbers?',
    'What is the number of degrees in a circle?',
    'What is the smallest prime number?',
    'What is the probability of flipping a coin and getting heads?',
    'What is the square root of 100?',
    'What is the smallest perfect square that is not a perfect cube?',
    'What is the only even prime number?',
    'What is the sum of the first 5 odd numbers?',
];

// Run the benchmark
(async () => {
    for (const prompt of prompts) {
        console.time('query' + prompts.indexOf(prompt));
        await query(prompt);
        console.timeEnd('query' + prompts.indexOf(prompt));
    }
})();
