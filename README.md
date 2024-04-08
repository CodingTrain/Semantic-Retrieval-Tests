# Semantic Retrieval

This repository features the testing code (and probably final code) we used for extracting the embeddings out of video transcripts for [Bizarro-Devin](https://github.com/CodingTrain/Bizarro-Devin/). There are a few files in this repository, all having their own purpose

[embeddings-transformers.js](/embeddings-transformers.js) is the file that generates embeddings from transcripts in the `transcripts` directory
[semantic-retrieval.js](/semantic-retrieval.js) can be used for retrieving from the embeddings based on a query
[semantic-retrieval-benchmark.js](/semantic-retrieval-benchmark.js) is used for benchmarking the retrieval, during my own tests it was ~180ms / retrieval

## How to use

### Generating embeddings

1. Make sure you've installed all dependencies by running `npm install`
2. Create a directory called `transcripts` and insert all json transcript files in here. Each file being a transcript of a video. The transcript json should be in the following format:

```json
{
    "text": "full transcript text",
    "chunks": [
        {
            "timestamp": [0.48, 7.04],
            "text": "..."
        }
    ]
}
```

However, the chunks array is currently not used. So this can be left out. 3. Create a `embeddings` directory for the embeddings of each transcript to be written to 4. Run `node embeddings-transformers.js` to run the script that generates the embeddings.
All embeddings should now be in the embeddings folder, as well as an `embeddings.json` file being present in the current working directory. This `embeddings.json` file is the combination of all embeddings generated from the transcripts.

### Semantic retrieval from embeddings

1. Make sure you've installed all dependencies by running `npm install`
2. Make sure you have the embeddings you want to retrieve from in an `embeddings.json` file. This file is usually already generated if you've generated them using the previous [generating embeddings](#generating-embeddings) section.
3. Open up the `semantic-retrieval.js` file and edit your query on line `25`.
4. Save the file and run `node semantic-retrieval.js` to retrieve the top 5 results from the embeddings.
