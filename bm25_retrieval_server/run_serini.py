"""BM25 retrieval service.

Usage:
    curl --location --request GET 'http://localhost:5000/batch_size' \
    --header 'Content-Type: application/json'

    curl --location --request POST 'http://localhost:5000/retrieve' \
    --header 'Content-Type: application/json' \
    -d '{"inputs": ["hello world", "hello world"], "n": 2}'
"""

import argparse
from pyserini.search.lucene import LuceneSearcher
from flask import Flask, request, jsonify


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="enwiki-paragraphs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app = Flask(__name__)
    searcher = LuceneSearcher.from_prebuilt_index(args.model)

    @app.route('/', methods=["GET"])
    def health_check():
        """Confirms service is running"""
        return f"BM25 service '{args.model}' is up and running.\n"

    @app.route('/batch_size', methods=["GET"])
    def get_batch_size():
        return jsonify({"batch_size": args.batch_size})

    @app.route('/num_samples', methods=["GET"])
    def get_num_samples():
        return jsonify({"num_samples": args.num_samples})

    @app.route('/retrieve', methods=["POST"])
    def get_prediction():
        if "n" in request.json:
            num_samples = int(request.json['n'])
        else:
            num_samples = args.num_samples

        inputs = request.json['inputs']
        ids = [str(_) for _ in range(len(inputs))]
        results = searcher.batch_search(inputs, ids, k=num_samples)
        results = [[res.raw for res in results[result_id]]
                   for result_id in ids]
        results = [item for sublist in results for item in sublist]
        return jsonify({"targets": results})

    app.run(host="0.0.0.0", port=args.port)
