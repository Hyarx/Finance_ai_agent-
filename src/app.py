from flask import Flask, request, jsonify
from agent import team

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    response = team.run(question, stream=False)
    answer = response.content if hasattr(response, "content") else str(response)
    return jsonify({"answer": answer})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)