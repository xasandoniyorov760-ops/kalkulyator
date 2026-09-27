import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)

# Ishlab chiqarishda "*" o'rniga faqat sizning kalkulyator sahifangiz manzilini yozing.
# Masalan: CORS(app, resources={r"/api/*": {"origins": "https://sizning-sayt.netlify.app"}})
CORS(app, resources={r"/api/*": {"origins": "*"}})

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)

SYSTEM_PROMPT = (
    "Sen — o'zbek tilida gaplashuvchi matematik yordamchisan. "
    "Foydalanuvchining savoliga qisqa, tushunarli va aniq javob ber. "
    "Agar masala bo'lsa, yechim bosqichlarini ko'rsat. "
    "Javobni faqat o'zbek tilida yoz."
)


@app.route("/api/ask", methods=["POST"])
def ask():
    if not GEMINI_API_KEY:
        return jsonify({"error": "Server tomonida API kalit sozlanmagan."}), 500

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Savol bo'sh bo'lishi mumkin emas."}), 400

    if len(question) > 500:
        return jsonify({"error": "Savol juda uzun (500 belgidan oshmasin)."}), 400

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "generationConfig": {"maxOutputTokens": 500, "temperature": 0.4},
    }

    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        result = resp.json()
        answer = (
            result.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        if not answer:
            return jsonify({"error": "AI javob bera olmadi, qayta urinib ko'ring."}), 502
        return jsonify({"answer": answer})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Serverga ulanishda xatolik: " + str(e)}), 502


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
