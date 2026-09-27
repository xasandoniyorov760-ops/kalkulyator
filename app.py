import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)

# Ishlab chiqarishda "*" o'rniga faqat sizning kalkulyator sahifangiz manzilini yozing.
# Masalan: CORS(app, resources={r"/api/*": {"origins": "https://sizning-sayt.netlify.app"}})
CORS(app, resources={r"/api/*": {"origins": "*"}})

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODELS = ["gemini-flash-latest", "gemini-2.0-flash", "gemini-2.5-flash"]
GEMINI_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
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

    last_status = None
    for model in GEMINI_MODELS:
        url = GEMINI_URL_TMPL.format(model=model)
        try:
            resp = requests.post(url, params={"key": GEMINI_API_KEY}, json=payload, timeout=20)
        except requests.exceptions.RequestException:
            return jsonify({"error": "Serverga ulanishda tarmoq xatoligi yuz berdi."}), 502

        if resp.status_code == 404:
            last_status = 404
            continue  # try the next model name

        if not resp.ok:
            app.logger.error("Gemini error %s: %s", resp.status_code, resp.text[:300])
            return jsonify({"error": "AI xizmatida vaqtinchalik muammo (kod: " + str(resp.status_code) + ")."}), 502

        try:
            result = resp.json()
            answer = (
                result.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
        except (ValueError, KeyError, IndexError):
            return jsonify({"error": "AI javobini o'qib bo'lmadi."}), 502

        if not answer:
            return jsonify({"error": "AI javob bera olmadi, qayta urinib ko'ring."}), 502
        return jsonify({"answer": answer})

    return jsonify({"error": "Hech qaysi AI model topilmadi (404). API kalit sozlamalarini tekshiring."}), 502


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
