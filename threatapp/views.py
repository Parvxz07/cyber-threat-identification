from django.shortcuts import render
from classifier.model import predict
from classifier.preprocess import flag_keywords


def index(request):
    result = None
    if request.method == "POST":
        text = request.POST.get("input_text", "").strip()
        if text:
            label, confidence = predict(text)
            flagged = flag_keywords(text)
            result = {
                "text": text,
                "label": label,
                "confidence": confidence,
                "flagged_keywords": flagged,
            }
    return render(request, "index.html", {"result": result})
