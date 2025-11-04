# MI beadandó – Fordítás + Kép + Hangulatelemzés
# Készítők: Kálmán Béla (LB0RE5), Andó Ákos (D84EJM), Pál Bence (GHXXQA)
# Módosítás: első futáskor online letöltés, utána offline futás automatikusan

import os, sys
from PIL import Image, ImageDraw
from transformers import (
    pipeline, AutoTokenizer,
    AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
)

# --- opcionális: SD (ha nincs telepítve, DIFFUSERS_OK=False és fallbackre esik) ---
try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_OK = True
except Exception:
    DIFFUSERS_OK = False

# --- MODEL KONSTANSOK ---
MODEL_TRANSLATE = "Helsinki-NLP/opus-mt-hu-en"
MODEL_IMAGE     = "runwayml/stable-diffusion-v1-5"
MODEL_SENTIMENT = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# --- Segédfüggvény: modellek letöltése, ha nincsenek meg helyben ---
def ensure_model_available(model_name: str, cls):
    """Megpróbálja helyben betölteni a modellt, ha nem sikerül, online letölti."""
    try:
        cls.from_pretrained(model_name, local_files_only=True)
        print(f"✔ Modell már helyben elérhető: {model_name}")
    except Exception:
        print(f"ℹ Modell hiányzik, letöltés online: {model_name}")
        cls.from_pretrained(model_name)
        print(f"✔ Modell letöltve és cache-be mentve: {model_name}")

# --------------------- FORDÍTÁS ---------------------

def forditas_helyben(hu_text: str) -> str:
    ensure_model_available(MODEL_TRANSLATE, AutoTokenizer)
    ensure_model_available(MODEL_TRANSLATE, AutoModelForSeq2SeqLM)
    tok = AutoTokenizer.from_pretrained(MODEL_TRANSLATE, local_files_only=True)
    mod = AutoModelForSeq2SeqLM.from_pretrained(MODEL_TRANSLATE, local_files_only=True)
    trans = pipeline("translation", model=mod, tokenizer=tok, device=-1)
    out = trans(hu_text)
    return out[0]["translation_text"]

# --------------------- HANGULATELEMZÉS ---------------------

def hangulat_helyben(en_text: str):
    ensure_model_available(MODEL_SENTIMENT, AutoTokenizer)
    ensure_model_available(MODEL_SENTIMENT, AutoModelForSequenceClassification)
    tok = AutoTokenizer.from_pretrained(MODEL_SENTIMENT, local_files_only=True)
    mod = AutoModelForSequenceClassification.from_pretrained(MODEL_SENTIMENT, local_files_only=True)
    sent = pipeline("sentiment-analysis", model=mod, tokenizer=tok, device=-1)
    res = sent(en_text)[0]
    lb = (res.get("label") or "").lower()
    sc = float(res.get("score") or 0.0)
    ford = {"negative": "negatív", "neutral": "semleges", "positive": "pozitív"}
    return ford.get(lb, lb), round(sc * 100, 2)

# --------------------- KÉP GENERÁLÁS ----------------------------

def probal_generativ_kepet(prompt_en: str, cel: str) -> bool:
    if not DIFFUSERS_OK:
        return False
    try:
        # ha nincs cache-ben, egyszer online letölti
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_IMAGE,
            local_files_only=False,
            safety_checker=None
        )
        pipe = pipe.to("cpu")
        pipe.enable_attention_slicing()
        img = pipe(
            prompt_en,
            num_inference_steps=15,
            guidance_scale=7.5,
        ).images[0]
        img.save(cel)
        return True
    except Exception as e:
        print("SD generálás nem sikerült:", e)
        return False

def fallback_kep(prompt_en: str, cel: str):
    W, H = 1024, 640
    img = Image.new("RGB", (W, H), (240, 244, 248))
    can = ImageDraw.Draw(img)
    can.rectangle([(0, 0), (W, 80)], fill=(25, 118, 210))
    can.text((24, 22), "Kép (helyi fallback)", fill=(255, 255, 255))
    pad = 24
    can.rounded_rectangle([(pad, 120), (W - pad, H - pad)], radius=20,
                          outline=(100, 116, 139), width=3, fill=(250, 250, 252))
    can.text((pad + 24, 150), f"Prompt (EN): {prompt_en}", fill=(30, 41, 59))
    can.text((pad + 24, H - 70), "Megjegyzés: illusztratív helyi kép (internet nélkül).",
             fill=(71, 85, 105))
    img.save(cel)

# --------------------- FŐPROGRAM ---------------------

print("MI beadandó – Fordítás + Kép + Hangulatelemzés")
print("Adj meg egy magyar mondatot (időjárás témában különösen célszerű).")
hu = input("Bemenet: ").strip()
while not hu:
    hu = input("Kérlek, írj be valamit: ").strip()

# 1) Fordítás
print("\nFordítás folyamatban...")
try:
    en = forditas_helyben(hu)
    print("Angol változat:", en)
    print("Forrás (fordítás):", MODEL_TRANSLATE)
except Exception as e:
    print("Fordítás hiba:", e)
    sys.exit(1)

# 2) Kép generálás
print("\nKép generálása folyamatban...")
cel = os.path.join(os.getcwd(), "output.png")
if probal_generativ_kepet(en, cel):
    print("A kép elkészült:", cel)
else:
    fallback_kep(en, cel)
    print("Fallback kép készült:", cel)
print("Forrás (kép):", MODEL_IMAGE)

# 3) Hangulatelemzés
print("\nHangulatelemzés folyamatban...")
try:
    cimke_hu, szazalek = hangulat_helyben(en)
    print(f"Hangulat: {cimke_hu} ({szazalek} %)")
except Exception as e:
    print("Hangulatelemzés hiba:", e)
print("Forrás (hangulatelemzés):", MODEL_SENTIMENT)

print("\nKész.")
