# gamf-mestinal-la01-beadando
NJE-GAMF / Mérnökinformatikus Bsc / 2025 / Mesterséges intelligencia alapjai / Beadandó

# MI beadandó – Fordítás + Kép + Hangulatelemzés

**Készítők:**  
- Kálmán Béla (LB0RE5)  
- Andó Ákos (D84EJM)  
- Pál Bence (GHXXQA)  

**Módosítás:** az első futtatáskor a modellek online töltődnek le, később offline használat is lehetséges.

---

## Áttekintés

Ez a program három fő funkciót valósít meg:

1. **Fordítás magyar → angol**  
   A `Helsinki-NLP/opus-mt-hu-en` modell segítségével fordítja le a magyar szöveget angolra.

2. **Kép generálás**  
   A `runwayml/stable-diffusion-v1-5` modellt használva generál illusztratív képeket a fordított angol prompt alapján. Ha a diffusers pipeline nincs telepítve vagy nem működik, a program egy helyi fallback képet hoz létre.

3. **Hangulatelemzés**  
   A `cardiffnlp/twitter-roberta-base-sentiment-latest` modell elemzi a szöveg hangulatát (pozitív, negatív, semleges).

A program interaktív módon kéri be a magyar szöveget, majd lépésről lépésre végrehajtja a fordítást, képgenerálást és hangulatelemzést.

---

## Szükséges feltételek

- **Python verzió:** 3.12 vagy korábbi  
- **Függőségek:**  
  ```text
  pillow
  transformers
  torch
  diffusers
  accelerate
  safetensors
  sentencepiece
  protobuf

## Futtatás

A program futtatásához a következő parancsokat kell végrehajtani:

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
python main.py