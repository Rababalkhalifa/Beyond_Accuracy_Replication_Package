# Prompt Configurations, Instruction Schemas, and Response Formats

**Scope:** Canonical and diagnostic prompt templates for Frozen Instruction-Tuned LLMs (`qwen2.5:3b-instruct` and `phi3.5:latest`). Extracted verbatim from historical runners and Appendix B of the manuscript.

> [!IMPORTANT]
> **Label Integrity:**
> * **Climate Stance Classification** strictly evaluates `BELIEVER` versus `DENIER`.
> * **Yelp Sentiment Classification** strictly evaluates `POS` versus `NEG`.
> Under no circumstances are stance labels mapped to sentiment labels.

---

## 1. Climate Stance Classification (`BELIEVER` vs. `DENIER`)

### System Prompt
```text
Follow the classification instructions exactly.
```

### Prompt Configurations

#### Configuration `p1_minimal` (Minimalist Baseline)
```text
Classify the climate stance of TEXT as BELIEVER or DENIER.
Return exactly one JSON object with keys label, confidence, abstain.
Definitions:
BELIEVER: accepts, supports, or expresses concern about climate change, climate science, or climate action.
DENIER: rejects, doubts, mocks, minimizes, or argues against climate change, climate science, or climate action.
confidence must be between 0 and 1. abstain must be false.
TEXT:
{text}
```

#### Configuration `p3_json_strict` (Canonical Longitudinal Benchmark)
```text
Determine the climate stance label for TEXT. Allowed labels are exactly BELIEVER and DENIER.
Output one JSON object only. Do not add markdown or explanation.
Definitions:
BELIEVER: accepts, supports, or expresses concern about climate change, climate science, or climate action.
DENIER: rejects, doubts, mocks, minimizes, or argues against climate change, climate science, or climate action.
Use this exact structure: {"label":"BELIEVER","confidence":0.90,"abstain":false}
confidence must be a number in [0,1]. abstain must be false.
TEXT:
{text}
```

#### Configuration `p5_abstain` (Selective Prediction & Explicit Abstention)
```text
Classify the climate stance of TEXT as BELIEVER or DENIER.
Return one JSON object only with keys label, confidence, abstain.
Definitions:
BELIEVER: accepts, supports, or expresses concern about climate change, climate science, or climate action.
DENIER: rejects, doubts, mocks, minimizes, or argues against climate change, climate science, or climate action.
Set abstain=true only when the text is genuinely mixed, unclear, or insufficient for a defensible decision. Even when abstaining, provide the best-guess label and confidence no higher than 0.50.
TEXT:
{text}
```

### JSON Response Schema (Climate)
```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "label": {
      "type": "string",
      "enum": ["BELIEVER", "DENIER"]
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "abstain": {
      "type": "boolean"
    }
  },
  "required": ["label", "confidence", "abstain"]
}
```

---

## 2. Yelp Sentiment Classification (`POS` vs. `NEG`)

### System Prompt
```text
Follow the user instructions exactly. Output only what is requested.
```

### Prompt Configurations

#### Configuration `p1_minimal` (Minimalist Persona)
```text
You are a sentiment classifier.
Return ONLY valid JSON with keys: label, confidence, abstain.
Allowed labels: POS, NEG.
confidence MUST be a number between 0 and 1.
Text: {text}
```

#### Configuration `p3_json_strict` (Canonical Longitudinal Benchmark)
```text
Classify the sentiment of TEXT into exactly one label: POS or NEG.
Output MUST be one JSON object on one line. No markdown and no extra keys.
confidence must be a float in [0,1].
Use this exact shape:
{"label":"POS","confidence":0.9,"abstain":false}
TEXT:
{text}
```

#### Configuration `p5_abstain` (Cautious Classifier & Explicit Abstention)
```text
You are a cautious sentiment classifier.
Return ONLY one JSON object on one line. No markdown and no extra keys.
Allowed labels: POS, NEG.
Use this exact shape:
{"label":"NEG","confidence":0.0,"abstain":true}
If the sentiment is mixed, sarcastic, unclear, or genuinely uncertain:
- set abstain=true
- still provide your best-guess label
- confidence <= 0.5
If confident, set abstain=false and confidence closer to 1.
TEXT:
{text}
```

### JSON Response Schema (Yelp)
```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "label": {
      "type": "string",
      "enum": ["POS", "NEG"]
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "abstain": {
      "type": "boolean"
    }
  },
  "required": ["label", "confidence", "abstain"]
}
```

---

## 3. Structural Parsing and Validation Protocol

Responses from Ollama are validated through a two-stage parsing pipeline:
1. **JSON Decoding:** Direct `json.loads(response)`. If formatting fails due to leading/trailing markdown fences, a regex extractor locates `{.*}` blocks.
2. **Field Verification:**
   - `label`: Must be strictly within the allowed vocabulary (`{BELIEVER, DENIER}` or `{POS, NEG}`). Any invalid token triggers `status = "invalid_label"`.
   - `confidence`: Must be a finite floating-point value in $[0.0, 1.0]$.
   - `abstain`: Boolean. For `p1_minimal` and `p3_json_strict`, `abstain` must be `false`; any `true` flag is recorded as `unexpected_abstain`. For `p5_abstain`, if `abstain == true`, `confidence` must not exceed `0.50` (`abstain_confidence_violation`).
