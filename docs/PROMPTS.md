# Prompt Configurations, Instruction Schemas, and Response Formats

**Scope:** Canonical and diagnostic prompt templates for Frozen Instruction-Tuned LLMs (`qwen2.5:3b-instruct` and `phi3.5:latest`). Extracted verbatim from Appendix A (pp. 35–36) of the manuscript and historical runners.

* **Evaluated Frozen Model Artifacts (Ollama):**
  * `qwen2.5:3b-instruct` (`digest: 357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`, `quantization: Q4_K_M`)
  * `phi3.5:latest` (`digest: 61819fb370a3c1a9be6694869331e5f85f867a079e9271d66cb223acb81d04ba`, `quantization: Q4_0`)

> [!IMPORTANT]
> **Label Integrity:**
> * **Climate Stance Classification** strictly evaluates `BELIEVER` versus `DENIER`.
> * **Yelp Sentiment Classification** strictly evaluates `POS` versus `NEG`.
> Under no circumstances are stance labels mapped to sentiment labels.

---

## 1. Climate Stance Classification (`BELIEVER` vs. `DENIER` — Manuscript Appendix A.1)

### System Prompt
```text
Follow the user instructions exactly. Output only what is requested.
```

### Prompt Configurations

#### Configuration `p1_minimal` (`Minimal`)
```text
You are a climate-stance classifier.
Return ONLY valid JSON with keys: label, confidence, abstain.
Allowed labels: BELIEVER, DENIER.

Definitions:
- BELIEVER: the text accepts, supports, or expresses concern about climate change, climate science, or climate action.
- DENIER: the text rejects, doubts, mocks, minimizes, or argues against climate change, climate science, or climate action.

confidence MUST be a number between 0 and 1.
Text: {text}
```

#### Configuration `p3_json_strict` (`Strict JSON` — Canonical Longitudinal Benchmark)
```text
Classify the climate stance of TEXT into exactly one label: BELIEVER or DENIER.
Output MUST be one JSON object on one line. No markdown.
confidence must be a float in [0,1].

Definitions:
BELIEVER = accepts, supports, or expresses concern about climate change, climate science, or climate action.
DENIER = rejects, doubts, mocks, minimizes, or argues against climate change, climate science, or climate action.

Allowed JSON examples:
{"label":"BELIEVER","confidence":0.9,"abstain":false}
{"label":"DENIER","confidence":0.9,"abstain":false}

TEXT:
{text}
```

#### Configuration `p5_abstain` (`Abstention-aware` — Selective Prediction & Explicit Abstention)
```text
You are a cautious climate-stance classifier.
Return ONLY one JSON object on one line. No markdown.
Allowed labels: BELIEVER, DENIER.

Definitions:
BELIEVER = accepts, supports, or expresses concern about climate change, climate science, or climate action.
DENIER = rejects, doubts, mocks, minimizes, or argues against climate change, climate science, or climate action.

Use this exact JSON shape:
{"label":"BELIEVER","confidence":0.0,"abstain":true}

If the text is mixed, sarcastic, unclear, only quotes someone else's stance without endorsing it, or is not clearly about climate stance:
- set abstain=true
- still provide your best-guess label
- confidence <= 0.5

If confident:
- set abstain=false
- confidence closer to 1.

TEXT:
{text}
```

> **Provenance Note (Unified Reproducibility Runner):** In the later unified runner (`code/reproducibility/run_paper_models.py`), a condensed equivalent formulation of the Climate prompts (`Determine the climate stance label for TEXT...`) was also implemented; both formulations are preserved in `prompts/prompts.json`.

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

## 2. Yelp Sentiment Classification (`POS` vs. `NEG` — Manuscript Appendix A.2)

### System Prompt
```text
Follow the user instructions exactly. Output only what is requested.
```

### Prompt Configurations

#### Configuration `p1_minimal` (`Minimal`)
```text
You are a sentiment classifier.
Return ONLY valid JSON with keys: label, confidence, abstain.
Allowed labels: POS, NEG.
confidence MUST be a number between 0 and 1.
Text: {text}
```

#### Configuration `p3_json_strict` (`Strict JSON` — Canonical Longitudinal Benchmark)
```text
Classify the sentiment of TEXT into exactly one label: POS or NEG.
Output MUST be one JSON object on one line. No markdown and no extra keys.
confidence must be a float in [0,1].
Use this exact shape:
{"label":"POS","confidence":0.9,"abstain":false}
TEXT:
{text}
```

#### Configuration `p5_abstain` (`Abstention-aware` — Cautious Classifier & Explicit Abstention)
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
   - `abstain`: Boolean. For `p1_minimal` (`Minimal`) and `p3_json_strict` (`Strict-JSON`), `abstain` must be `false`; any `true` flag is recorded as `unexpected_abstain`. For `p5_abstain` (`Abstention-aware`), if `abstain == true`, `confidence` must not exceed `0.50` (`abstain_confidence_violation`).
