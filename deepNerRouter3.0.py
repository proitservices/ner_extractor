
# ./deepNerRouter1.0.py
# Fixed: restored full post-processing (duplicate filter) logic that was accidentally truncated
# Now returns final_unique_facts correctly
# All previous features (questions, intent, normalization, timing) preserved

import spacy
import json
import sys
import re
import time
from collections import defaultdict

# ────────────────────────────────────────────────
# I. SPAZY MODEL SETUP
# ────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading the model 'en_core_web_md'...", file=sys.stderr)
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

print("Model loaded.", file=sys.stderr)

# ────────────────────────────────────────────────
# II. SAMPLE INPUT (for testing)
# ────────────────────────────────────────────────
core_text = """
Data is made of structures, doors are a structure, locks are a structure.
What is the correct structure for the Access Point lldp request to obtain the switch its connected to?
Data is made of structures, doors are a structure, locks are a structure, what is the key to unlock the system?
How do I get LLDP to work with Cisco switches?
Why is the port showing down if VLAN 10 is configured?
Could I use this AP with PoE+ given only 802.3af is available?
When I run show lldp neighbors the device behaves like timeout.
Should I, given no CDP support, be able to see the neighbor?
What is the command to obtain neighbor table?
Where can I find information on LLDP TLVs given I am working on Aruba?
When I send lldp packet the switch drops it, why is this?
"""

# ────────────────────────────────────────────────
# III. LEXICONS & CONFIG ── REQUIREMENTS
# ────────────────────────────────────────────────

CORE_REQUIREMENTS = ["need", "require"]
OBLIGATION_MODALS = ["must", "have to", "shall", "is"]
SUGGESTION_MODALS = ["should", "ought to"]
ABILITY_MODALS = ["can", "could", "may", "might"]
HYPOTHETICAL_MODALS = ["would", "find"]
ALL_REQUIREMENT_INDICATORS = CORE_REQUIREMENTS + OBLIGATION_MODALS + SUGGESTION_MODALS + ABILITY_MODALS + HYPOTHETICAL_MODALS

LOW_VALUE_EXCLUSIONS = {
    "just", "finally", "when", "right", "now", "here", "there", "even",
    "then", "up", "down", "how", "where", "totally", "of", "to", "for", "with",
    "a", "an", "the", "but", "it", "that", "those", "these", "and", "or",
    "correctly", "messy", "simple", "exact", "differently", "being"
}

REDUNDANCY_KEYWORDS = ["capitalized differently", "to count it even if", "so i know for sure", "just to confirm"]

# ────────────────────────────────────────────────
# IV. LEXICONS & CONFIG ── QUESTIONS
# ────────────────────────────────────────────────

QUESTION_STARTERS = {
    "HOW":      ["how", "how do", "how can", "how to", "how does", "how would"],
    "WHY":      ["why", "why does", "why is", "why would"],
    "CAN":      ["can", "could", "is it possible", "am i able to", "can i"],
    "SHOULD":   ["should", "should i", "ought i", "is it recommended"],
    "WHAT":     ["what", "what is", "what are", "what's the"],
    "WHERE":    ["where", "where can i", "where do i", "where is"],
    "WHEN":     ["when", "when i", "when does"],
    "WHO":      ["who", "whom", "who can", "who should"],
}

QUESTION_TYPE_PATTERNS = {
    "TROUBLESHOOTING_CAUSE":     r"(why|why is|why does|why would).*?(behave|happening|occur|fail|not work|error|issue|problem|drop|timeout)",
    "TROUBLESHOOTING_BEHAVIOR":  r"(when|whenever).*?(i do|perform|execute|try|run|send).*?(happens|occurs|results|behaves|does|drops)",
    "CAPABILITY":                r"(can|could|able to|possible to).*?(with this|given|if|under|in case of)",
    "RECOMMENDATION":            r"(should|should i|ought to).*?(given|if|when|in case|under).*?(try|switch|do|use|avoid)",
    "COMMAND_SYNTAX":            r"(command|syntax|way|how).*?(obtain|get|run|execute|perform|trigger|show)",
    "INFORMATION_LOCATION":      r"(where|where can|where do).*?(find|look up|read|see|documentation|info|guide|manual)",
    "RESPONSIBILITY":            r"(who|whom).*?(should|can|able to|responsible).*?(tell|explain|help|support|answer)",
    "GENERAL_WHAT":              r"what is|what are|what's",
}

# ────────────────────────────────────────────────
# V. LEXICONS & CONFIG ── INTENT & PARAPHRASING
# ────────────────────────────────────────────────

DOMAIN_LEXICON = {
    "ap": "Access Point",
    "access point": "Access Point",
    "sw": "switch",
    "ccu": "compute unit",
    "lldp": "LLDP",
    "vlan": "VLAN",
    "poe": "PoE",
    "cdp": "CDP",
    "tlv": "TLV",
}

SUSPICIOUS_PATTERNS = [
    r"key.*unlock|unlock.*key",
    r"password|passcode|credential|secret key|master key",
    r"hack|exploit|backdoor|breach|root|admin access",
    r"unlock the system|open the system|break into",
]

# ────────────────────────────────────────────────
# VI. UTILITY FUNCTIONS
# ────────────────────────────────────────────────

def clean_target_string(text):
    """Removes non-alphanumeric noise and standardizes spacing for comparison."""
    text = re.sub(r'[^a-zA-Z0-9\s\.\/]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def classify_intent(sent_text: str) -> tuple:
    """Classify sentence intent as 'legit' or 'malicious'."""
    lower_text = sent_text.lower()
    
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, lower_text):
            return "malicious", f"Suspicious pattern matched: '{pattern}' – unnatural in networking context"
    
    has_domain_term = any(term in lower_text for term in DOMAIN_LEXICON)
    prefix_present = "data is made of structures" in lower_text
    
    if prefix_present and not has_domain_term and "structure" in lower_text:
        return "malicious", "Prefix pattern detected without domain-relevant follow-up – possible probe"
    
    return "legit", "Query aligns with networking/telecom domain"

def normalize_terms(text: str) -> str:
    """Replace domain abbreviations/synonyms with canonical terms."""
    lower_text = text.lower()
    for abbr, full in DOMAIN_LEXICON.items():
        lower_text = re.sub(r'\b' + re.escape(abbr) + r'\b', full.lower(), lower_text)
    return lower_text

# ────────────────────────────────────────────────
# VII. CLASSIFICATION HELPERS
# ────────────────────────────────────────────────

def classify_modality(token_lemma):
    if token_lemma in CORE_REQUIREMENTS or token_lemma in OBLIGATION_MODALS:
        return "HARD_REQUIREMENT"
    if token_lemma in SUGGESTION_MODALS:
        return "STRONG_SUGGESTION"
    if token_lemma in ABILITY_MODALS:
        return "CAPABILITY"
    if token_lemma in HYPOTHETICAL_MODALS:
        return "DESIRE/HYPOTHETICAL"
    return "ACTION"

def tag_constraint_type(constraint_text):
    text = constraint_text.lower()
    identity_keywords = ["unique", "uuid", "secret code", "name", "code", "distinct", "filename"]
    if any(k in text for k in identity_keywords):
        return "IDENTITY/NAMING"
    latency_words = ["instantly", "quickly", "sec", "min", "ms", "real-time", "immediately"]
    if any(t in text for t in latency_words):
        return "TIME/LATENCY"
    location_path_keywords = [
        "folder", "directory", "file", "database", "disk", "server",
        "location", "path", "uri", "url", "network", "./data"
    ]
    if any(k in text for k in location_path_keywords):
        return "LOCATION/PATH"
    return "BEHAVIORAL"

def classify_scope(fact):
    target = fact.get("object", "").lower()
    target_action = fact.get("target_action", "").lower()
    constraints = [c.lower() for c in fact.get("constraints", [])]
    all_text = f"{target} {target_action} {' '.join(constraints)}"

    dir_keywords = ["folder", "directory", "module", "package", "library", "config file", "./data", "path", "uri", "url"]
    if any(k in all_text for k in dir_keywords):
        return "DIRECTORY/MODULE"

    method_actions = ["add", "count", "look", "tell", "give", "validate", "convert", "is", "modify", "generate", "process"]
    if target_action in method_actions or "insensitive" in all_text or "confirm" in all_text or "unique" in all_text:
        return "METHOD/FUNCTION"

    file_keywords = ["file", "document", "text file", "filename", "read", "write", "save", "stored", "log"]
    if any(k in all_text for k in file_keywords):
        return "FILE"

    class_keywords = ["service", "component", "object", "api", "handler", "interface", "class", "connection", "system"]
    if any(k in all_text for k in class_keywords):
        return "CLASS/COMPONENT"
    return "CLASS/COMPONENT"

def classify_question_type(sent_text: str) -> str:
    lower_text = sent_text.lower()
    for qtype, pattern in QUESTION_TYPE_PATTERNS.items():
        if re.search(pattern, lower_text, re.IGNORECASE):
            return qtype
    for starter, words in QUESTION_STARTERS.items():
        if any(lower_text.startswith(w) for w in words):
            return f"{starter}_QUESTION"
    return "OTHER_QUESTION"

# ────────────────────────────────────────────────
# VIII. NON-NLP EXTRACTION
# ────────────────────────────────────────────────

def extract_dependencies(text):
    dependency_facts = []
    pattern = re.compile(r'^(Requires|Dependencies|System Requirement[s]?): (.*?)$', re.MULTILINE | re.IGNORECASE)
    for _, deps_str in pattern.findall(text):
        deps_list = [item.strip() for item in re.split(r',\s*|;\s*', deps_str) if item.strip()]
        if deps_list:
            dependency_facts.append({
                "actor": "System",
                "modality": "HARD_REQUIREMENT",
                "indicator": "require",
                "target_action": "use",
                "object": f"Dependencies: {', '.join(deps_list)}",
                "constraints": [],
                "source_sentence": f"Requires: {deps_str.strip()}"
            })
    return dependency_facts

# ────────────────────────────────────────────────
# IX. QUESTION EXTRACTION HELPER
# ────────────────────────────────────────────────

def extract_question_fact(sent) -> list:
    sent_text = sent.text.strip()
    if not (sent_text.endswith("?") or (sent and sent[0].lemma_ in ["how","why","what","where","when","who","should","can","could"])):
        return []

    intent, intent_reason = classify_intent(sent_text)
    qtype = classify_question_type(sent_text)

    main_verb = next((t for t in sent if t.dep_ == "ROOT" and t.pos_ == "VERB"), None)
    if not main_verb:
        main_verb = next((t for t in sent if t.pos_ == "VERB"), None)

    target_action = main_verb.lemma_ if main_verb else ""
    object_parts = []
    for child in (main_verb.children if main_verb else []):
        if child.dep_ in ("dobj", "attr", "ccomp", "xcomp"):
            object_parts.append(" ".join(t.text for t in child.subtree).strip())

    constraints = []
    for token in sent:
        if token.dep_ in ("advmod", "prep", "advcl", "acomp", "obl"):
            subtree = " ".join(t.text for t in token.subtree).strip()
            if len(subtree.split()) > 1:
                constraints.append(subtree)

    normalized_object = normalize_terms(" ".join(object_parts))
    normalized_constraints = [normalize_terms(c) for c in constraints]

    fact = {
        "modality": "QUESTION",
        "question_type": qtype,
        "indicator": sent[0].text if sent else "",
        "target_action": target_action,
        "object": normalized_object.strip(),
        "constraints": normalized_constraints,
        "source_sentence": sent_text,
        "intent": intent,
        "intent_reason": intent_reason
    }

    if fact["object"] or fact["target_action"] or fact["question_type"] != "OTHER_QUESTION":
        return [fact]
    return []

# ────────────────────────────────────────────────
# X. CORE EXTRACTION LOGIC
# ────────────────────────────────────────────────

def systematic_fact_extraction(text):
    """
    Processes text → extracts both requirements and questions.
    """
    extracted_facts = extract_dependencies(text)
    doc = nlp(text)

    FOLDER_PATTERN = re.compile(r'(in the |inside a |called |to |network )?(\.?[/\\\w]+\sfolder|\.?[/\\\w]+\spath|\.?[/\\\w]+\sfile)', re.IGNORECASE)

    for sent in doc.sents:
        sent_text = sent.text.strip()

        has_strong_modality = any(t.lemma_ in CORE_REQUIREMENTS or t.lemma_ in OBLIGATION_MODALS for t in sent)
        if not has_strong_modality and ("Right now" in sent_text or "Hello, I run" in sent_text):
            continue

        explicit_folder_constraints = []
        for prefix, folder_name in FOLDER_PATTERN.findall(sent_text):
            explicit_folder_constraints.append((prefix + folder_name).strip())

        all_sentence_constraints = []
        for token in sent:
            if token.dep_ in ("prep", "advcl", "advmod", "appos"):
                if token.head.pos_ != "VERB" or token.dep_ in ("advcl", "advmod"):
                    all_sentence_constraints.append(" ".join(t.text for t in token.subtree).strip())

        is_question_sentence = (
            sent_text.endswith("?") or
            (sent and sent[0].lemma_ in ["how", "why", "what", "where", "when", "who", "should", "can", "could"])
        )

        if is_question_sentence:
            question_facts = extract_question_fact(sent)
            extracted_facts.extend(question_facts)
            continue

        for verb_token in sent:
            if verb_token.pos_ != "VERB" and verb_token.lemma_ not in CORE_REQUIREMENTS:
                continue
            if verb_token.dep_ not in ("ROOT", "ccomp", "xcomp", "advcl", "relcl", "conj") and verb_token.lemma_ not in CORE_REQUIREMENTS:
                continue

            action_token = verb_token
            target_action = verb_token.lemma_
            modality = classify_modality(verb_token.lemma_)

            for child in verb_token.children:
                if child.dep_ == "aux" and child.pos_ == "AUX" and child.lemma_ in ALL_REQUIREMENT_INDICATORS:
                    action_token = child
                    modality = classify_modality(child.lemma_)
                    break

            fact = {
                "actor": "",
                "modality": modality,
                "indicator": action_token.lemma_,
                "target_action": target_action,
                "object": "",
                "constraints": [],
                "source_sentence": sent_text
            }

            for child in verb_token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    fact["actor"] = " ".join(t.text for t in child.subtree).strip()
                    break
            if not fact["actor"]:
                fact["actor"] = "User (I)" if verb_token.lemma_ in ["need", "want"] else "System (It)"

            object_phrase = []
            for child in verb_token.children:
                if child.dep_ in ("dobj", "ccomp", "attr"):
                    object_phrase.append(" ".join(t.text for t in child.subtree))
            if not object_phrase and verb_token.lemma_ in CORE_REQUIREMENTS + OBLIGATION_MODALS:
                for child in verb_token.children:
                    if child.dep_ == "xcomp":
                        object_phrase.append(" ".join(t.text for t in child.subtree))

            if object_phrase:
                fact["object"] = " / ".join(object_phrase).strip()

            local_constraints = []
            for child in verb_token.children:
                if child.dep_ in ("advmod", "prep", "acomp", "advcl", "prt"):
                    local_constraints.append(" ".join(t.text for t in child.subtree).strip())
                if child.dep_ == "neg":
                    local_constraints.append(f"NOT {verb_token.lemma_}")

            all_constraints_for_fact = local_constraints + all_sentence_constraints + explicit_folder_constraints

            if all_constraints_for_fact:
                unique_constraints = sorted(set(c for c in all_constraints_for_fact if c))
                mid_filter_constraints = []
                for c in unique_constraints:
                    c_lower = c.lower()
                    tokens = c_lower.split()
                    if len(tokens) == 1 and tokens[0] in LOW_VALUE_EXCLUSIONS:
                        continue
                    if c_lower in LOW_VALUE_EXCLUSIONS or c_lower.strip() in ["right now", "of text", "just to confirm it saved correctly"]:
                        continue
                    if any(k in c_lower for k in REDUNDANCY_KEYWORDS):
                        continue
                    mid_filter_constraints.append(c)

                final_constraints = []
                mid_filter_constraints.sort(key=len, reverse=True)
                for i, c1 in enumerate(mid_filter_constraints):
                    c1_lower = c1.lower()
                    is_redundant = any(
                        len(c2) > len(c1) and c1_lower in c2.lower()
                        for j, c2 in enumerate(mid_filter_constraints) if i != j
                    )
                    if not is_redundant:
                        final_constraints.append(c1)

                fact["constraints"] = final_constraints

            if fact["object"] and (fact["modality"] != "ACTION" or fact["indicator"] in CORE_REQUIREMENTS):
                extracted_facts.append(fact)

    # ── POST-PROCESSING: DUPLICATE FILTER ────────────────
    final_unique_facts = []
    facts_by_source = defaultdict(list)
    for fact in extracted_facts:
        facts_by_source[fact["source_sentence"]].append(fact)

    for source, facts_group in facts_by_source.items():
        if len(facts_group) <= 1:
            final_unique_facts.extend(facts_group)
            continue

        sorted_facts = sorted(facts_group, key=lambda f: (
            ["HARD_REQUIREMENT", "STRONG_SUGGESTION", "CAPABILITY", "DESIRE/HYPOTHETICAL", "ACTION", "QUESTION"].index(f["modality"]),
            len(f.get("object", ""))
        ), reverse=True)

        strongest = sorted_facts[0]
        final_unique_facts.append(strongest)

        strongest_clean = clean_target_string(strongest.get("object", ""))
        for weaker in sorted_facts[1:]:
            weaker_clean = clean_target_string(weaker.get("object", ""))
            if weaker_clean and weaker_clean in strongest_clean:
                continue
            final_unique_facts.append(weaker)

    return final_unique_facts

# ────────────────────────────────────────────────
# XI. MAIN EXECUTION & OUTPUT
# ────────────────────────────────────────────────

if __name__ == "__main__":
    start_time = time.perf_counter()

    facts = systematic_fact_extraction(core_text)

    end_time = time.perf_counter()
    duration_sec = end_time - start_time
    duration_ms = duration_sec * 1000

    print(f"\n--- 🤖 DEEP NER ROUTER 1.0 (REQUIREMENTS + QUESTIONS + INTENT) ---", file=sys.stderr)
    print(f"Extraction completed in {duration_sec:.3f} s ({duration_ms:.0f} ms)", file=sys.stderr)
    print(f"Total items extracted: {len(facts)}\n", file=sys.stderr)

    structured_output = {
        "requirements": [],
        "questions": []
    }

    for fact in facts:
        if fact.get("modality") == "QUESTION":
            constraints_tagged = [
                {"type": tag_constraint_type(c), "description": c}
                for c in fact.get("constraints", [])
            ]
            item = {
                "id": f"Q_{len(structured_output['questions'])+1:02d}",
                "type": fact.get("question_type", "UNKNOWN"),
                "indicator": fact.get("indicator", ""),
                "target_action": fact.get("target_action", ""),
                "target": fact["object"],
                "constraints": constraints_tagged,
                "source": fact["source_sentence"],
                "intent": fact.get("intent", "unknown"),
                "intent_reason": fact.get("intent_reason", "")
            }
            structured_output["questions"].append(item)
        else:
            scope = classify_scope(fact)
            tagged_rules = [
                {"type": tag_constraint_type(r), "description": r}
                for r in fact.get("constraints", [])
            ]
            item = {
                "id": f"REQ_{len(structured_output['requirements'])+1:02d}",
                "actor": fact.get("actor", ""),
                "priority": fact["modality"],
                "scope": scope,
                "indicator": fact["indicator"],
                "target_action": fact["target_action"],
                "target": fact["object"],
                "rules": tagged_rules,
                "source": fact["source_sentence"]
            }
            structured_output["requirements"].append(item)

    print(json.dumps(structured_output, indent=4))
