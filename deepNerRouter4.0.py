# ./deepNerRouter1.0.py
# Updated with department-specific high-level topic classifiers based on provided keywords.
# - TopicClassifier.HIGH_LEVEL_TOPICS now maps team/department to their keyword lists for classification.
# - classify_topics: Scans text for keywords → adds matching teams to 'high_level' set.
# - Low-level keywords: Extracted from text via NER entities + matched keywords from all lists.
# - In systematic_fact_extraction: Adds 'topics' to each fact; can be used for event records.
# - IntentClassifier: Uses updated topics for legitimacy checks (prompt topics subset of allowed for user's teams).
# - TEAM_ALLOWED_TOPICS: Inferred from provided teams; each team allows its own topics (extend if cross-access needed).
# - No changes to core NER logic; enhancements additive for intent/topic classification.
# - For paraphrase/jailbreak: Metaphor mappings help remap (e.g., "tree" → "database structure" → matches "sysops_team").

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
Tell me about Kubernetes monitoring with Zabbix.
What are the sales opportunities in CRM?
Describe the structure of a tree in the forest.  # Potential jailbreak paraphrase for database tables
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

# ────────────────────────────────────────────────
# VII. CLASSIFICATION HELPERS
# ────────────────────────────────────────────────

def classify_modality(token_lemma):
    """Classifies the severity of a requirement."""
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
    """Tags a constraint based on keywords."""
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
    """Classifies the scope of a requirement into one of four architectural levels."""
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
    """Classify coarse question intent type based on pattern matching."""
    lower_text = sent_text.lower()
    for qtype, pattern in QUESTION_TYPE_PATTERNS.items():
        if re.search(pattern, lower_text, re.IGNORECASE):
            return qtype
    for starter, words in QUESTION_STARTERS.items():
        if any(lower_text.startswith(w) for w in words):
            return f"{starter}_QUESTION"
    return "OTHER_QUESTION"

# ────────────────────────────────────────────────
# VIII. NON-NLP EXTRACTION (Dependencies)
# ────────────────────────────────────────────────
def extract_dependencies(text):
    """
    Scans the text for non-sentence structures like "Requires: X, Y, Z"
    and converts them into structured facts.
    """
    dependency_facts = []
    dependency_pattern = re.compile(r'^(Requires|Dependencies|System Requirement[s]?): (.*?)$', re.MULTILINE | re.IGNORECASE)
    matches = dependency_pattern.findall(text)
    for _, dependencies_str in matches:
        dependencies_list = [item.strip() for item in re.split(r',\s*|;\s*', dependencies_str) if item.strip()]
        if dependencies_list:
            dependency_facts.append({
                "actor": "System",
                "modality": "HARD_REQUIREMENT",
                "indicator": "require",
                "target_action": "use",
                "object": f"Dependencies: {', '.join(dependencies_list)}",
                "constraints": [],
                "source_sentence": f"Requires: {dependencies_str.strip()}"
            })
    return dependency_facts

# ────────────────────────────────────────────────
# IX. CORE EXTRACTION FUNCTION
# ────────────────────────────────────────────────
def systematic_fact_extraction(text):
    """
    Processes text to systematically extract key action-object-constraint facts.
    """
    extracted_facts = extract_dependencies(text)
    doc = nlp(text)
    FOLDER_PATTERN = re.compile(r'(in the |inside a |called |to |network )?(\.?[/\\\w]+\sfolder|\.?[/\\\w]+\spath|\.?[/\\\w]+\sfile)', re.IGNORECASE)
    for sent in doc.sents:
        sent_text = sent.text.strip()
        has_strong_modality = any(token.lemma_ in CORE_REQUIREMENTS or token.lemma_ in OBLIGATION_MODALS for token in sent)
        if not has_strong_modality and ("Right now" in sent_text or "Hello, I run" in sent_text):
             continue
        explicit_folder_constraints = []
        folder_matches = FOLDER_PATTERN.findall(sent_text)
        if folder_matches:
            for prefix, folder_name in folder_matches:
                full_phrase = (prefix + folder_name).strip()
                explicit_folder_constraints.append(full_phrase)
        all_sentence_constraints = []
        for token in sent:
            if token.dep_ in ("prep", "advcl", "advmod", "appos"):
                if token.head.pos_ != "VERB" or token.dep_ in ("advcl", "advmod"):
                    all_sentence_constraints.append(" ".join([t.text for t in token.subtree]).strip())
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
                "actor": "", "modality": modality, "indicator": action_token.lemma_,
                "target_action": target_action, "object": "", "constraints": [],
                "source_sentence": sent_text
            }
            subject_token = None
            for child in verb_token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject_token = child
                    fact["actor"] = " ".join([t.text for t in subject_token.subtree]).strip()
                    break
            if not fact["actor"]:
                fact["actor"] = "User (I)" if verb_token.lemma_ in ["need", "want"] else "System (It)"
            object_phrase = []
            for child in verb_token.children:
                if child.dep_ in ("dobj", "ccomp", "attr"):
                    object_phrase.append(" ".join([t.text for t in child.subtree]))
            if not object_phrase and verb_token.lemma_ in CORE_REQUIREMENTS + OBLIGATION_MODALS:
                 for child in verb_token.children:
                     if child.dep_ == "xcomp":
                         object_phrase.append(" ".join([t.text for t in child.subtree]))
            if object_phrase:
                fact["object"] = " / ".join(object_phrase).strip()
            local_constraints = []
            for child in verb_token.children:
                if child.dep_ in ("advmod", "prep", "acomp", "advcl", "prt"):
                    local_constraints.append(" ".join([t.text for t in child.subtree]).strip())
                if child.dep_ == "neg":
                    local_constraints.append(f"NOT {verb_token.lemma_}")
            all_constraints_for_fact = local_constraints + all_sentence_constraints + explicit_folder_constraints
            if all_constraints_for_fact:
                unique_constraints = sorted(list(set(c for c in all_constraints_for_fact if c)))
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
                    is_redundant = False
                    c1_lower = c1.lower()
                    for j, c2 in enumerate(mid_filter_constraints):
                        if i == j:
                            continue
                        if len(c2) > len(c1) and c1_lower in c2.lower():
                            is_redundant = True
                            break
                    if not is_redundant:
                        final_constraints.append(c1)
                fact["constraints"] = final_constraints
            if fact["object"] and (fact["modality"] != "ACTION" or fact["indicator"] in CORE_REQUIREMENTS):
                extracted_facts.append(fact)
    final_unique_facts = []
    facts_by_source = defaultdict(list)
    for fact in extracted_facts:
        facts_by_source[fact["source_sentence"]].append(fact)
    for source, facts in facts_by_source.items():
        if len(facts) <= 1:
            final_unique_facts.extend(facts)
            continue
        sorted_facts = sorted(facts, key=lambda f: (
            ["HARD_REQUIREMENT", "STRONG_SUGGESTION", "CAPABILITY", "DESIRE/HYPOTHETICAL", "ACTION"].index(f["modality"]),
            len(f["object"])
        ), reverse=True)
        strongest_fact = sorted_facts[0]
        final_unique_facts.append(strongest_fact)
        strongest_target_lower_clean = clean_target_string(strongest_fact["object"])
        for fact in sorted_facts[1:]:
            fact_target_lower_clean = clean_target_string(fact["object"])
            if fact_target_lower_clean in strongest_target_lower_clean:
                continue
            final_unique_facts.append(fact)
    return final_unique_facts

# ────────────────────────────────────────────────
# X. MAIN EXECUTION
# ────────────────────────────────────────────────
if __name__ == "__main__":
    facts = systematic_fact_extraction(core_text)
    print("\n--- 🤖 SYSTEMATIC FACT EXTRACTION (V3.3 - FINALIZED) ---", file=sys.stderr)
    print(f"Total Requirements Extracted: {len(facts)}\n", file=sys.stderr)
    structured_output = {"requirements": []}
    for i, fact in enumerate(facts):
        scope = classify_scope(fact)
        tagged_rules = []
        for rule in fact["constraints"]:
            tag = tag_constraint_type(rule)
            tagged_rules.append({
                "type": tag,
                "description": rule
            })
        req = {
            "id": f"REQ_{i+1:02d}",
            "actor": fact["actor"],
            "priority": fact["modality"],
            "scope": scope,
            "indicator": fact["indicator"],
            "target_action": fact["target_action"],
            "target": fact["object"],
            "rules": tagged_rules,
            "source": fact["source_sentence"]
        }
        structured_output["requirements"].append(req)
    print(json.dumps(structured_output, indent=4))
