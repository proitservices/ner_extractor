import spacy
import json
import sys 
import re 
from collections import defaultdict

# --- I. SETUP AND LEXICON DEFINITION ---

# Load the model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading the model 'en_core_web_md'...", file=sys.stderr)
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# Sample text for testing (using the full provided text)
core_text = """
Project Brief: Quick Note Keeper & Search Tool
Requires: FastAPI, Python 3.9+, standard uuid library
The Need: Saving Notes Quickly
Hello, I run a small service where I constantly get quick bits of information—maybe a customer's unique request, a temporary tracking number, or a brief reminder about a job. Right now, I just type these into a simple text file, but it gets messy, and I can never quickly find what I'm looking for later.
I need a simple web service where I can send a piece of text, and the service does two things instantly: 1) It gives the note a totally unique name, like a secret code (a UUID), so I know for sure it's distinct. 2) It saves that exact note as a simple text file, using that unique code as the filename, inside a special folder called ./data. When it finishes, I need the service to tell me the unique code it used, just to confirm it saved correctly. This is my 'save' tool.
The Need: Finding Information Fast
The second and perhaps most critical feature is the search function. Over time, I'm going to have hundreds of these little notes. I need a way to quickly check how many times a certain word or phrase has been mentioned across all of those stored text files. For example, I might want to know how many times the word "emergency" or the phrase "delivery delay" has been written down in all my notes.
When I run a search, I'll send the specific word or phrase I'm looking for. The service must then look inside every single text file in the ./data folder and count how many times that word or phrase appears, being careful to count it even if it's capitalized differently (it needs to be case-insensitive). Finally, it must add up all the counts from all the files and give me just one final number: the total number of times the phrase was found across everything. This will be a simple search query that gives me a count. It doesn't need to tell me which file the word was in, just the total count is enough.
"""

core_text = """
We're building a small but critical microservice that keeps an eye on certain Confluence spaces that are tightly linked to our Jira projects. The whole point is to automatically detect when someone creates a brand-new page or when an existing page gets edited, and then push only the meaningful changes (or the full page if it's brand new) downstream to the rest of our platform so the rest of the automation chain can react immediately.

After I spoke with our internal IT and architecture team, they were pretty firm that this new service has to fit seamlessly into the existing microservice landscape. That means it must be written in Python 3.12, run on FastAPI (exactly like all the other services we already have in production), and be packaged as a Docker container. We'll be orchestrating everything with Docker Compose, and the customer (well, actually our own product teams) needs to be able to spin up multiple replicas of this validator just by changing the replica count in the compose file – nothing fancy, just standard Docker Compose scaling.

Day-to-day operations need to be observable the way the rest of the platform already is. There must be a simple /health endpoint that returns nothing more than a status field (“healthy” or “unhealthy”) and an errors array – empty when everything is fine, populated with short clear messages when something is wrong. Our SRE guys also insisted on integrating with our GlitchTip instance (it's the open-source Sentry alternative we run internally), so the service has to do the regular “I'm alive” heartbeat ping and forward any exceptions or errors automatically to GlitchTip – no extra work for us later.

All logs have to go to a single file at /var/log/confluence-validator.log inside the container (we'll bind-mount that to the host so we can collect them with the rest of the logs). Structured JSON logs would be perfect, but at minimum timestamp + level + message.

Now, the actual job of the service. It has to talk to the Atlassian APIs (both Jira and Confluence) using proper API tokens that we'll provide at runtime through environment variables. It should be able to:

- scan configured Confluence spaces (or spaces linked to certain Jira projects) for brand-new pages that have never been seen before and return the entire rendered page content,
- notice when an existing page was updated,
- if the update happened in, say, the last seven days (we can make that window configurable later), figure out exactly what changed and return only the new/updated portions together with enough context (section heading, paragraph number, something like that) so the downstream services know precisely where the change occurred,
- if the page was heavily rewritten (more than half the content changed) it can just return the whole new version – we don't want to over-engineer the diff logic.

Basically we want a clean, focused payloads: either “here's a completely new document” or “here's exactly what changed in document X since last time”.

Rate limiting, retries, and proper back-off against the Atlassian APIs are expected – we get throttled pretty quickly if we're not polite. Everything runs in the same Docker network as the other microservices, so internal service-to-service calls stay fast and don't leave the private network.

That's the gist of it. We need something robust, boringly reliable, and completely in line with the Python/FastAPI/Docker pattern the rest of the platform already follows. If you can deliver that, we'll be able to drop it into production next sprint with almost zero friction. Let me know if anything sounds unclear – happy to jump on a quick call and walk through a couple of example pages so you see the kind of content we're dealing with.
"""



# --- MODALITY LEXICON ---
CORE_REQUIREMENTS = ["need", "require"] 
OBLIGATION_MODALS = ["must", "have to", "shall", "is"]
SUGGESTION_MODALS = ["should", "ought to"]
ABILITY_MODALS = ["can", "could", "may", "might"]
HYPOTHETICAL_MODALS = ["would", "find"] 
ALL_REQUIREMENT_INDICATORS = CORE_REQUIREMENTS + OBLIGATION_MODALS + SUGGESTION_MODALS + ABILITY_MODALS + HYPOTHETICAL_MODALS

# --- CONSTRAINT CLEANING LIST ---
LOW_VALUE_EXCLUSIONS = {
    "just", "finally", "when", "right", "now", "here", "there", "even", 
    "then", "up", "down", "how", "where", "totally", "of", "to", "for", "with", 
    "a", "an", "the", "but", "it", "that", "those", "these", "and", "or",
    "correctly", "messy", "simple", "exact", "differently", "being" 
}
# Keywords that indicate redundant constraints when the target already captures the essence.
REDUNDANCY_KEYWORDS = ["capitalized differently", "to count it even if", "so i know for sure", "just to confirm"]

# --- UTILITY FUNCTION FOR CLEANING STRINGS ---
def clean_target_string(text):
    """Removes non-alphanumeric noise and standardizes spacing for comparison."""
    # Remove all punctuation except spaces and known path elements (like / and .)
    text = re.sub(r'[^a-zA-Z0-9\s\.\/]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


# --- II. CLASSIFICATION FUNCTIONS ---

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
    
    # --- IDENTITY/NAMING TRIGGERS ---
    identity_keywords = ["unique", "uuid", "secret code", "name", "code", "distinct", "filename"]
    if any(k in text for k in identity_keywords):
        return "IDENTITY/NAMING"

    # --- TEMPORAL TRIGGERS ---
    latency_words = ["instantly", "quickly", "sec", "min", "ms", "real-time", "immediately"]
    if any(t in text for t in latency_words):
        return "TIME/LATENCY"

    # --- LOCATION/PATH TRIGGERS ---
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

    # 1. DIRECTORY / MODULE Scope Check 
    dir_keywords = ["folder", "directory", "module", "package", "library", "config file", "./data", "path", "uri", "url"]
    if any(k in all_text for k in dir_keywords):
        return "DIRECTORY/MODULE"

    # 2. METHOD / FUNCTION Scope Check 
    method_actions = ["add", "count", "look", "tell", "give", "validate", "convert", "is", "modify", "generate", "process"]
    if target_action in method_actions or "insensitive" in all_text or "confirm" in all_text or "unique" in all_text:
        return "METHOD/FUNCTION"

    # 3. FILE Scope Check 
    file_keywords = ["file", "document", "text file", "filename", "read", "write", "save", "stored", "log"]
    if any(k in all_text for k in file_keywords):
        return "FILE"
    
    # 4. CLASS / COMPONENT Scope Check 
    class_keywords = ["service", "component", "object", "api", "handler", "interface", "class", "connection", "system"]
    if any(k in all_text for k in class_keywords):
        return "CLASS/COMPONENT"

    return "CLASS/COMPONENT" 

# --- III. NON-NLP EXTRACTION (Dependencies) ---

def extract_dependencies(text):
    """
    Scans the text for non-sentence structures like "Requires: X, Y, Z"
    and converts them into structured facts.
    """
    dependency_facts = []
    
    # Generic regex to find "Requires: A, B, C" or "Dependencies: A, B, C"
    dependency_pattern = re.compile(r'^(Requires|Dependencies|System Requirement[s]?): (.*?)$', re.MULTILINE | re.IGNORECASE)
    
    matches = dependency_pattern.findall(text)
    
    for _, dependencies_str in matches:
        # Clean up the string and split by common list separators
        dependencies_list = [
            item.strip() for item in re.split(r',\s*|;\s*', dependencies_str) if item.strip()
        ]
        
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

# --- IV. CORE EXTRACTION FUNCTION ---

def systematic_fact_extraction(text):
    """
    Processes text to systematically extract key action-object-constraint facts.
    """
    
    # 1. Start with Non-NLP facts
    extracted_facts = extract_dependencies(text)
    doc = nlp(text)

    FOLDER_PATTERN = re.compile(r'(in the |inside a |called |to |network )?(\.?[/\\\w]+\sfolder|\.?[/\\\w]+\spath|\.?[/\\\w]+\sfile)', re.IGNORECASE)

    for sent in doc.sents:
        sent_text = sent.text.strip()
        
        # Solution A: Skip initial problem description sentence if modality is weak
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

            # 4. Identify Subject (Actor)
            subject_token = None
            for child in verb_token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject_token = child
                    fact["actor"] = " ".join([t.text for t in subject_token.subtree]).strip()
                    break
            
            if not fact["actor"]:
                fact["actor"] = "User (I)" if verb_token.lemma_ in ["need", "want"] else "System (It)"

            # 5. Identify Object (Target) 
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


            # 6. Constraint Aggregation & Cleaning 
            local_constraints = [] 

            for child in verb_token.children:
                if child.dep_ in ("advmod", "prep", "acomp", "advcl", "prt"):
                    local_constraints.append(" ".join([t.text for t in child.subtree]).strip())
                if child.dep_ == "neg":
                    local_constraints.append(f"NOT {verb_token.lemma_}")

            all_constraints_for_fact = local_constraints + all_sentence_constraints + explicit_folder_constraints
            
            if all_constraints_for_fact:
                unique_constraints = sorted(list(set(c for c in all_constraints_for_fact if c)))
                
                # First pass: Filter low-value tokens and high-redundancy keywords
                mid_filter_constraints = []
                for c in unique_constraints:
                    c_lower = c.lower()
                    tokens = c_lower.split()
                    
                    # Basic low-value filter
                    if len(tokens) == 1 and tokens[0] in LOW_VALUE_EXCLUSIONS:
                        continue
                    if c_lower in LOW_VALUE_EXCLUSIONS or c_lower.strip() in ["right now", "of text", "just to confirm it saved correctly"]:
                        continue
                    
                    # Filter based on redundancy keywords
                    if any(k in c_lower for k in REDUNDANCY_KEYWORDS):
                        continue
                        
                    mid_filter_constraints.append(c)
                    
                # Second pass: Remove redundant sub-phrases 
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

            # 7. Fact Appending
            if fact["object"] and (fact["modality"] != "ACTION" or fact["indicator"] in CORE_REQUIREMENTS):
                extracted_facts.append(fact)

    # --- V. POST-PROCESSING: AGGRESSIVE DUPLICATION FILTER (V3.3 FIX) ---
    
    final_unique_facts = []
    facts_by_source = defaultdict(list)
    for fact in extracted_facts:
        facts_by_source[fact["source_sentence"]].append(fact)

    for source, facts in facts_by_source.items():
        if len(facts) <= 1:
            final_unique_facts.extend(facts)
            continue
            
        # Prioritize by: 1. Modality, 2. Target Length (Complexity)
        sorted_facts = sorted(facts, key=lambda f: (
            ["HARD_REQUIREMENT", "STRONG_SUGGESTION", "CAPABILITY", "DESIRE/HYPOTHETICAL", "ACTION"].index(f["modality"]),
            len(f["object"])
        ), reverse=True)
        
        strongest_fact = sorted_facts[0]
        final_unique_facts.append(strongest_fact)
        
        # FIX: Clean the strongest target for reliable substring checking
        strongest_target_lower_clean = clean_target_string(strongest_fact["object"])
        
        for fact in sorted_facts[1:]:
            
            # FIX: Clean the weaker target for reliable substring checking
            fact_target_lower_clean = clean_target_string(fact["object"])
            
            # Aggressive Check: If the weaker fact's target is a SUBSTRING of the strongest fact's target, drop it.
            if fact_target_lower_clean in strongest_target_lower_clean:
                continue
            
            final_unique_facts.append(fact)

    return final_unique_facts

# --- VI. MAIN EXECUTION ---

if __name__ == "__main__":
    facts = systematic_fact_extraction(core_text)

    # Print headers to STDERR
    print("\n--- 🤖 SYSTEMATIC FACT EXTRACTION (V3.3 - FINALIZED) ---", file=sys.stderr)
    print(f"Total Requirements Extracted: {len(facts)}\n", file=sys.stderr)

    # Build the final JSON structure
    structured_output = {"requirements": []} 

    for i, fact in enumerate(facts):
        scope = classify_scope(fact) 
        
        # Process and tag constraints
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

    # Print the final, clean JSON object to STDOUT
    print(json.dumps(structured_output, indent=4))
