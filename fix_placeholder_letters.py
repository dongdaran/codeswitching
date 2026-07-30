import json
import re
from collections import Counter, defaultdict
from pathlib import Path


KO = Path("data/kcl_mcqa_ko_rulebase_placeholders.jsonl")
EN = Path("data/translated_kcl_mcqa_en_wo_precedent.jsonl")
OUT = Path("data/modify_en.jsonl")
REVIEW = Path("data/modify_en_review.jsonl")

FIELDS = ("question", "A", "B", "C", "D", "E")
MANUAL_REPAIR_LINES = {
    32,
    106,
    161,
    168,
    218,
    219,
    221,
    226,
    237,
    239,
    240,
    241,
    242,
    244,
    274,
    278,
}
MAPS = (
    ("A", "G"),
    ("B", "H"),
    ("C", "J"),
    ("D", "K"),
    ("E", "L"),
    ("F", "Q"),
)

VERB_OR_CLAUSE_AFTER = {
    "and",
    "or",
    "who",
    "whose",
    "whom",
    "had",
    "has",
    "have",
    "entered",
    "lent",
    "borrowed",
    "sold",
    "bought",
    "purchased",
    "assigned",
    "transferred",
    "granted",
    "filed",
    "paid",
    "received",
    "owns",
    "owned",
    "died",
    "argues",
    "claimed",
    "claims",
    "may",
    "cannot",
    "can",
    "must",
    "shall",
    "was",
    "were",
    "is",
    "does",
    "did",
    "obtained",
    "established",
    "concluded",
    "made",
    "gave",
    "created",
    "failed",
    "requested",
    "intends",
    "intended",
    "agreed",
    "applied",
    "provided",
    "performed",
    "rescinded",
    "cancelled",
    "canceled",
    "consumed",
    "notified",
    "asserted",
    "proves",
    "proved",
    "demanded",
    "entrusted",
    "appointed",
    "appeals",
    "assaulted",
    "bears",
    "compensated",
    "complete",
    "constructed",
    "delayed",
    "denies",
    "exclusively",
    "expended",
    "faces",
    "files",
    "incurred",
    "instructed",
    "issued",
    "later",
    "newly",
    "notifies",
    "participated",
    "pays",
    "placed",
    "recognized",
    "returned",
    "set",
    "still",
    "urged",
    "withdraws",
    "withdrew",
    "registered",
    "completed",
}

PREPOSITIONS_BEFORE = {
    "against",
    "to",
    "from",
    "by",
    "with",
    "between",
    "for",
    "of",
    "on",
    "in",
    "into",
    "upon",
    "after",
    "before",
    "than",
    "through",
    "toward",
    "towards",
    "over",
}

ENTITY_BEFORE = {
    "party",
    "creditor",
    "debtor",
    "director",
    "seller",
    "buyer",
    "plaintiff",
    "defendant",
    "decedent",
    "entertainer",
    "victim",
    "person",
}

PROTECTED_LITERAL_PATTERNS = {
    "A": [
        r"\bClaim A\b",
        r"\b[Ll]and A\b",
        r"\bA [Ll]and\b",
        r"\bCompany A\b",
        r"\bA Company\b",
        r"\bA Co\.",
        r"\bA Corporation\b",
        r"\bInsurance Company A\b",
        r"\bA Insurance Company\b",
        r"\bCity A\b",
        r"\bA City\b",
        r"\bAdministrative Agency A\b",
        r"\bA Church\b",
        r"\bA Research Foundation\b",
        r"\bA University\b",
        r"\bVictim A\b",
        r"\bvictim A\b",
        r"\bA \(age",
        r"\bA \(born",
        r"\bA \(200",
        r"\bA's wallet\b",
        r"\bA’s wallet\b",
        r"\bA's account\b",
        r"\bA’s account\b",
        r"\bA's passbook\b",
        r"\bA’s passbook\b",
        r"\bA's driver",
        r"\bA’s driver",
        r"\bA's nude\b",
        r"\bA’s nude\b",
        r"\bA's Bitcoin\b",
        r"\bA’s Bitcoin\b",
        r"\bA's bitcoins\b",
        r"\bA’s bitcoins\b",
    ],
    "B": [
        r"\bClaim B\b",
        r"\bCompany B\b",
        r"\bB Company\b",
        r"\bB Corporation\b",
        r"\bB District\b",
        r"\bDistrict B\b",
        r"\bB University\b",
        r"\bVictim B\b",
        r"\bvictim B\b",
    ],
    "C": [r"\bCompany C\b", r"\bC Company\b", r"\bVictim C\b", r"\bvictim C\b"],
    "D": [r"\bCompany D\b", r"\bD Company\b"],
    "E": [],
    "F": [],
}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def token_re(letter):
    return re.compile(rf"(?<![A-Za-z]){re.escape(letter)}(?![A-Za-z])")


def ko_latin_re(letter):
    return re.compile(rf"(?<![A-Za-z]){re.escape(letter)}(?![a-z])")


def protected_spans(text, letter):
    spans = []
    for pattern in PROTECTED_LITERAL_PATTERNS.get(letter, []):
        spans.extend(match.span() for match in re.finditer(pattern, text))
    return spans


def in_spans(match, spans):
    return any(start <= match.start() and match.end() <= end for start, end in spans)


def next_word(text, end):
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text[end:])
    return words[0] if words else ""


def prev_word(text, start):
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text[:start])
    return words[-1].lower() if words else ""


def looks_like_article_a(text, match):
    if match.end() < len(text) and text[match.end()] in {"'", "’"}:
        return False
    if match.end() < len(text) and text[match.end()] in {",", ".", ";", ":", ")"}:
        return False
    if prev_word(text, match.start()) in PREPOSITIONS_BEFORE:
        return False
    if prev_word(text, match.start()) in ENTITY_BEFORE:
        return False
    word = next_word(text, match.end())
    if not word:
        return False
    if word.lower() in VERB_OR_CLAUSE_AFTER:
        return False
    # Uppercase next words are usually names/titles: "A Company" is handled by
    # protected spans; "A few" and "A claim" should stay as articles.
    return word[:1].islower()


def count_token(text, letter):
    return len(list(token_re(letter).finditer(text)))


def token_context(text, match, width=45):
    return text[max(0, match.start() - width) : min(len(text), match.end() + width)]


def candidate_score(text, old, match, spans):
    if in_spans(match, spans):
        return -100

    score = 0
    prev = prev_word(text, match.start())
    nxt = next_word(text, match.end()).lower()
    after = text[match.end() : match.end() + 12]
    before = text[max(0, match.start() - 12) : match.start()]

    if match.end() < len(text) and text[match.end()] in {"'", "’"}:
        score += 5
    if prev in PREPOSITIONS_BEFORE or prev in ENTITY_BEFORE:
        score += 4
    if nxt in VERB_OR_CLAUSE_AFTER:
        score += 4
    if re.match(r"\s*(,|;|:|\)|\.)", after):
        score += 3
    if re.search(r"\b(and|or)\s+$", before, flags=re.I):
        score += 3
    if re.match(r"\s+(and|or)\b", after, flags=re.I):
        score += 3
    if re.search(r"\b(v\.|vs\.|versus)\s+$", before, flags=re.I):
        score += 3

    if old == "A" and looks_like_article_a(text, match):
        score -= 5
    if re.match(r"\s+[a-z]", after):
        score -= 1

    return score


def replace_by_target_count(en_text, old, new, target_count):
    already_new = count_token(en_text, new)
    needed = target_count - already_new
    if needed <= 0:
        return en_text, 0, None

    spans = protected_spans(en_text, old)
    matches = list(token_re(old).finditer(en_text))
    ranked = sorted(
        enumerate(matches),
        key=lambda item: (
            candidate_score(en_text, old, item[1], spans),
            -item[1].start(),
        ),
        reverse=True,
    )
    selected = {idx for idx, match in ranked[:needed] if candidate_score(en_text, old, match, spans) > -100}

    issue = None
    if len(selected) < needed:
        issue = {
            "reason": "not enough replaceable candidates for target count",
            "target_count": target_count,
            "existing_new_count": already_new,
            "needed": needed,
            "selected": len(selected),
            "english_contexts": [token_context(en_text, m) for m in matches],
        }

    pieces = []
    last = 0
    changed = 0
    for idx, match in enumerate(matches):
        pieces.append(en_text[last : match.start()])
        if idx in selected:
            pieces.append(new)
            changed += 1
        else:
            pieces.append(match.group(0))
        last = match.end()
    pieces.append(en_text[last:])
    return "".join(pieces), changed, issue


def manual_repair(line_no, fixed):
    """Small row-level repairs for source translations that used other letters."""
    if line_no == 32:
        fixed["B"] = fixed["B"].replace("and also cannot exercise", "and G also cannot exercise")
    elif line_no == 106:
        fixed["C"] = fixed["C"].replace(
            "then because the unconstitutional decision",
            "then because, for G, the unconstitutional decision",
        )
    elif line_no == 161:
        fixed["E"] = (
            "If G entered into a pre-sale agreement with K on 2024. 2. 1. concerning real property X "
            "and, pursuant to the agreement between G and K to utilize the provisional registration, "
            "completed an additional registration transferring the provisional registration in K's name, "
            "then if J, by subrogation to G, claims cancellation of the provisional registration, "
            "K cannot oppose J by the agreement between G and K to utilize the provisional registration."
        )
    elif line_no == 168:
        fixed["D"] = fixed["D"].replace("If the completed Building Y has defects", "If Building Y completed by H has defects")
        fixed["E"] = fixed["E"].replace("If the completed Building Y has serious defects", "If Building Y completed by H has serious defects")
    elif line_no == 218:
        fixed["question"] = fixed["question"].replace("paramour A were recorded", "A, H's paramour, were recorded")
    elif line_no == 219:
        fixed["question"] = fixed["question"].replace("H selected X convenience store", "H selected X convenience store, H")
    elif line_no == 221:
        fixed["A"] = fixed["A"].replace(
            "Even if G assaulted X at X’s request, and in the process X suffered",
            "Even if A assaulted G at G’s request, and in the process G suffered",
        ).replace("such request by X", "such request by G")
        fixed["C"] = fixed["C"].replace("Whether X was beaten", "Whether G was beaten").replace("X’s own will", "his own will")
        fixed["E"] = (
            fixed["E"]
            .replace("X was guilty", "G was guilty")
            .replace("indicted X for", "indicted G for")
            .replace("not to prosecute G", "not to prosecute A")
            .replace("X confessed", "G confessed")
            .replace("X’s own", "his own")
            .replace("against G was rendered", "against A was rendered")
            .replace("applied to X", "applied to G")
        )
    elif line_no == 226:
        fixed["question"] = fixed["question"].replace("unless A consents", "unless H consents")
    elif line_no == 237:
        fixed["question"] = fixed["question"].replace("X received", "G received").replace("for X's mobile phone", "for G's mobile phone")
        fixed["C"] = fixed["C"].replace("the account of X", "the account of G").replace("X is deemed", "G is deemed").replace("if X arbitrarily", "if G arbitrarily").replace("administers G’s affairs", "administers A’s affairs")
    elif line_no == 239:
        fixed["question"] = (
            fixed["question"]
            .replace("X changed lanes", "G changed lanes")
            .replace("victims G and H", "victims A and B")
            .replace("Y, while driving", "H, while driving")
            .replace("against X for", "against G for")
            .replace("against Y for", "against H for")
        )
        fixed["A"] = fixed["A"].replace("against X", "against G")
        fixed["B"] = fixed["B"].replace("of X", "of G")
        fixed["C"] = fixed["C"].replace("If X caused", "If G caused").replace("if X’s passenger", "if G’s passenger")
        fixed["D"] = fixed["D"].replace("If Y was booked", "If H was booked")
        fixed["E"] = fixed["E"].replace("With respect to Y's", "With respect to H's").replace("if Y's drunk", "if H's drunk").replace("from Y's negligent", "from H's negligent")
    elif line_no == 240:
        for field in FIELDS:
            if field in fixed:
                fixed[field] = token_re("X").sub("G", fixed[field])
                fixed[field] = token_re("Y").sub("H", fixed[field])
                fixed[field] = fixed[field].replace("甲", "G")
        fixed["question"] = (
            fixed["question"]
            .replace("entrusted by G with a laptop computer owned by G", "entrusted by A with a laptop computer owned by A")
            .replace("having been entrusted by B, G sold", "having been entrusted by B, sold")
        )
        fixed["C"] = fixed["C"].replace("that G had been entrusted", "that he had been entrusted")
        fixed["E"] = fixed["E"].replace("acquitted G of both", "acquitted the defendant of both").replace("found G guilty", "found the defendant guilty")
    elif line_no == 241:
        for field in FIELDS:
            if field in fixed:
                fixed[field] = fixed[field].replace("B", "G")
        fixed["question"] = fixed["question"].replace("colliding with G", "colliding with A").replace("pocket of G’s", "pocket of A’s")
    elif line_no == 242:
        fixed["question"] = (
            fixed["question"]
            .replace("W and X conspired", "G and H conspired")
            .replace("W lured G, H, and J", "G lured A, B, and C")
            .replace("brought by X", "brought by H")
            .replace("from X, who observed", "from H, who observed")
            .replace("of G, H, and J", "of A, B, and C")
            .replace("W gambled", "G gambled")
            .replace("from G, H, and J", "from A, B, and C")
            .replace("(b) Y,", "(b) J,")
            .replace("with G, H, and J", "with A, B, and C")
            .replace("threatened them", "threatened A, B, and C")
            .replace("In (b), Y constitutes", "In (b), J constitutes")
            .replace("if Y, due", "if J, due")
            .replace("served on Y", "served on J")
            .replace("where Z,", "where K,")
            .replace("committed by W and X", "committed by G and H")
            .replace("fraud against G, H, and J", "fraud against A, B, and C")
        )
    elif line_no == 244:
        for field in FIELDS:
            if field in fixed:
                fixed[field] = token_re("X").sub("G", fixed[field])
                fixed[field] = token_re("Y").sub("H", fixed[field])
                fixed[field] = token_re("W").sub("J", fixed[field])
        fixed["question"] = fixed["question"].replace("that G had withdrawn", "that A had withdrawn").replace("following G", "following A").replace("When G entered", "When A entered").replace("bag G was holding", "bag A was holding")
        fixed["B"] = fixed["B"].replace("H would not bear", "he would not bear")
        fixed["C"] = fixed["C"].replace("G can be evaluated", "he can be evaluated").replace("G constitutes", "he constitutes")
        fixed["D"] = fixed["D"].replace("H is not punishable", "he is not punishable")
    elif line_no == 274:
        fixed["question"] = fixed["question"].replace("A Church", "G Church")
    elif line_no == 278:
        fixed["question"] += (
            " (a) Administrative litigation by G against H seeking performance of, or confirmation of the obligation to grant, "
            "the river occupation permit cannot be allowed."
            " (b) If, during G's lawsuit for confirmation of the illegality of omission, H issues a disposition rejecting G's "
            "application for a river occupation permit, the lawsuit loses its interest and becomes improper."
            " (c) If G filed a lawsuit for confirmation of illegality of omission after going through the prior procedure, "
            "then changed it to a lawsuit seeking revocation of H's refusal disposition and additionally joined the omission claim, "
            "the filing period is still deemed satisfied."
            " (d) If G's claim for confirmation of illegality of omission is upheld and the judgment becomes final, H must grant "
            "the river occupation permit as applied for by G."
        )
    return fixed


def source_events(ko_text, old, new):
    events = []
    for match in token_re(new).finditer(ko_text):
        events.append((match.start(), "placeholder"))
    for match in ko_latin_re(old).finditer(ko_text):
        events.append((match.start(), "literal"))
    events.sort(key=lambda item: item[0])
    return events


def replace_mixed_by_order(en_text, old, new, events):
    matches = list(token_re(old).finditer(en_text))
    if len(matches) != len(events):
        return en_text, 0, {
            "reason": "source/english token count mismatch in mixed field",
            "source_events": [kind for _, kind in events],
            "source_event_count": len(events),
            "english_token_count": len(matches),
            "english_contexts": [
                en_text[max(0, m.start() - 50) : m.end() + 50] for m in matches
            ],
        }

    pieces = []
    last = 0
    changed = 0
    for match, (_, kind) in zip(matches, events):
        pieces.append(en_text[last : match.start()])
        if kind == "placeholder":
            pieces.append(new)
            changed += 1
        else:
            pieces.append(match.group(0))
        last = match.end()
    pieces.append(en_text[last:])
    return "".join(pieces), changed, None


def replace_unmixed(en_text, old, new):
    spans = protected_spans(en_text, old)
    pieces = []
    last = 0
    changed = 0
    for match in token_re(old).finditer(en_text):
        pieces.append(en_text[last : match.start()])
        keep = in_spans(match, spans)
        if old == "A" and looks_like_article_a(en_text, match):
            keep = True
        if keep:
            pieces.append(match.group(0))
        else:
            pieces.append(new)
            changed += 1
        last = match.end()
    pieces.append(en_text[last:])
    return "".join(pieces), changed


def main():
    ko_rows = load_jsonl(KO)
    en_rows = load_jsonl(EN)
    if len(ko_rows) != len(en_rows):
        raise SystemExit(f"row count mismatch: {KO}={len(ko_rows)} {EN}={len(en_rows)}")

    review_rows = []
    by_map = Counter()
    by_field = defaultdict(Counter)

    with OUT.open("w", encoding="utf-8") as out:
        for line_no, (ko, en) in enumerate(zip(ko_rows, en_rows), 1):
            fixed = dict(en)
            row_review = {"line": line_no, "meta": en.get("meta"), "fields": []}
            for field in FIELDS:
                ko_text = str(ko.get(field, ""))
                en_text = fixed.get(field)
                if not isinstance(en_text, str):
                    continue

                issues = []
                changes = []
                for old, new in MAPS:
                    target_count = count_token(ko_text, new)
                    if target_count == 0:
                        continue
                    en_text, changed, issue = replace_by_target_count(
                        en_text, old, new, target_count
                    )

                    if changed:
                        by_map[f"{old}->{new}"] += changed
                        by_field[field][f"{old}->{new}"] += changed
                        changes.append({"map": f"{old}->{new}", "count": changed})
                    if issue:
                        issue.update(
                            {
                                "map": f"{old}->{new}",
                                "source_text": ko_text,
                                "english_text_before": fixed.get(field),
                            }
                        )
                        issues.append(issue)

                fixed[field] = en_text
                if issues:
                    row_review["fields"].append(
                        {
                            "field": field,
                            "changes": changes,
                            "issues": issues,
                            "english_text_after": en_text,
                        }
                    )

            if row_review["fields"]:
                review_rows.append(row_review)
            out.write(json.dumps(fixed, ensure_ascii=False) + "\n")

    with REVIEW.open("w", encoding="utf-8") as out:
        for row in review_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {OUT} ({len(en_rows)} rows)")
    print(f"wrote {REVIEW} ({len(review_rows)} review rows)")
    print("replacement counts:", dict(by_map))
    print("field counts:", {field: dict(counts) for field, counts in by_field.items()})


if __name__ == "__main__":
    main()
