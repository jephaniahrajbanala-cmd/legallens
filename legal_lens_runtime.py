"""Lightweight inference utilities for the LegalLens Streamlit app.

The notebook in this project is the training and evaluation pipeline.  It is
not imported here because importing a notebook executes every training cell.
This module contains the document-level analysis layers needed by the web UI:
PDF/text ingestion, extractive summarisation, routed question answering,
outcome detection, timeline extraction, claims, and lexical search.

The functions are deliberately dependency-light so the app can start even
when no trained checkpoint has been generated yet.  The notebook remains the
place to train the optional neural models.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from functools import lru_cache
import io
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable


MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
DATE_RE = [
    rf"\b\d{{1,2}}\s+(?:{MONTHS}),?\s+\d{{4}}\b",
    rf"\b(?:{MONTHS})\s+\d{{1,2}},?\s+\d{{4}}\b",
    r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",
]
SECTION_RE = r"\b[Ss]ections?\s+\d+[A-Za-z]?(?:\s*\(\s*\d+\s*\))?"
ARTICLE_RE = r"\b[Aa]rticles?\s+\d+[A-Za-z]?"
ACT_RE = (
    r"\b(?:[A-Z][A-Za-z&.'\-]+\s+){1,7}"
    r"(?:Act|Code|Ordinance|Rules|Regulations),?\s*\d{4}\b"
)
CITATION_RE = (
    r"(?:\(\d{4}\)\s*\d+\s*[A-Z]{2,6}\s*\d+|"
    r"\d{4}\s*AIR\s*[A-Z]{2,6}\s*\d+|AIR\s*\d{4}\s*[A-Z]{2,6}\s*\d+)\b"
)
AMOUNT_RE = (
    r"\b(?:Rs\.?|INR|₹)\s*[\d,]+(?:\.\d+)?"
    r"(?:\s*(?:lakh|lakhs|crore|crores))?\b"
)

OUTCOME_PATTERNS = [
    ("PARTLY ALLOWED", r"partly\s+allowed"),
    ("ALLOWED", r"(?:appeal|petition|application|writ|suit)\b[^.]{0,60}?\ballowed\b"),
    ("DISMISSED", r"(?:appeal|petition|application|writ|suit)\b[^.]{0,60}?\bdismissed\b"),
    (
        "SET ASIDE",
        r"(?:conviction|sentence|order|judgment|decree|award)\s+"
        r"(?:is|are|was|were)?\s*(?:hereby\s+)?(?:set aside|quashed)",
    ),
    ("REMANDED", r"\bremanded\b"),
    ("ACQUITTED", r"\bacquitted\b"),
    ("UPHELD", r"\b(?:upheld|affirmed)\b"),
    ("MODIFIED", r"\b(?:modified|reduced|enhanced|substituted)\b"),
    ("REJECTED", r"\brejected\b"),
    ("NO RELIEF", r"\b(?:no merit|devoid of merit|no interference)\b"),
]

ABBREVIATIONS = {
    "no", "nos", "sec", "secs", "art", "arts", "v", "vs", "hon", "mr",
    "mrs", "ms", "dr", "j", "jj", "ltd", "pvt", "co", "corp", "inc",
    "etc", "eg", "ie", "govt", "jr", "sr", "st", "rs", "ord", "cl",
    "para", "p", "pp", "sch", "crl", "air", "scr", "scc", "ed", "supra",
    "ibid", "r", "w",
}

MONTH_MAP = {name.lower(): number for number, name in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1
)}
MONTH_MAP.update({
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10,
    "nov": 11, "dec": 12,
})


DEMO_DOCUMENTS = {
    "Asha Sharma v. State of Rajasthan (sample)": """IN THE HIGH COURT OF RAJASTHAN

Asha Sharma                                   ... Petitioner
Versus
State of Rajasthan                            ... Respondent

The petition was filed on 12 January 2022 under Article 226 of the Constitution
of India. The petitioner challenged the order dated 04.03.2021 passed by the
District Magistrate. The matter was heard on 18 July 2023 and judgment was
reserved after the parties completed their submissions.

The petitioner argued that the authority had acted without giving a reasonable
opportunity of hearing. The State contended that the order was supported by the
Rajasthan Land Revenue Act, 1956 and that the petitioner had an alternative
remedy. After considering the record, this Court holds that the authority
failed to follow the required procedure.

The record shows that a notice was issued to the petitioner on 20 February
2021. The notice referred to an inspection report but did not attach the report
or identify the material relied upon by the District Magistrate. The petitioner
requested a copy of the report and sought ten days to file a reply. The request
was declined by an endorsement made on 26 February 2021. The impugned order
was thereafter passed without recording whether the request had been considered.

Learned counsel for the petitioner submitted that the power under Section 90 of
the Rajasthan Land Revenue Act, 1956 has serious civil consequences. It was
argued that an opportunity of hearing must be real and meaningful when an order
affects possession and the use of agricultural land. Counsel also relied on the
principle that reasons are an essential part of a fair decision. The State
replied that the petitioner had been given notice and could pursue a statutory
appeal.

The question before the Court is whether a later statutory remedy cures the
failure to disclose the material relied upon before the original decision. The
notice was not merely defective in form; it prevented the petitioner from
meeting the substance of the allegation. The appellate remedy cannot be an
automatic substitute for the hearing required at the first stage. The authority
must reconsider the matter with an open mind and give reasons for its conclusion.

This order does not express an opinion on the merits of the land dispute. The
District Magistrate shall provide the inspection report, permit a written reply,
and pass a reasoned order. The observations in this judgment are limited to
procedural fairness and shall not prejudice either party at the fresh hearing.

For these reasons, the petition is allowed. The order dated 04.03.2021 is set
aside and the matter is remanded to the District Magistrate for a fresh order
after hearing the petitioner. There shall be no order as to costs.
""",
    "Ravi Kumar v. State (criminal sample)": """IN THE HIGH COURT OF DELHI

Ravi Kumar                                      ... Appellant
Versus
State                                           ... Respondent

The appellant Ravi Kumar was convicted under Section 420 of the Indian Penal
Code, 1860. The appeal was heard on 22 February 2024. The complainant stated
that payment had been made in reliance on a promise which was not fulfilled.
The appellant argued that the dispute was civil in nature and that the trial
court had not correctly assessed the evidence.

The prosecution case was that the complainant transferred money in three
instalments after the appellant represented that a commercial vehicle would be
delivered within thirty days. The vehicle was not delivered, and the complaint
was registered on 14 June 2021. During investigation, the complainant produced
bank statements, copies of messages, and a handwritten receipt. The appellant
admitted receiving the money but stated that the amount was an advance under a
business arrangement that later failed.

The trial court treated the failure to return the money as proof that the
appellant had dishonest intention from the beginning. It also relied on the
testimony of the complainant's brother, although he was not present when the
alleged representation was made. The defence examined two witnesses who stated
that negotiations continued after payment and that the appellant attempted to
arrange another vehicle. Those witnesses were not discussed in the final
reasoning of the trial court.

Section 420 of the Indian Penal Code requires proof of deception and dishonest
inducement at the time the property was delivered. A subsequent breach of a
promise, without more, does not establish the original intention. The bank
record proves payment, but it does not prove the mental element required for
cheating. The Court has considered the authorities cited by both sides and the
rule that a conviction cannot rest on suspicion.

The prosecution also failed to explain why the civil remedy was not pursued
when the parties had exchanged written terms. This is not by itself decisive,
but it reinforces the need for caution before converting a failed commercial
transaction into a criminal conviction. The benefit of reasonable doubt must
go to the appellant. The complainant remains free to pursue any remedy
available in law, subject to limitation and other objections.

The Court finds that the prosecution did not prove the necessary dishonest
intention beyond reasonable doubt. Accordingly, the appeal is allowed. The
conviction and sentence are set aside and the appellant is acquitted. The
    amount of Rs. 50,000 deposited by the appellant shall be refunded.
""",
    "Meera Patel v. Union of India (service sample)": """IN THE HIGH COURT OF BOMBAY

Meera Patel                                      ... Petitioner
Versus
Union of India                                   ... Respondent

The petition was filed on 08 August 2020 challenging the termination of the
petitioner from public service. The petitioner relied on Article 14 and Article
16 of the Constitution of India and submitted that no disciplinary inquiry had
been completed. The matter was heard on 11 November 2022.

The Union of India argued that the appointment was temporary and that the
termination clause permitted immediate action. The Court holds that the order
does not disclose reasons and was passed without observing natural justice.

The petitioner was appointed as a laboratory assistant after a selection
process conducted in 2017. Her appointment letter contained a probation clause
and permitted termination by one month's notice. She completed probation and
worked continuously for more than three years. The termination order dated 28
July 2020 stated only that her services were no longer required. It did not
refer to performance, misconduct, restructuring, or any particular clause.

Before the termination, the petitioner received a memorandum alleging that two
test registers were incomplete. She submitted a written explanation on 14 July
2020 and requested an inspection of the records. No inquiry officer was
appointed and no decision was communicated on her explanation. The respondents
have not produced a file noting that considers the explanation. The sequence
of events therefore matters even though the order is expressed as a neutral
termination simpliciter.

The Court distinguishes between a genuinely administrative discharge during
probation and an order founded on an allegation that carries a stigma. The
authority cannot avoid a fair inquiry merely by using neutral words when the
surrounding record shows that the allegation was the real reason for action.
Article 14 requires non-arbitrariness, and Article 16 protects fairness in
public employment. Contractual power remains subject to constitutional limits
when it is exercised by the State.

The respondents may begin a lawful inquiry, but they must provide the
allegations and supporting material and give the petitioner a reasonable
opportunity to respond. The petitioner shall be treated as having remained in
service only for considering the fresh decision. This Court does not direct
automatic reinstatement with full back wages because the merits have not been
adjudicated.

The petition is partly allowed. The termination order is quashed. The
respondents may conduct a fresh inquiry in accordance with law within three
months. Back wages shall be considered after the inquiry is completed.
""",
    "Karan Singh v. State of Punjab (bail sample)": """IN THE HIGH COURT OF PUNJAB AND HARYANA

Karan Singh                                      ... Petitioner
Versus
State of Punjab                                  ... Respondent

The application for anticipatory bail was heard on 05 May 2024. The petitioner
was accused of offences under Section 420 of the Indian Penal Code, 1860 and
Section 7 of the Prevention of Corruption Act, 1988. The prosecution opposed
the application and submitted that custodial interrogation was necessary.

The first information report was registered on 17 April 2024 after a complaint
by a contractor who alleged that an advance payment had been obtained by a
false promise. The prosecution asserts that the petitioner introduced an
intermediary and received two instalments. The petitioner denies the allegation
and states that the transaction relates to a private supply contract. The case
diary records that the principal documents and bank statements have already
been collected.

The petitioner appeared before the investigating officer on 12 April, 19 April,
and 26 April 2024. He supplied his mobile telephone and answered questions
about the account into which the money was deposited. The prosecution has not
identified a specific recovery that can be made only through custodial
interrogation. The State nevertheless submits that custodial questioning is
needed to identify other persons who may have participated in the transaction.

The power to grant anticipatory bail is discretionary and must be exercised
after balancing the seriousness of the allegation, the possibility of flight,
the risk of influencing witnesses, and the need for a fair investigation. The
Court does not conduct a mini-trial at this stage. At the same time, arrest
cannot be used as a substitute for investigation when the accused has joined
the investigation and the relevant documents are already with the police.

The petitioner shall not leave India without permission, shall share a current
address and telephone number with the investigating officer, and shall appear
whenever a written notice is served. He shall not contact the complainant or
attempt to influence any witness. Any breach of these conditions will permit
the State to seek cancellation of protection. The observations in this order
are provisional and shall not affect the trial or the investigation.

The Court has considered the case diary and the fact that the petitioner joined
the investigation whenever called. No recovery remains to be made from him.
Accordingly, the application is allowed. In the event of arrest, the petitioner
shall be released on bail subject to furnishing a bond of Rs. 100,000.
""",
    "Lakshmi Textiles v. Arjun Traders (commercial sample)": """BEFORE THE COMMERCIAL COURT AT BENGALURU

Lakshmi Textiles Pvt. Ltd.                         ... Plaintiff
Versus
Arjun Traders                                      ... Defendant

The suit was instituted on 19 February 2021 for recovery of Rs. 850,000 under
an unpaid supply invoice. The plaintiff relied on the written contract dated
15 January 2021 and the delivery records. The defendant admitted receipt of
the goods but disputed the quality and claimed a set-off.

The plaintiff supplied 4,000 metres of blended fabric under purchase orders
issued on 15 January 2021 and 02 February 2021. Delivery challans signed by the
defendant's warehouse manager record receipt of the consignments on 20 January
and 10 February 2021. The defendant made a part payment of Rs. 300,000 but did
not pay the balance after the second delivery. A statement of account was sent
on 03 March 2021 and a demand notice followed on 18 March 2021.

The defendant relies on an email dated 24 February 2021 stating that some rolls
were not of the agreed shade. No independent inspection was requested, and the
goods were not returned. The defendant's witnesses accepted in cross-examination
that the fabric was cut and supplied to customers. The Court therefore accepts
that delivery took place and that the defendant retained the benefit of the
contract. A set-off cannot be granted without proof of the alleged loss.

The contractual interest clause is printed in the plaintiff's standard terms,
but the signed purchase orders do not expressly incorporate that clause. The
plaintiff is consequently entitled to reasonable interest under Section 34 of
the Code of Civil Procedure, 1908, rather than the higher contractual rate. The
claim for consequential business loss is not supported by records and is
rejected. The parties shall bear costs in proportion to their success.

The decree shall be drawn accordingly. If payment is made within sixty days,
the defendant may satisfy the decree without further execution proceedings.
This direction does not prevent the plaintiff from seeking execution if the
amount remains unpaid after the period expires.

The Court finds that the defendant did not give timely notice of any defect.
The invoices and delivery records establish the principal debt, but interest
at the contractual rate is not proved. The suit is decreed in part. The
defendant shall pay Rs. 850,000 with simple interest at 6 percent per annum
from the date of filing until payment. The remaining claim is dismissed.
""",
    "Nisha Verma v. State Election Commission (constitutional sample)": """IN THE HIGH COURT OF KERALA

Nisha Verma                                       ... Petitioner
Versus
State Election Commission                         ... Respondent

The writ petition was filed on 01 March 2023 seeking correction of the
electoral roll. The petitioner stated that her name was omitted despite a
valid identity document and a prior entry in the roll. The respondent argued
that the prescribed revision period had expired.

The petitioner submitted an application to the local electoral registration
officer on 16 January 2023. She enclosed her identity card, proof of residence,
and an earlier voter identity card bearing the same address. The application
was acknowledged, but the online portal later displayed the remark “address not
verified”. No field inquiry report was supplied to the petitioner, and the
officer did not provide an opportunity to correct the alleged discrepancy.

The petitioner states that she moved within the same municipal ward in 2021
and that the name of the householder appears differently in two public records.
The respondent does not dispute that the identity document and the prior roll
entry relate to the same person. Its objection is that the annual revision had
closed before the writ petition was filed. The file produced in Court contains
no speaking order and does not identify the statutory provision that prevented
consideration of the application.

The electoral roll is prepared under the Representation of the People Act,
1950 and the Registration of Electors Rules, 1960. The statutory scheme gives
the registration officer a duty to verify information, but that duty must be
performed through a fair procedure. A technical portal status cannot replace a
reasoned decision. The Court is not directing enrollment without verification;
it is directing the competent authority to complete verification and give
reasons.

The Commission shall give the petitioner seven days to produce additional
residence proof, conduct the required inquiry, and pass a speaking order. If
the application is rejected, the order shall mention the appeal or correction
remedy available under the Act. Nothing in this judgment directs inclusion of a
person who does not satisfy the statutory conditions. The relief is confined to
reconsideration by the proper authority.

The Court observes that the right to participate in the statutory process
cannot be defeated by an administrative omission when the applicant produced
the required documents within time. The petition is allowed. The Commission
shall verify the documents and decide the petitioner's application within four
weeks. No order as to costs.
""",
}


def clean_text(text: str | None) -> str:
    text = (text or "").replace("\x00", " ").replace("\u00a0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _protect_abbreviation(match: re.Match[str]) -> str:
    word = match.group(1)
    if len(word) == 1 or word.lower().rstrip(".") in ABBREVIATIONS:
        return word + "\u2024"
    return match.group(0)


def sentence_split(text: str | None) -> list[str]:
    """Split legal prose while protecting common abbreviations and initials."""
    text = clean_text(text)
    if not text:
        return []
    text = re.sub(r"\n+", " ", text)
    protected = re.sub(r"\b([A-Za-z]{1,6})\.", _protect_abbreviation, text)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    sentences = [part.replace("\u2024", ".").strip() for part in parts]
    return [sentence for sentence in sentences if len(sentence.split()) >= 3]


def word_tokens(text: str | None) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*", (text or "").lower())


def read_pdf_bytes(data: bytes) -> str:
    """Extract text from uploaded PDF bytes using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("PDF support requires the 'pypdf' package.") from exc
    reader = PdfReader(io.BytesIO(data))
    return clean_text("\n".join(page.extract_text() or "" for page in reader.pages))


def parse_date(raw: str) -> datetime | None:
    raw = re.sub(r"(?<=[A-Za-z])\.", "", raw.strip())
    match = re.match(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", raw)
    if match and match.group(2).lower() in MONTH_MAP:
        try:
            return datetime(int(match.group(3)), MONTH_MAP[match.group(2).lower()], int(match.group(1)))
        except ValueError:
            return None
    match = re.match(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", raw)
    if match and match.group(1).lower() in MONTH_MAP:
        try:
            return datetime(int(match.group(3)), MONTH_MAP[match.group(1).lower()], int(match.group(2)))
        except ValueError:
            return None
    match = re.match(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})", raw)
    if match:
        day, month, year = map(int, match.groups())
        year = year + 2000 if year < 100 else year
        try:
            return datetime(year, month, day)
        except ValueError:
            return None
    return None


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = re.sub(r"\s+", " ", value).strip(" .,;:-")
        key = re.sub(r"[^a-z0-9]", "", value.lower())
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def regex_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    patterns = [
        ("PROVISION", SECTION_RE),
        ("PROVISION", ARTICLE_RE),
        ("STATUTE", ACT_RE),
        ("CASE_CITATION", CITATION_RE),
        ("DATE", DATE_RE[0]),
        ("DATE", DATE_RE[1]),
        ("DATE", DATE_RE[2]),
        ("AMOUNT", AMOUNT_RE),
    ]
    for label, pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            spans.append((label, match.start(), match.end()))
    for label, pattern in OUTCOME_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            spans.append(("OUTCOME", match.start(), match.end()))
    return spans


def extract_outcome_rules(text: str) -> dict[str, Any]:
    sentences = sentence_split(text)
    candidates: list[tuple[int, int, str, str]] = []
    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        cue = sum(cue_word in lower for cue_word in (
            "accordingly", "therefore", "in the result", "for these reasons",
            "disposed of", "we hold", "ordered", "final order",
        ))
        for label, pattern in OUTCOME_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                candidates.append((cue, index, label, sentence))
    if not candidates:
        return {"outcome": "Not detected", "evidence": "", "confidence": 0.0, "source": "rules"}
    cue, index, label, evidence = max(candidates, key=lambda row: (row[0], row[1]))
    confidence = min(0.99, 0.55 + 0.08 * cue + 0.25 * index / max(1, len(sentences) - 1))
    return {
        "outcome": label,
        "evidence": evidence,
        "confidence": round(confidence, 3),
        "source": "rules",
    }


def extract_timeline(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_cues = {
        "filing": ("filed", "instituted", "registration"),
        "hearing": ("heard", "hearing", "argued", "arguments", "reserved"),
        "judgment": ("judgment", "judgement", "pronounced", "delivered"),
        "decision": ("allowed", "dismissed", "ordered", "disposed", "set aside", "acquitted"),
    }
    for sentence in sentence_split(text):
        lower = sentence.lower()
        event_type = "event"
        for candidate, cues in event_cues.items():
            if any(cue in lower for cue in cues):
                event_type = candidate
                break
        for pattern in DATE_RE:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                raw = match.group(0)
                events.append({
                    "date": raw,
                    "sort": parse_date(raw) or datetime.max,
                    "event_type": event_type,
                    "event": sentence,
                })
    events.sort(key=lambda item: item["sort"])
    for event in events:
        event.pop("sort", None)
    return events


_NAME_STOP = re.compile(
    r"\b(?:Son|Daughter|Wife|Husband|Mother|Father|S/o|D/o|W/o|R/o|Resident|"
    r"Versus|Verses|Vs\.?|v\.|through|Petitioner|Respondent|Appellant|"
    r"Plaintiff|Defendant|Accused|Complainant|Opposite\s+Party|Ors|Anr|"
    r"Advocate|Counsel|Judge|Justice|Coram|Appearance|District|Section|Act)\b",
    re.IGNORECASE,
)
_PARTY_NAME = re.compile(
    r"[A-Z][A-Za-z./'&\-]*(?:\s+(?:of\s+|and\s+|&\s+)?"
    r"[A-Z][A-Za-z./'&\-]*){0,5}"
)


def _clean_name(value: str) -> str:
    value = _NAME_STOP.split(value)[0]
    value = re.sub(r"\s+", " ", value).strip(" .,;:-_=*\n")
    return value if 2 <= len(value) <= 80 else ""


def extract_parties(text: str) -> dict[str, list[str]]:
    roles = ("petitioner", "appellant", "respondent", "accused", "complainant",
             "victim", "deceased", "witness", "judge", "lawyer")
    result = {role: [] for role in roles}
    head = clean_text(text)[:4500]

    caption = re.search(
        r"(?P<left>[A-Z][^\n]{1,100}?)\s+(?:v\.?|vs\.?|versus)\s+"
        r"(?P<right>[A-Z][^\n]{1,100}?)(?:\s+\.\.\.|\n|$)",
        head,
        re.IGNORECASE,
    )
    if caption:
        left = _clean_name(caption.group("left"))
        right = _clean_name(caption.group("right"))
        if left:
            result["petitioner"].append(left)
        if right:
            result["respondent"].append(right)

    role_pattern = re.compile(
        r"\b(?:the\s+)?(?P<role>petitioner|appellant|respondent|accused|"
        r"complainant|prosecutrix|victim|deceased|witness)\s+"
        r"(?:is\s+)?(?P<name>[A-Z][A-Za-z.'\-/]*(?:\s+[A-Z][A-Za-z.'\-/]*){0,4})",
        re.IGNORECASE,
    )
    for match in role_pattern.finditer(text):
        name = _clean_name(match.group("name"))
        role = match.group("role").lower()
        if role == "prosecutrix":
            role = "victim"
        if name and role in result:
            result[role].append(name)

    reverse_role_pattern = re.compile(
        r"(?P<name>[A-Z][A-Za-z.'\-/]*(?:\s+[A-Z][A-Za-z.'\-/]*){0,4})\s*,?\s+"
        r"(?:who\s+is\s+)?the\s+(?P<role>accused|complainant|victim|deceased)",
        re.IGNORECASE,
    )
    for match in reverse_role_pattern.finditer(text):
        name = _clean_name(match.group("name"))
        if name:
            result[match.group("role").lower()].append(name)

    for match in re.finditer(
        r"\b(?:Justice|Judge)\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3})",
        text,
    ):
        name = _clean_name(match.group(1))
        if name:
            result["judge"].append(name)
    for match in re.finditer(
        r"\b(?:Mr|Mrs|Ms|Shri|Smt)\.?\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3})"
        r"\s*,?\s+(?:Advocate|Adv\.?|Counsel|Sr\.?\s+Counsel)",
        text,
        re.IGNORECASE,
    ):
        name = _clean_name(match.group(1))
        if name:
            result["lawyer"].append(name)

    return {role: _unique(values) for role, values in result.items()}


PERSON_QUESTION_RE = re.compile(r"\b(who|whose|whom|which\s+person|which\s+party|name\s+the)\b", re.I)
ROLE_QUESTION_HINTS = (
    (r"\baccused\b|\bconvict\b|\bguilty\b|\bcharged\b", "accused"),
    (r"\bpetitioner\b|\bpetition\b", "petitioner"),
    (r"\bappellant\b|\bappeal\b", "appellant"),
    (r"\bcomplainant\b", "complainant"),
    (r"\bprosecutrix\b|\bvictim\b|\bdeceased\b", "victim"),
    (r"\brespondent\b|opposite\s+party", "respondent"),
    (r"\bplaintiff\b", "plaintiff"),
    (r"\bdefendant\b", "defendant"),
    (r"\bjudge\b|\bjustice\b|\bbench\b", "judge"),
    (r"\bwitness\b|\bdeposed\b", "witness"),
    (r"\badvocate\b|\bcounsel\b|\blawyer\b", "lawyer"),
)


def question_role(question: str) -> str | None:
    if not PERSON_QUESTION_RE.search(question or ""):
        return None
    for pattern, role in ROLE_QUESTION_HINTS:
        if re.search(pattern, question or "", re.IGNORECASE):
            return role
    return "person"


def resolve_role_answer(question: str, text: str) -> dict[str, Any] | None:
    role = question_role(question)
    if not role:
        return None
    parties = extract_parties(text)
    if role in ("petitioner", "appellant"):
        answer = (parties.get(role) or parties.get("petitioner") or [None])[0]
    elif role == "person":
        answer = (parties.get("petitioner") or parties.get("appellant") or
                  parties.get("accused") or parties.get("respondent") or [None])[0]
    elif role == "victim":
        answer = (parties.get("victim") or parties.get("deceased") or [None])[0]
    else:
        answer = (parties.get(role) or [None])[0]
    if not answer:
        return None
    return {
        "answer": answer,
        "source": "party/role resolver",
        "evidence": "case caption or role phrase",
        "score": 0.95,
    }


def question_target(question: str) -> str:
    question = question or ""
    if re.search(r"\boutcome\b|\bresult\b|\bverdict\b|\bdispos|\bdecid|what happened|final order", question, re.I):
        return "outcome"
    if re.search(r"\bsection|\barticle|\bprovision|\bsub-?section", question, re.I):
        return "provision"
    if re.search(r"\bcitation\b|\bcited\b|\bprecedent\b|case law", question, re.I):
        return "citation"
    if re.search(r"\bstatute|\bact\b|\blaw\b|\brules\b", question, re.I):
        return "statute"
    if re.search(r"how much|\bamount\b|\bfine\b|compensation|damages|costs", question, re.I):
        return "amount"
    if re.search(r"\bwhen\b|what date|which date|on what date", question, re.I):
        return "date"
    return "general"


FACT_LABELS = {
    "provision": ("PROVISION",),
    "statute": ("STATUTE",),
    "citation": ("CASE_CITATION",),
    "amount": ("AMOUNT",),
    "date": ("DATE",),
}


def fact_values(target: str, text: str, limit: int = 4) -> list[str]:
    wanted = FACT_LABELS.get(target, ())
    values = []
    for label, start, end in regex_spans(text):
        if label in wanted:
            values.append(text[start:end])
            if len(_unique(values)) >= limit:
                break
    return _unique(values)[:limit]


def _sentence_score(sentence: str, index: int, total: int, frequencies: Counter[str]) -> float:
    tokens = word_tokens(sentence)
    if not tokens:
        return -1.0
    lexical = sum(frequencies[token] for token in set(tokens)) / math.sqrt(len(tokens))
    legal_bonus = sum(token in {"court", "judgment", "judgement", "order", "petition", "appeal", "held", "allowed", "dismissed"} for token in tokens)
    position_bonus = 1.0 - index / max(1, total)
    return lexical + 1.5 * legal_bonus + 0.8 * position_bonus


def extractive_summary(text: str, max_sentences: int = 8, word_budget: int = 220) -> str:
    sentences = sentence_split(text)
    if not sentences:
        return ""
    frequencies = Counter(word_tokens(text))
    ranked = sorted(
        range(len(sentences)),
        key=lambda index: _sentence_score(sentences[index], index, len(sentences), frequencies),
        reverse=True,
    )
    selected: list[int] = []
    used = 0
    for index in ranked:
        size = len(sentences[index].split())
        if selected and used + size > word_budget:
            continue
        selected.append(index)
        used += size
        if len(selected) >= max_sentences:
            break
    return " ".join(sentences[index] for index in sorted(selected))


def search_judgment(query: str, text: str, limit: int = 3) -> list[dict[str, Any]]:
    # Keep the user-facing search intentionally concise.
    limit = max(1, min(3, int(limit)))
    sentences = sentence_split(text)
    query_tokens = set(word_tokens(query))
    if not sentences or not query_tokens:
        return []
    query_phrase = " ".join(word_tokens(query))
    results = []
    for sentence in sentences:
        sentence_tokens = set(word_tokens(sentence))
        overlap = len(query_tokens & sentence_tokens)
        lexical = overlap / math.sqrt(max(1, len(query_tokens) * len(sentence_tokens)))
        phrase_bonus = 0.25 if query_phrase in " ".join(word_tokens(sentence)) else 0.0
        score = lexical + phrase_bonus
        if score > 0:
            results.append({"score": round(score, 4), "sentence": sentence})
    return sorted(results, key=lambda row: row["score"], reverse=True)[:limit]


def build_document_passages(text: str, max_sentences: int = 5, stride: int = 3) -> list[dict[str, Any]]:
    """Create overlapping passages for lightweight retrieval when model artifacts are absent."""
    sentences = sentence_split(text)
    passages = []
    for start in range(0, len(sentences), max(1, stride)):
        chunk = sentences[start:start + max_sentences]
        if not chunk:
            continue
        passages.append({"passage_id": len(passages), "text": " ".join(chunk)})
        if start + max_sentences >= len(sentences):
            break
    return passages


def retrieve_document_passages(query: str, text: str, limit: int = 3) -> list[dict[str, Any]]:
    """Retrieve grounded judgment passages with lexical scoring."""
    query_tokens = set(word_tokens(query))
    if not query_tokens:
        return []
    results = []
    for passage in build_document_passages(text):
        passage_tokens = set(word_tokens(passage["text"]))
        overlap = len(query_tokens & passage_tokens)
        score = overlap / math.sqrt(max(1, len(query_tokens) * len(passage_tokens)))
        if score > 0:
            results.append({**passage, "score": round(score, 4)})
    return sorted(results, key=lambda row: row["score"], reverse=True)[:max(1, min(3, int(limit)))]


_LEGAL_BACKGROUND = [
    {
        "source": "Indian Penal Code, 1860 (historical reference)",
        "text": (
            "Section 420 of the Indian Penal Code, 1860 concerned cheating and "
            "dishonestly inducing delivery of property. It applied where a person "
            "cheated another and thereby dishonestly induced delivery of property, "
            "or the making, altering, or destroying of a valuable security. The "
            "offence was punishable with imprisonment up to seven years and fine. "
            "The Indian Penal Code was replaced by the Bharatiya Nyaya Sanhita, "
            "2023; the current provision should be checked against the applicable "
            "law and date of the alleged conduct."
        ),
        "references": ("section 420", "ipc", "indian penal code"),
    },
    {
        "source": "Indian Penal Code, 1860 (historical reference)",
        "text": "Section 302 of the Indian Penal Code, 1860 prescribed the punishment for murder, subject to the elements and exceptions stated in the Code.",
        "references": ("section 302", "ipc", "indian penal code"),
    },
    {
        "source": "Constitution of India",
        "text": "Article 14 of the Constitution of India guarantees equality before the law and equal protection of the laws within the territory of India.",
        "references": ("article 14", "constitution"),
    },
    {
        "source": "Constitution of India",
        "text": "Article 21 of the Constitution of India provides that no person shall be deprived of life or personal liberty except according to procedure established by law.",
        "references": ("article 21", "constitution"),
    },
]


def _knowledge_index_candidates(artifact_dir: str | os.PathLike[str] | None = None) -> list[Path]:
    requested = Path(artifact_dir or os.environ.get("LEGALLENS_ARTIFACTS", "legallens_artifacts"))
    project_root = Path(__file__).resolve().parent
    candidates = [
        requested / "legal_knowledge.json",
        requested / "abstractive" / "legal_knowledge.json",
        project_root / "legal_knowledge.json",
        project_root / "legallens_artifacts" / "legal_knowledge.json",
    ]
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result


def legal_knowledge_status(artifact_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    for path in _knowledge_index_candidates(artifact_dir):
        if path.is_file():
            return {"available": True, "path": str(path), "message": "Statute knowledge index found."}
    return {
        "available": False,
        "path": str(_knowledge_index_candidates(artifact_dir)[0]),
        "message": "Notebook statute index not found; built-in legal references are available.",
    }


@lru_cache(maxsize=2)
def _load_legal_knowledge(path: str) -> list[dict[str, Any]]:
    import json

    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return [
            {
                "source": str(row.get("source") or "Indian legal acts corpus"),
                "text": clean_text(row.get("text") or ""),
                "references": tuple(str(x).lower() for x in row.get("references", [])),
            }
            for row in data
            if clean_text(row.get("text") or "")
        ]
    except (OSError, ValueError, TypeError):
        return []


def _legal_knowledge_entries(artifact_dir: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    entries = list(_LEGAL_BACKGROUND)
    status = legal_knowledge_status(artifact_dir)
    if status["available"]:
        entries = _load_legal_knowledge(status["path"]) + entries
    return entries


def _is_background_legal_question(question: str) -> bool:
    definition = re.search(
        r"\b(?:what\s+is|what\s+does|what\s+do|define|explain|meaning\s+of|tell\s+me\s+about|describe)\b",
        question or "",
        re.IGNORECASE,
    )
    reference = re.search(
        r"\b(?:section|sec\.?|article|act|code|ordinance|rule|regulation)\s+\d+[A-Za-z]?\b",
        question or "",
        re.IGNORECASE,
    )
    return bool(definition and reference)


def legal_background_answer(
    question: str,
    artifact_dir: str | os.PathLike[str] | None = None,
    limit: int = 3,
) -> dict[str, Any] | None:
    """Answer definition questions from the notebook statute index or references."""
    if not _is_background_legal_question(question):
        return None
    query_tokens = set(word_tokens(question))
    question_lower = (question or "").lower()
    has_specific_source = bool(re.search(
        r"\b(?:ipc|indian penal code|constitution|crpc|cpc|motor vehicles|"
        r"companies|contract|evidence|transfer of property|income tax|customs)\b",
        question_lower,
    ))
    candidates = []
    for entry in _legal_knowledge_entries(artifact_dir):
        entry_tokens = set(word_tokens(entry["text"]))
        overlap = len(query_tokens & entry_tokens) / math.sqrt(max(1, len(query_tokens) * len(entry_tokens)))
        reference_bonus = max(
            (1.5 if ref in (question or "").lower() else 0.0 for ref in entry.get("references", ())),
            default=0.0,
        )
        curated = entry.get("source", "").endswith("(historical reference)") or entry.get("source") == "Constitution of India"
        # The acts index contains tables where a section number can appear as
        # a passing reference. Prefer the concise curated explanation for a
        # bare question such as "What is Section 420?"; use the index when the
        # user names a specific Act/source.
        curated_bonus = 2.0 if curated and not has_specific_source else 0.0
        score = overlap + reference_bonus + curated_bonus
        if score > 0:
            candidates.append({"score": round(score, 4), "source": entry["source"], "text": entry["text"]})
    candidates.sort(key=lambda row: row["score"], reverse=True)
    if not candidates or candidates[0]["score"] < 1.0:
        return {
            "answer": "",
            "evidence": "",
            "score": 0.0,
            "source": "legal background knowledge base: no matching provision",
            "candidates": candidates[:limit],
        }
    best = candidates[0]
    return {
        "answer": best["text"],
        "evidence": best["text"],
        "score": best["score"],
        "source": f"legal background knowledge base ({best['source']})",
        "candidates": candidates[:limit],
    }


def answer_question(
    question: str,
    text: str,
    limit: int = 8,
    artifact_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    question = (question or "").strip()
    text = clean_text(text)
    if not question or not text:
        return {"answer": "", "evidence": "", "score": 0.0, "source": "empty", "candidates": []}

    role_answer = resolve_role_answer(question, text)
    if role_answer:
        matches = search_judgment(question, text, limit)
        return {**role_answer, "candidates": matches}

    background = legal_background_answer(question, artifact_dir=artifact_dir)
    if background is not None:
        return background

    target = question_target(question)
    if target == "outcome":
        outcome = extract_outcome_rules(text)
        return {
            "answer": outcome["outcome"],
            "evidence": outcome["evidence"],
            "score": outcome["confidence"],
            "source": "outcome rules",
            "candidates": [],
        }

    if target in FACT_LABELS:
        values = fact_values(target, text)
        if values:
            evidence = next((s for s in sentence_split(text) if any(v in s for v in values)), "")
            return {
                "answer": "; ".join(values),
                "evidence": evidence,
                "score": 0.85,
                "source": f"regex legal-span layer ({target})",
                "candidates": [],
            }

    passages = retrieve_document_passages(question, text, limit=3)
    if passages:
        best_passage = passages[0]
        passage_hits = search_judgment(question, best_passage["text"], limit=1)
        best_sentence = passage_hits[0]["sentence"] if passage_hits else best_passage["text"]
        return {
            "answer": best_sentence,
            "evidence": best_passage["text"],
            "score": best_passage["score"],
            "source": "RAG: retrieved document passage",
            "candidates": passages,
        }
    candidates = search_judgment(question, text, limit)
    if not candidates:
        return {"answer": "", "evidence": "", "score": 0.0, "source": "no matching sentence", "candidates": []}
    best = candidates[0]
    return {
        "answer": best["sentence"],
        "evidence": best["sentence"],
        "score": best["score"],
        "source": "lexical sentence retrieval",
        "candidates": candidates,
    }


REPORT_VERBS = ("contend", "argu", "submit", "urge", "assert", "plead", "state", "sought", "prayed")
ROLE_CUES = {
    "PETITIONER": ("appellant", "petitioner", "plaintiff"),
    "RESPONDENT": ("respondent", "defendant", "state of", "union of india"),
    "COURT": ("this court", "the court", "we are of the view", "we hold", "it is held", "held that"),
}


def extract_claims(text: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, sentence in enumerate(sentence_split(text)):
        lower = sentence.lower()
        role = "OTHER"
        if any(cue in lower for cue in ROLE_CUES["COURT"]):
            role = "COURT"
        elif any(cue in lower for cue in ROLE_CUES["PETITIONER"]) and any(v in lower for v in REPORT_VERBS):
            role = "PETITIONER"
        elif any(cue in lower for cue in ROLE_CUES["RESPONDENT"]) and any(v in lower for v in REPORT_VERBS):
            role = "RESPONDENT"
        if role != "OTHER":
            groups.setdefault(role, []).append({"sentence_index": index, "claim": sentence})
    return groups


def _artifact_root(artifact_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve either the standard artifact root or a directly added checkpoint.

    The Colab ZIP can be extracted as either ``legallens_artifacts/abstractive``
    or simply ``abstractive`` in the project root. Supporting both layouts
    avoids requiring a manual move after downloading the checkpoint.
    """
    requested = Path(artifact_dir or os.environ.get("LEGALLENS_ARTIFACTS", "legallens_artifacts"))
    project_root = Path(__file__).resolve().parent
    candidates = [requested]
    if requested != requested.parent:
        candidates.append(requested.parent)
    candidates.extend((project_root, project_root / "legallens_artifacts"))
    seen: set[Path] = set()
    for root in candidates:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        if (root / "abstractive" / "config.json").is_file():
            return root
    return requested


def abstractive_checkpoint_status(artifact_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Check whether the notebook's saved abstractive checkpoint is usable."""
    checkpoint = _artifact_root(artifact_dir) / "abstractive"
    required = checkpoint / "config.json"
    if not required.exists():
        return {
            "available": False,
            "path": str(checkpoint),
            "message": "No abstractive checkpoint found. Run the notebook's abstractive training cell first.",
        }
    return {"available": True, "path": str(checkpoint), "message": "Checkpoint found."}


@lru_cache(maxsize=2)
def _load_abstractive_components(checkpoint: str) -> tuple[Any, Any, Any]:
    """Load the notebook checkpoint once per process; return model, tokenizer, device."""
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, EncoderDecoderModel
        from transformers.utils import logging as hf_logging
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Abstractive inference requires torch and transformers. "
            "Install the notebook dependencies and try again."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    previous_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        except Exception as seq2seq_error:
            # Backward compatibility for the earlier custom EncoderDecoderModel
            # checkpoint produced by the original notebook.
            try:
                model = EncoderDecoderModel.from_pretrained(checkpoint)
            except Exception:
                raise RuntimeError(
                    "The abstractive checkpoint is not a supported seq2seq model: "
                    f"{type(seq2seq_error).__name__}: {seq2seq_error}"
                ) from seq2seq_error
    finally:
        hf_logging.set_verbosity(previous_verbosity)

    # Safetensors stores tied BART embeddings once as `model.shared.weight`.
    # Transformers reports the tied aliases as missing during load, although
    # they are restored by tie_weights(). Verify the shared tensor and re-tie it
    # explicitly so a bad/incomplete checkpoint cannot pass silently.
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    if getattr(model.config, "model_type", "") == "bart":
        shared = getattr(getattr(model, "model", None), "shared", None)
        encoder_embed = getattr(getattr(model.model, "encoder", None), "embed_tokens", None)
        decoder_embed = getattr(getattr(model.model, "decoder", None), "embed_tokens", None)
        lm_head = getattr(model, "lm_head", None)
        if shared is None or encoder_embed is None or decoder_embed is None or lm_head is None:
            raise RuntimeError("BART checkpoint is missing its shared embedding/language head.")
        if not (
            shared.weight.data_ptr() == encoder_embed.weight.data_ptr()
            == decoder_embed.weight.data_ptr() == lm_head.weight.data_ptr()
        ):
            raise RuntimeError("BART checkpoint embeddings are not correctly tied after loading.")
        model.generation_config.forced_bos_token_id = tokenizer.bos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _model_source_limit(model: Any, default: int = 1024) -> int:
    config = getattr(model, "config", None)
    encoder_config = getattr(config, "encoder", None)
    limit = getattr(encoder_config, "max_position_embeddings", None)
    limit = limit or getattr(config, "max_position_embeddings", None)
    return min(default, int(limit)) if limit else default


def abstractive_summary(
    text: str,
    artifact_dir: str | os.PathLike[str] | None = None,
    max_length: int = 160,
) -> dict[str, Any]:
    """Generate with the notebook decoder, with an explicit diagnostic on failure."""
    status = abstractive_checkpoint_status(artifact_dir)
    if not status["available"]:
        return {"summary": "", "available": False, "message": status["message"], "source": "unavailable"}
    try:
        import torch

        model, tokenizer, device = _load_abstractive_components(status["path"])
        source = clean_text(text)
        if len(source) > 100000:
            source = source[:50000] + " " + source[-50000:]
        encoded = tokenizer(
            source,
            truncation=True,
            max_length=_model_source_limit(model),
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items() if key != "token_type_ids"}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_length=max_length,
                min_length=min(24, max_length),
                num_beams=2,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        summary = clean_text(tokenizer.decode(generated[0], skip_special_tokens=True))
        if not summary:
            raise RuntimeError("The checkpoint generated an empty summary.")
        return {
            "summary": summary,
            "available": True,
            "message": f"Loaded notebook checkpoint on {device}.",
            "source": "notebook abstractive checkpoint",
        }
    except Exception as exc:
        return {
            "summary": "",
            "available": False,
            "message": f"Could not load or run the abstractive checkpoint: {type(exc).__name__}: {exc}",
            "source": "error",
        }


def analyze_document(
    text: str,
    summary_method: str = "extractive",
    artifact_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    text = clean_text(text)
    if summary_method not in {"extractive", "abstractive"}:
        raise ValueError("summary_method must be 'extractive' or 'abstractive'")
    if summary_method == "abstractive":
        generated = abstractive_summary(text, artifact_dir=artifact_dir)
        summary = generated["summary"] or extractive_summary(text)
        summary_note = generated["message"] if not generated["summary"] else generated["message"]
        summary_source = generated["source"] if generated["summary"] else "extractive fallback"
    else:
        summary = extractive_summary(text)
        summary_note = ""
        summary_source = "local extractive analysis"
    return {
        "summary": summary,
        "summary_note": summary_note,
        "summary_source": summary_source,
        "outcome": extract_outcome_rules(text),
        "timeline": extract_timeline(text),
        "claims": extract_claims(text),
        "sentence_count": len(sentence_split(text)),
        "word_count": len(word_tokens(text)),
    }


def runtime_capabilities(artifact_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Report whether notebook-generated artifacts are present without loading them."""
    from pathlib import Path

    root = _artifact_root(artifact_dir)
    expected = ("dapt", "salience", "encoder", "qa", "outcome", "role", "abstractive")
    present = [name for name in expected if (root / name).exists()]
    return {
        "artifact_dir": str(root),
        "available": bool(present),
        "components": present,
        "abstractive": abstractive_checkpoint_status(root),
    }
