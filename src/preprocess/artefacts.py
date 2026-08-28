"""Artefact / leakage removal (master doc §11.1 step 4) and the artefact-only leakage classifier.

Principle: strip **publisher / format boiler-plate** that identifies *where an article came from* (datelines, photo
credits, embed captions, bylines, title tags, outlet suffixes, URL debris) — never content words.  Bare source names
inside prose ("… told Reuters", "Breitbart News reported", "announced via Twitter") are content and stay (see
``data/README.md``).

Two pattern lists, both applied by ``strip_frame``:

``PATTERNS`` — article **text** (WELFake and ISOT); also the features of ``artefact_only_accuracy``:

* Reuters datelines ``"WASHINGTON (Reuters) - "``, ``"(Reuters) - "``, any remaining ``"(Reuters)"``
  (raw WELFake: 21,061 real / 0 fake docs)
* URLs (4,899 docs), e-mail addresses, Twitter handles (10,111), ``pic.twitter.com/…`` links (4,423) — **and the debris
  WELFake's own tokenisation left behind**: ``"pic. twitter."`` / ``"https: ."`` stubs (295 docs, 293 of them real),
  the embedded-tweet time-stamp ``"(@handle) May 20, 2017"`` (6,957) and empty parentheses ``"( )"`` (753)
* fake-site trailers ``"Featured image via/by/: …"`` (8,614 fake / 3 real), ``"Read more …"`` (3,146),
  ``"21st Century Wire says …"`` (748), ``"SUPPORT 21WIRE … @21WIRE.TV"`` (823 fake / 0 real), ``"Continue this story at …"`` (350)
* photo / image credits ``"Photo: Chip Somodevilla via Getty Images."``, ``"(Photo by AFP)"``, ``"Image credit: WTVF"``,
  ``"Name/Getty Images"`` (``Getty``: 4,227 fake vs 41 real docs) and source lines ``"Via: Breitbart News"`` (3,578 fake / 0 real),
  ``"Source: Raw Story"`` / ``"For entire story: …"`` / ``"{snip}"`` (≈ 900 fake / 14 real)
* embed captions ``"Watch it below:"``, ``"Watch the video here:"``, ``"WATCH:"`` (2,539 fake / 111 real),
  ``"Here's the video via YouTube."`` (1,108 fake / 10 real)
* bylines ``"Follow Trent Baker on Twitter"`` (2,101 real / 308 fake — WELFake's Breitbart share),
  ``"Charlie Nash is a reporter for Breitbart Tech."`` (399 real / 105 fake), ``"Email tips and suggestions to …"``
* bracketed format tags ``[VIDEO]`` / ``(IMAGES)`` / ``[TWEETS]`` / ``[DETAILS]`` … inside the text (236 docs)

``TITLE_PATTERNS`` — **titles** only:

* the format tags above (9,461 fake WELFake titles vs 4 real; ISOT 5,472 fake / 0 real)
* trailing outlet suffixes ``" - The New York Times"`` (6,223 real), ``" - Breitbart"`` (2,338 real), ``" - TruthFeed"``,
  ``" - America's Finest News Source"`` … — a **closed list** built from the raw title-suffix counts (8,584 real / 648 fake
  titles matched); only publisher / author names are listed, never editorial tails such as ``" - Prepare To Laugh"``

All counts are raw-WELFake documents matched (72,134 rows, measured 2026-08-28 with this file).  ``artefact_features``
counts the ``PATTERNS`` per document so that a classifier trained **only** on them quantifies how much of the in-domain
accuracy is leakage (≈ 0.996 on raw ISOT).  What deliberately survives: prose uses of *via*, *video*, *Twitter*,
*below*, *Reuters*, *Breitbart*, and the ``21WIRE`` name inside sentences.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_CAP = r"[A-Z][\w'’&-]*(?:\.(?!\w))?"           # one capitalised token (names, outlets, initials "D.")
_CAPS = rf"{_CAP}(?: {_CAP}){{0,4}}"           # up to five of them

# --- wire-service datelines (ISOT real class) -----------------------------------------------------------------------
# Order matters: datelines first, then the bare token.
DATELINE_RE = re.compile(
    r"^\s*(?:[A-Z][A-Za-z0-9.'’/&-]*(?:[ ,]+[A-Z][A-Za-z0-9.'’/&-]*){0,6}\s*)?\((?:Reuters)\)\s*[-–—:]*\s*",
)
REUTERS_RE = re.compile(r"\(\s*Reuters\s*\)")

# --- fake-site trailers ---------------------------------------------------------------------------------------------
# "Featured image via screengrab", "Featured Image: Win McNamee/Getty Images", "Featured Photo by Scott Olson/Getty Images"
FEATURED_IMAGE_RE = re.compile(
    r"Featured? (?:image|photo)s?(?:\s*(?::|\bvia\b|\bby\b|\bfrom\b|\bcredits?:?|[-–]))?[^\n]*$", re.IGNORECASE | re.MULTILINE
)
READ_MORE_RE = re.compile(r"Read more:?[^\n]*$", re.IGNORECASE | re.MULTILINE)
WIRE_RE = re.compile(r"21st Century Wire says[^\n]*", re.IGNORECASE)
# "SUPPORT 21WIRE  SUBSCRIBE & BECOME A MEMBER @21WIRE.TV", "SUPPORT OUR WORK BY SUBSCRIBING …"
SUPPORT_RE = re.compile(r"SUPPORT (?:21WIRE|OUR WORK)[^\n]*|@?\s?21WIRE\.TV")
# "Continue this story at Washington Times" (21WIRE syndication stub)
CONTINUE_RE = re.compile(rf"Continue (?:this|the) (?:story|report|article)(?: (?:at|on|here)(?: {_CAPS})?)?")

# --- URLs / handles and the debris left by WELFake's own tokenisation ----------------------------------------------
URL_RE = re.compile(r"(?:https?:\s*//\s*|www\.)\S+", re.IGNORECASE)  # WELFake has "https:// twitter.com/..."
# "pic.twitter.com/GMco1PkJiL" (often glued: "2017pic.twitter.com/…"), and the pre-mangled "pic. twitter." / "pic. twitter. com" stubs
PIC_RE = re.compile(r"pic\.\s?twitter(?:\.\s?com)?(?:\s?/\s?\S+)?\.?", re.IGNORECASE)
# "https: ." / "https:" with nothing usable after it (the scheme survived, the address did not)
HTTP_STUB_RE = re.compile(r"\bhttps?:\s*\.?(?![\w/])", re.IGNORECASE)
# embedded-tweet time-stamp "(@BenSasse) September 28, 2017" / "( ) May 20, 2017" / "(@wikileaks) 4:03 AM – 23 Oct 2016"
TWEET_STAMP_RE = re.compile(
    rf"[-–—]?\s*\(\s*@?\w*\s*\)\s*(?:{_MONTH}\.?\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}:\d{{2}}\s*[AP]M\s*[-–—]\s*\d{{1,2}}\s+\w{{3}}\.?\s+\d{{4}}),?"
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.\s?[\w.-]+\b")   # "ctomlinson@breitbart. com" (WELFake tokenisation)
HANDLE_RE = re.compile(r"(?<![\w])@\w{1,30}")
EMPTY_PARENS_RE = re.compile(r"\(\s*\)")      # "Sen. Ted Cruz ( )" — party tag removed upstream by the dataset; runs last

# --- photo / image / source credits ----------------------------------------------------------------------------------
_AGENCY = (
    r"Getty(?: [Ii]mages)?|Flickr(?: Creative Commons)?|Screen ?grab|Screen ?shot|Screen ?capture|screencap|Twitter|YouTube|"
    r"Facebook|Instagram|Reddit|Wikimedia(?: Commons)?|Wikipedia|Creative Commons|AP|AFP|EPA|Reuters|Bloomberg|Shutterstock|"
    r"Pool|file|video|C-SPAN|CNN|MSNBC|Fox(?: News)?|NBC|ABC|CBS|CNBC|PBS|BBC|RNC|DNC|"
    rf"{_CAP}(?: {_CAP}){{0,2}} (?:News|Times|Post|Journal|Tribune|Herald|Daily|Gazette|Beacon)(?![a-z])|[A-Z]{{3,5}}(?![a-z])"
)
# "Photo: Chip Somodevilla via Getty images." / "Photo by Joe Raedle/Getty Images" / "Image credit: video screen capture via Fox 4"
PHOTO_CREDIT_RE = re.compile(
    rf"\(?\b(?:Official {_CAP}(?: {_CAP}){{0,3}} )?(?:Photo(?:graph)?s?|Images?|Pictures?|Screenshots?|Screen ?grabs?|Screen ?captures?)"
    rf"(?: [Cc]redits?| [Cc]ourtesy(?: of)?| [Ss]ource)?\s*(?::|\bby\b|\bvia\b|\bfrom\b)\s*[^\n.!?]{{0,80}}?"
    rf"(?:{_AGENCY})(?: [Ii]mages)?(?: for {_CAPS})?\s*\.?\)?"
    r"|\bPhoto by\s*$",
    re.MULTILINE,
)
# "Image credit: Nashville Public Schools", "(image credit: AP)" — the word "credit" makes the agency list unnecessary
IMAGE_CREDIT_RE = re.compile(
    rf"\(?\b(?:[Pp]hoto|[Ii]mage|[Pp]icture)s? [Cc]redits?[ \t]*(?::|;|\bfrom\b|\bvia\b|\bby\b)?[ \t]*"
    rf"(?:(?:{_CAP}|via|on|of|by|from|cc|and|video|screen|capture|screenshot|screengrab|\d[\w.]*)[ \t/,&|-]*){{1,10}}\.?\)?"
)
# "(Official White House Photo by Amanda Lucidon)", "(U.S. Marine Corps photo by Staff Sgt. Robert Storm)", "(File photo by AFP)"
PAREN_PHOTO_RE = re.compile(r"\([^()\n]{0,40}\b[Pp]hoto(?:graph)?s? (?:by|credit|courtesy)[^()\n]{0,80}\)")
# "Alex Wong/Getty Images", "via Getty" — credit remnants without a keyword
GETTY_RE = re.compile(rf"(?:{_CAP}(?: {_CAP}){{0,3}}\s*(?:/|via|-)\s*)?Getty(?: [Ii]mages?)?(?: for {_CAPS})?\.?")
# "Via: Breitbart News" / "Via Free Beacon:" source lines (3,274 fake / 0 real raw WELFake docs)
VIA_SOURCE_RE = re.compile(rf"(?i:via):[ \t]*(?:[A-Za-z][\w'’&.-]*[ \t]?){{1,4}}|\bVia {_CAPS}:")
# "Source: Raw Story" (649 fake / 8 real processed WELFake docs), "For entire story: Breitbart News", "Click HERE for entire story.", "{snip}"
SOURCE_LINE_RE = re.compile(rf"\bSources?:[ \t]*{_CAPS}\.?|For entire story:?[ \t]*{_CAPS}|(?:Go to {_CAPS} |Click HERE )?for entire story\.?|\{{snip\}}")

# --- embed captions ---------------------------------------------------------------------------------------------------
# "Watch it below:", "Watch the video here:", "LISTEN HERE:", "You can watch the segment below.", "WATCH:"
WATCH_RE = re.compile(
    r"(?<![A-Za-z])(?:You can |Please )?(?:Watch|Listen(?: to)?)(?:\s[^.\n:!?]{0,50}?)?\s?(?:below|here)(?: LIVE)?\s*[:.]?"
    r"|(?<![A-Za-z])Watch(?: (?:the |this |it |the full |the whole )?(?:video|clip|interview|segment|footage|exchange|moment))?:",
    re.IGNORECASE,
)
# "Here s the video via YouTube.", "Here's the clip of his comments, courtesy of Now This:", "Here is a screen grab of …"
HERES_VIDEO_RE = re.compile(
    rf"\bHere(?:'s|’s| s| is) (?:the|a|another|his|her|their) (?:full |second |first |entire |whole )?"
    rf"(?:video|clip|tweet|tweets|screenshot|screencap|screen ?shot|screen ?grab|screen ?capture|snapshot|footage|audio|segment|interview|exchange)"
    rf"(?: (?:of|from) [^:.\n]{{0,40}})?(?:,? (?:via|courtesy of|courtesty of) {_CAPS})?\s*[:.]?"
)

# --- bylines ---------------------------------------------------------------------------------------------------------
# "Follow Trent Baker on Twitter", "You can follow him on Twitter or like his page at Facebook.",
# "Follow Chris Tomlinson on Twitter at or email at …", "You can follow Ben Kew on Facebook, on Twitter at , or email him at …"
FOLLOW_RE = re.compile(
    rf"(?:(?:You can |Please )?[Ff]ollow (?:him|her|them|me|us|our \w+|@?{_CAP}(?: {_CAP}){{0,3}}) on (?:Facebook,? (?:and |or )?on )?Twitter|Follow on Twitter)"
    r"(?:\s*(?:at|:)?\s*@[\w.]+)?(?:\s+at\b)?(?:,?\s+(?:or|and)\s+(?:like|add|friend|follow|join|email)[^.\n]{0,60})?"
    r"(?: for [^.\n]{0,60})?\.?"
)
# "Charlie Nash is a reporter for Breitbart Tech." (+ the rest of that sentence)
BYLINE_RE = re.compile(
    rf"{_CAP}(?: {_CAP}){{1,3}} is (?:a|an|the) (?:[\w-]+ ){{0,3}}"
    r"(?:reporter|writer|editor|columnist|correspondent|contributor|journalist|producer)(?: and [\w ]{0,20})? "
    rf"(?:for|at|with|of) {_CAPS}[^.\n]{{0,80}}\.?"
)
EMAIL_TIPS_RE = re.compile(r"\[ ?Email (?:him|her) ?\]|\bEmail (?:tips|suggestions|him|her|us|them)\b[^.\n]{0,60}\.?")

# --- bracketed format tags (titles and text) ---------------------------------------------------------------------------
_TAG = (
    r"(?:NSFW |GRAPHIC |LIVE )?(?:VIDEOS?|TWEETS?|IMAGES?|PHOTOS?|PICS?|PICTURES?|AUDIO|SCREENSHOTS?|SCREENGRABS?|WATCH|LISTEN|"
    r"PODCAST|INFOGRAPHIC|DETAILS|LIVE ?BLOG|LIVE ?STREAM|LIVESTREAM|LIVE)"
)
# "[VIDEO]", "(IMAGES)", "[VIDEO/TWEETS]", "[Video]", "(Video, 5.26 mins)", "[Video Documentary]"
FORMAT_TAG_RE = re.compile(
    rf"[\[(]\s*{_TAG}(?:\s*[/&+,]\s*{_TAG})*(?:,[^\])}}]{{0,30}}| Documentary| w/ Transcript| and (?:Transcript|Photos?|Proof))?\s*[\])}}]",
    re.IGNORECASE,
)

# --- trailing outlet suffix in titles ---------------------------------------------------------------------------------
# Closed list from the raw-WELFake title-suffix counts (2026-08-28): every entry is a publisher or author name.
OUTLETS = [
    "The New York Times", "Breitbart(?: News)?", "America['’]s Finest News Source", "TruthFeed", "Russia News Now",
    "New Eastern Outlook", "EndingFed News Network", "Daily Wire", "RedFlag News", "The Vineyard of the Saker",
    "RT Arabic", "RT", "Collective Evolution", "Russia & India Report", "Paul Craig Roberts", "Wikileaks",
    "Conservative Daily Post", "GomerBlog", "USAPoliticsNow", "The Federalist Papers", "New Century Times",
    "Morning News USA", "White House", "American Lookout", "OffGuardian", "The Knowledge You Crave", "MagaFeed",
    "ClickHole", "RIA", "Motivate3\\.com", "Jim Willie", "ICH", "New Earth Media", "Tyler Durden", "RealClearPolitics",
    "Jason Ditz", "CounterCurrents\\.org", "Top Right News", "Fort Russ", "Upside Down Media", "American Herald Tribune",
    "Bill Holter", "The Boston Globe", "The Washington Post", "Fox News", "CNN", "NPR", "POLITICO", "The Onion",
    "Infowars", "The Huffington Post", "HuffPost", "Sputnik(?: News)?", "The Daily Caller", "The Daily Beast",
    "Zero Hedge", "Global Research", "Activist Post", "Natural News", "The Duran", "Consortiumnews", "Reuters",
    "BBC News", "The Guardian", "The Hill", "Vox", "Salon", "Slate", "21st Century Wire",
]
OUTLET_SUFFIX_RE = re.compile(r"\s+[-–—|]\s*(?:" + "|".join(OUTLETS) + r")\s*$", re.IGNORECASE)

MULTISPACE_RE = re.compile(r"[ \t]{2,}")
DANGLING_RE = re.compile(r"\s+[-–—|:]\s*$")   # a separator left at the very end of a title after stripping

# (name, regex) pairs — the strip order for TEXT; also the feature list of the leakage classifier
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("dateline", DATELINE_RE),
    ("reuters", REUTERS_RE),
    ("featured_image", FEATURED_IMAGE_RE),
    ("read_more", READ_MORE_RE),
    ("wire", WIRE_RE),
    ("support_21wire", SUPPORT_RE),
    ("continue_story", CONTINUE_RE),
    ("url", URL_RE),
    ("pic_twitter", PIC_RE),
    ("http_stub", HTTP_STUB_RE),
    ("tweet_stamp", TWEET_STAMP_RE),     # before the handle regex so "(@name) May 20, 2017" is still intact
    ("email", EMAIL_RE),
    ("handle", HANDLE_RE),
    ("paren_photo", PAREN_PHOTO_RE),
    ("photo_credit", PHOTO_CREDIT_RE),
    ("image_credit", IMAGE_CREDIT_RE),
    ("getty", GETTY_RE),
    ("via_source", VIA_SOURCE_RE),
    ("source_line", SOURCE_LINE_RE),
    ("watch_below", WATCH_RE),
    ("heres_video", HERES_VIDEO_RE),
    ("follow_twitter", FOLLOW_RE),
    ("byline", BYLINE_RE),
    ("email_tips", EMAIL_TIPS_RE),
    ("format_tag", FORMAT_TAG_RE),
    ("empty_parens", EMPTY_PARENS_RE),   # last: also removes the "( )" that the captions above leave behind
]

# strip order for TITLES
TITLE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("format_tag", FORMAT_TAG_RE),
    ("outlet_suffix", OUTLET_SUFFIX_RE),
]


def _strip(text: str, patterns: list[tuple[str, re.Pattern]], hits: dict[str, int] | None = None) -> str:
    for name, rx in patterns:
        text, n = rx.subn(" ", text)
        if n and hits is not None:
            hits[name] += 1
    return MULTISPACE_RE.sub(" ", text).strip()


def strip_artefacts(text: str) -> str:
    """Remove every text artefact pattern from one document."""
    if not text:
        return text
    return _strip(text, PATTERNS)


def strip_title(title: str) -> str:
    """Remove format tags and the trailing outlet suffix from one title."""
    if not title:
        return title
    return DANGLING_RE.sub("", _strip(title, TITLE_PATTERNS)).strip()


def strip_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the text patterns to ``text`` and the title patterns to ``title`` (one pass, counting matched docs).

    Returns (frame, {pattern: n_docs_matched}); title patterns are keyed ``title_<name>``.
    """
    hits = {name: 0 for name, _ in PATTERNS}
    hits.update({f"title_{name}": 0 for name, _ in TITLE_PATTERNS})
    out = df.copy()
    out["text"] = pd.Series([_strip(t, PATTERNS, hits) if t else t for t in out["text"].tolist()],
                            index=out.index, dtype="string")
    if "title" in out.columns:
        title_hits = {name: 0 for name, _ in TITLE_PATTERNS}
        titles = [DANGLING_RE.sub("", _strip(t, TITLE_PATTERNS, title_hits)).strip() if t else t
                  for t in out["title"].fillna("").astype(str).tolist()]
        out["title"] = pd.Series(titles, index=out.index, dtype="string")
        hits.update({f"title_{k}": v for k, v in title_hits.items()})
    return out, hits


def artefact_features(texts: pd.Series) -> np.ndarray:
    """Per-document counts of each text artefact pattern (shape: n_docs × n_patterns)."""
    cols = [texts.str.count(rx.pattern, flags=rx.flags).fillna(0).to_numpy() for _, rx in PATTERNS]
    return np.column_stack(cols).astype(float)


def artefact_only_accuracy(texts: pd.Series, labels: pd.Series, seed: int = 42) -> dict:
    """Train a logistic regression on the artefact counts only (5-fold CV accuracy)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X = artefact_features(texts)
    y = labels.to_numpy()
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    per_pattern = {name: float((X[:, i] > 0).mean()) for i, (name, _) in enumerate(PATTERNS)}
    return {"cv_accuracy": float(acc.mean()), "cv_std": float(acc.std()), "doc_frac_with_pattern": per_pattern}
