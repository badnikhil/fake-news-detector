"""Unit tests for the residual publisher / format artefact patterns added to `src/preprocess/artefacts.py`
(MSE1 follow-up: the top LR coefficients must not be `via`, `video`, `getty`, `breitbart`, `follow`, `pic`, `https` …).

Every case is a shortened real WELFake / ISOT snippet; content words must survive, boiler-plate must go."""
from __future__ import annotations

import pandas as pd
import pytest

from src.preprocess.artefacts import PATTERNS, TITLE_PATTERNS, strip_artefacts, strip_frame, strip_title


def test_pattern_lists_are_named_and_unique():
    names = [n for n, _ in PATTERNS]
    assert len(names) == len(set(names))
    assert {"dateline", "url", "handle", "photo_credit", "follow_twitter", "format_tag"} <= set(names)
    assert [n for n, _ in TITLE_PATTERNS] == ["format_tag", "outlet_suffix"]


# ---------------------------------------------------------------------------------------------------
# titles
@pytest.mark.parametrize(
    "title, expected",
    [
        ("This NRA Commercial Is So Stupid It Will Melt Your Brain (VIDEO)", "This NRA Commercial Is So Stupid It Will Melt Your Brain"),
        ("Trump Supporter Was FALSELY Charged With [VIDEO]", "Trump Supporter Was FALSELY Charged With"),
        ("(VIDEO) For the Love of Winston Smith", "For the Love of Winston Smith"),
        ("Trump Just Did This [VIDEO/TWEETS]", "Trump Just Did This"),
        ("Mike Lindell Sending 60,000 Pillows to Houston [Video]", "Mike Lindell Sending 60,000 Pillows to Houston"),
        ("Global Opinion Of Donald Trump (IMAGES)", "Global Opinion Of Donald Trump"),
        ("Some Announcement [DETAILS]", "Some Announcement"),
        ("MSNBC's Joan Walsh: Ivanka Trump 'Applauds Cruelty' - Breitbart", "MSNBC's Joan Walsh: Ivanka Trump 'Applauds Cruelty'"),
        ("Let's Say Obamacare Is Repealed. What Then? - The New York Times", "Let's Say Obamacare Is Repealed. What Then?"),
        ("Area Man Confused - America’s Finest News Source", "Area Man Confused"),
        ("Breaking: Something Happened – TruthFeed", "Breaking: Something Happened"),
    ],
)
def test_title_tags_and_outlet_suffixes_are_stripped(title, expected):
    assert strip_title(title) == expected


@pytest.mark.parametrize(
    "title",
    [
        "Smirnoff Is BRILLIANTLY Trolling Donald Trump Over Russia - Prepare To Laugh",   # editorial tail, not an outlet
        "Trump Signals He May Be Done With Kushner - For The Most Trump Reason Ever",
        "House Votes on Health Care (Again)",                                              # not a format tag
        "Senator (R-TX) Speaks",
    ],
)
def test_content_titles_are_untouched(title):
    assert strip_title(title) == title


# ---------------------------------------------------------------------------------------------------
# text: photo / image / source credits
@pytest.mark.parametrize(
    "text, gone, kept",
    [
        ("that would be nice to see, too.Photo: Chip Somodevilla via Getty images.", ["Getty", "via", "Photo"], ["nice to see"]),
        ("replace Reid.Featured image via Ethan Miller/Getty Images. ", ["Featured", "Getty", "via"], ["replace Reid"]),
        ("is doing something.Featured Image: Screenshot", ["Featured", "Screenshot"], ["doing something"]),
        ("meddling.Photo by Joe Raedle/Getty Images.", ["Getty", "Photo by"], ["meddling"]),
        ("October 26, 2016. (Photo by AFP) Libyan forces have launched", ["Photo by AFP"], ["Libyan forces"]),
        ("Sept. 23, 2013. (Official White House Photo by Amanda Lucidon) But we can", ["Official White House Photo"], ["But we can"]),
        ("published by WTVF here.Image credit: WTVF", ["Image credit"], ["published by"]),
        ("not to fill out the survey.Via: Breitbart News", ["Via:", "Breitbart"], ["fill out the survey"]),
        ("Conservatives must be furious.Featured Image: Sara D. Davis/Getty Images", ["Getty", "Davis"], ["furious"]),
    ],
)
def test_credits_are_stripped(text, gone, kept):
    out = strip_artefacts(text)
    for g in gone:
        assert g not in out, (g, out)
    for k in kept:
        assert k in out, (k, out)


def test_via_and_photo_in_prose_are_kept():
    assert strip_artefacts("Trump announced via Twitter that he would fire Comey.") == "Trump announced via Twitter that he would fire Comey."
    assert strip_artefacts("He gained a following via Facebook and is unconventional.") == "He gained a following via Facebook and is unconventional."
    assert "photos:" in strip_artefacts('Mr. Wilson pointed to the photos: "They didn\'t start with much."')


# text: embed captions and bylines
@pytest.mark.parametrize(
    "text, gone, kept",
    [
        ("His hypocrisy is mind-blowing.Watch it below:", ["Watch it below"], ["mind-blowing"]),
        ("Muslims will not like the answer to that.Watch the video below:", ["Watch the video below"], ["the answer to that"]),
        ("filming the event.WATCH HERE:WATCH these awesome students", ["WATCH HERE"], ["these awesome students"]),
        ("You can watch the segment below. The end.", ["watch the segment below"], ["The end"]),
        ("Here s the video via YouTube.We have freedom of the press", ["Here s the video", "YouTube"], ["freedom of the press"]),
        ("she applauds cruelty. Follow Trent Baker on Twitter", ["Follow", "Twitter"], ["applauds cruelty"]),
        ("Charlie Nash is a reporter for Breitbart Tech. You can follow him on Twitter @MrNashington or like his page at Facebook.",
         ["reporter for Breitbart", "follow him on Twitter", "Facebook"], []),
        ("Follow Chris Tomlinson on Twitter at or email at ctomlinson@breitbart. com", ["Follow", "email"], []),
        ("Continue this story at Washington TimesREAD MORE TRUMP NEWS AT: 21st Century Wire 2016 FilesSUPPORT 21WIRE SUBSCRIBE & BECOME A MEMBER@ 21WIRE.TV",
         ["Continue this story", "READ MORE", "21WIRE", "SUBSCRIBE"], []),
    ],
)
def test_captions_and_bylines_are_stripped(text, gone, kept):
    out = strip_artefacts(text)
    for g in gone:
        assert g not in out, (g, out)
    for k in kept:
        assert k in out, (k, out)


def test_see_below_prose_is_kept():
    assert strip_artefacts("as you can see below, the numbers do not add up.") == "as you can see below, the numbers do not add up."


# text: URL / tweet debris
def test_tweet_embed_debris_is_stripped():
    out = strip_artefacts("What the hell, man? https://t.co/fsRl25AD12 pic.twitter.com/GMco1PkJiL  CNN (@CNN) December 6, 2016")
    assert out == "What the hell, man? CNN"
    out = strip_artefacts("please consider donating https:  .  THANKS A LOT pic. twitter.   —   Peter Tatchell (@PeterTatchell) April 16, 2017,  When confronted")
    assert "https" not in out and "pic" not in out and "2017" not in out and "@" not in out
    assert "THANKS A LOT" in out and "When confronted" in out
    assert strip_artefacts("Sen. Ted Cruz ( ) ripped into Elon Musk") == "Sen. Ted Cruz ripped into Elon Musk"
    out = strip_artefacts("many others. \n— WikiLeaks (@wikileaks) 4:03 AM – 23 Oct 2016 \nHe was a mentor")
    assert "2016" not in out and "@" not in out and "He was a mentor" in out
    # 'HTTP' as a word in prose is not a URL stub
    assert strip_artefacts("too slow to process HTTP requests and therefore") == "too slow to process HTTP requests and therefore"


def test_format_tag_in_text_is_stripped():
    assert strip_artefacts("Trump said it again [VIDEO] and then left.") == "Trump said it again and then left."


# frame-level
def test_strip_frame_handles_title_and_text_and_reports_counts():
    df = pd.DataFrame(
        {
            "title": ["Title One (VIDEO)", "Plain title - Breitbart", None],
            "text": ["Body.Featured image via screengrab", "WASHINGTON (Reuters) - Body two. Follow Jeff Poor on Twitter", "Body three"],
            "label": ["fake", "real", "real"],
        }
    )
    out, counts = strip_frame(df)
    assert list(out["title"]) == ["Title One", "Plain title", ""]
    assert list(out["text"]) == ["Body.", "Body two.", "Body three"]
    assert counts["featured_image"] == 1 and counts["dateline"] == 1 and counts["follow_twitter"] == 1
    assert counts["title_format_tag"] == 1 and counts["title_outlet_suffix"] == 1


@pytest.mark.parametrize(
    "text, gone, kept",
    [
        ("denounced these acts of violence. Source: Raw Story", ["Source:", "Raw Story"], ["acts of violence"]),
        ("wiping their naked butts with the flag.For entire story: Breitbart News", ["entire story", "Breitbart"], ["with the flag"]),
        ("lean on the kitchen table. Click HERE for entire story.At this point", ["entire story", "HERE"], ["kitchen table", "At this point"]),
        ("on Canadian soil.\n{snip}\nIt's tempting to look past", ["{snip}"], ["Canadian soil", "tempting"]),
    ],
)
def test_source_lines_are_stripped(text, gone, kept):
    out = strip_artefacts(text)
    for g in gone:
        assert g not in out, (g, out)
    for k in kept:
        assert k in out, (k, out)


def test_source_in_prose_is_kept():
    t = "A French diplomatic source said Paris wanted a deal, sources said."
    assert strip_artefacts(t) == t
