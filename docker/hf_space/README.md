---
title: Fake News & Misinformation Detector
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: Evidence-grounded 5-class fake-news checker (KIET B.Tech NLP project)
---

Evidence-grounded checker: headline / article / URL -> claims -> evidence (PolitiFact index + DuckDuckGo, Wikipedia,
Google Fact Check) -> NLI stance -> date checks -> one of REAL / FAKE / PARTIALLY TRUE / MISLEADING / UNVERIFIABLE with
confidence, evidence cards and a plain-language reason. Source code and setup: see the project repository.
This file is the Space's README (copy it to the root of the Space repo); deployment steps are in docs/deployment.md.
