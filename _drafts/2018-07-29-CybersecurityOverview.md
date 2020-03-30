---
layout: post
title: "A Search Tool for Cybersecurity Strategy Documents"
excerpt: "I recently built a text search and analysis tool as part of the United Nations' Unite Ideas initiative."
categories: [NLP, Web]
comments: true
---

Earlier this year, I came across an interesting initative from the United Nations called [Unite Ideas](https://uniteideas.spigit.com/main/Page/Home). From their website:
> This is a global community for individuals, teams, and organizations to join forces with the United Nations to develop & apply technology for social impact.

They pose various projects or "challenges" in a competition format with a review phase. [One of these projects](https://uniteideas.spigit.com/cybersecuritynlp/Page/Home) stood out to me as it involved building a text search and analysis tool over cybersecurity strategy documents (unstructured PDFs) published by UN members, so I decided to participate. You can view the prototype of my submission [here](https://llefebure.github.io/cybersecurity-nlp/).

At a high level, the project involved text extraction from PDFs, text cleaning, sentence tokenization, sentence classification, and a UI for displaying and making searchable all of that information. To avoid hosting costs, I decided on having a single page application (Vue.js) that consumes static JSON files produced by an analysis pipeline (Python). I did not have any real experience with modern Javascript frameworks before this project, so that portion was a good learning experience. However, the additional time it took to ramp up there came at the expense of spending time on improving the analysis through better text cleaning/extraction and more advanced NLP. Check out the [code](https://github.com/llefebure/cybersecurity-nlp) on GitHub for more details.

Overall, it was an enjoyable and educational project, and I very much look forward to participating in future Unite Ideas challenges!
