# Overview

This document is a small fixture used by the markdown chunker tests. It has a
handful of sections of varying sizes so the merge and split passes both run.

## Background

Short section.

## Mathematics

The loss is defined below as a display block:

$$
L(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

We minimise $L(\theta)$ with gradient descent, where $\nabla L$ points uphill
and we step in the $-\nabla L$ direction with learning rate $\eta$.

## Details

Paragraph one of the details section provides enough prose to make this the
largest section in the document, which lets the split pass exercise its
paragraph-boundary logic when a small max chunk size is configured.

Paragraph two continues the discussion with more sentences so that there is a
clear blank-line boundary for the splitter to choose. It keeps going for a few
lines to push the section comfortably over a low max threshold.

Paragraph three wraps things up with a final blank-line-delimited block.
