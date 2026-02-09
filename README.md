# Visual Token Pruning

## Overview

VLMs have a lot of data heterogeneity. Typically when processing inputs, models use many information-sparse visual tokens and a few information-dense text tokens.

Visual token pruning is the process of selectively removing visual tokens during prefill to reduce computation, memory usage, and even help models focus on more important tokens.

While this area has been explored quite a bit in the past couple years, there are two issues that warrant a re-evaluation of the most important approaches:

- [DatologyAI](https://www.datologyai.com/) has shown that current vision evals are flawed (see [DatBench](https://arxiv.org/pdf/2601.02316)). Namely, many questions can be solved blind! For a field that focuses on removing tokens in the input image and measuring resulting performance, this is very problematic.
- The "best" methods as of today have gotten pretty complex. We went from [pruning visual tokens from a single signal](https://arxiv.org/pdf/2403.06764) to [selecting relevant text tokens, computing cross-modal scores, adaptively sparsifying based on matrix rank, and recycling pruned tokens](https://arxiv.org/pdf/2410.04417).

I think it's worth re-evaluating some core techniques on updated benchmarks to answer the question "where is the best place to prune?".

## Pruning Signals

No matter how complex recent approaches have gotten, at their core they must compute some score among the visual tokens and remove the ones that score the lowest. Ideally this score is some proxy for how relevant the visual token is to the task at hand. There are three main ways to compute such a score:

1. Cross-modal attention scores
In the LLM decoder, we can leverage the attention scores between the text and visual tokens to decide which ones are worth keeping. The hope is that the visual tokens that are most relevant to the text query will have higher attention scores and can be kept.
2. Vision-only attention scores
Following the hypothesis that cross-modal attention scores are actually inaccurate, we can instead look to the vision encoder for a pruning signal. Approaches use vision <-> vision attention scores or [CLS] <-> vision attention scores to decide what tokens to remove.
3. Token embedding similarity scores
If we want our pruning approach to be guided by the text query but don't want to use attention scores, we can instead compute similarity scores between visual token & text token embeddings.

All three of these approaches have some sound theoretical justification but also have serious flaws. Namely, [this work](https://arxiv.org/pdf/2506.22283) cites a few issues. Given these findings, we can summarize our requirements for a pruning approach and see if we can find a better signal:
- **cross-modal**: I believe a pruning approach *must* leverage cross-modal signals. In a single image, the relevant regions will vary depending on the text query. Therefore, the ideal method will be text-aware.
- **no attention scores**: it's fairly clear that cross-modal attention is flawed. beyond just the fact that "attention is not explanation", the long-term decay property of RoPE means that visual tokens in the bottom half of input images tend to have higher attention scores just because they're closer to the text tokens in the input sequence.

These two requirements point heavily towards using token embeddings! But there's a third requirement:

- **"late" processing**: many papers now have shown that the later you prune during prefill, the better performance you will have. you can also prune more aggressively in later layers as information becomes more entangled across tokens.

Using token embeddings similarity scores is about as early in the VLM as you can get while keeping your approach cross-modal. I am proposing a fourth approach that takes into account all three of our requirements: using cross-modal similarity scores from *token representations* in later layers of the LLM. This approach:
- is cross-modal
- does not use attention/RoPE
- allows for later pruning
and also has further empirical justification: [this work](https://openreview.net/pdf?id=chanJGoa7f) states that as they are processed through layers, visual token representations become more text-interpretable. I think it's a reasonable intuition that cross-modal similarity scores from token representations would be a good proxy for this.

## Results

Pending evals, coming soon!
