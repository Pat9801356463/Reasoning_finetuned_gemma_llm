Competition Link : https://www.kaggle.com/competitions/google-tunix-hackathon
Problem Statement
Large Language Models (LLMs) can be prompted to explain their outputs; however, these explanations are often post-hoc, loosely correlated with the actual internal computations that produced the answer, and sometimes incorrect or irrelevant. Such explanations fail to provide true transparency and undermine trust in model reasoning.

For explanations to be meaningful, reasoning traces must emerge directly from the transformer’s internal mechanics, rather than being generated retrospectively as free-form text. Achieving this requires a principled study of the transformer architecture itself—particularly attention routing, linear transformations, and residual stream accumulation, which together govern how information flows and decisions are formed.

This work is deeply inspired by foundational research in mechanistic interpretability and transformer theory, including:

Elhage et al. (2021) – A Mathematical Framework for Transformer Circuits

Elhage et al. (2022) – Induction Heads

Vaswani et al. (2017) – Attention Is All You Need

Conceptual Framework: Inacted

We introduce Inacted, a structured framework for extracting reasoning traces directly from transformer internals.
The name Inacted is derived from three core components:

Intention → Action → Lead

Together, these components form a causally grounded representation of reasoning within a transformer layer.

Before proceeding further, we formally define each component and its associated feature vector..
Intention Feature Vector
Intention Feature Vector

Objective: Capture where and how a head chooses to attend.

Derived from attention weights 𝐴(𝑙,ℎ) at layer 𝑙 and head ℎ

𝑓intent(l,h) = [∥Δ(l,h)∥2,  Entropy(𝐴(l,h)),  MeanAttnDistance(𝐴(l,h))]
These features are learned via downstream training after proper feature analysis.

They are designed to answer:

How strong was the contribution of the head?

Was the attention focused or diffuse?

Was the attention local or long-range?

Action Feature Vector
Objective: Characterize what computation the head performs.

Derived from the output projection 𝑊𝑂, value projection 𝑊𝑉 and the MLP block:
𝑊𝑂,𝑊𝑉 and MLP:
𝑓action(l,h) = [∥𝑊𝑂𝑊𝑉∥𝐹,  𝜎1,  rank,  ∥Δ(l,h)∥2]
Where:
𝜎1 denotes the largest singular value
Rank captures representational complexity
These features are trained to identify:
Whether the head performs copying, routing, filtering, or transformation
The nature and strength of the operation applied to the residual stream

Lead Feature Vector
Objective: Determine whether a head actually influenced the final output.

Derived from the unembedding matrix 𝑊𝑈 and residual contribution:
𝑓lead(l,h) =[⟨𝑊𝑈(𝑦),Δ(l,h)⟩]
This directly measures alignment between a head’s contribution and the final predicted token, answering:
Did this head meaningfully affect the model’s answer?

Reasoning Trace : Concatenate over heads
fℓ=⨁​[fintent(l,h)​,faction(l,h)​,flead(l,h)]
h

Understanding these math work is important for LoRA parameters determination.
All this maths has been taken in account to determine LoRA Parmeters
One-Liner Descriptions for All Trainable Parameters:
Attention Layers:
q_proj: Queries - Determines WHAT to focus on in attention

k_proj: Keys - Creates context vectors for WHAT'S available

v_proj: Values - Extracts information from WHAT'S important

o_proj: Output - Routes processed information to WHERE it goes

MLP/FFN Layers:
gate_proj: Gating - DECIDES which information to process

up_proj: Expansion - EXPANDS features for complex computations

down_proj: Compression - COMPRESSES features back to output dimension

Embedding & Output:
embed_tokens: Embeddings - Converts TOKENS to vector representations

lm_head: Output head - Maps final features to VOCABULARY probabilities

norm/ln: Normalization - STABILIZES activations across layers

For IAL Reasoning:
q_proj → INTENTION (what to focus on)

v_proj → ACTION (what to extract)

gate_proj → DECISION (what to process)

down_proj → LEAD (how to format output)
