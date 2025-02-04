# Optimal Couplings for Distortion-Free Text Watermarking
The official code of **Optimal Couplings for Distortion-Free Text Watermarking** 

## Abstract
Large-language models (LLMs) are now able to produce text that is indistinguishable from human-generated content.
This has fueled the development of watermarks that imprint a 'signal' in LLM-generated text with minimal perturbation of an LLM's output.
This paper provides an analysis of text watermarking in a one-shot setting.
Through the lens of hypothesis testing with side information, we formulate and analyze the fundamental trade-off between watermark detection power and distortion in generated textual quality.
We argue that a key component in watermark design is  generating a coupling between the side information shared with the watermark detector and a random partition of the LLM vocabulary.
Our analysis identifies the optimal coupling and randomization strategy under the worst-case LLM next-token distribution that satisfies a min-entropy constraint. 
We provide a closed-form expression of the resulting detection rate under the proposed scheme and quantify the cost in a max-min sense.
Finally, we numerically compare the proposed scheme with the theoretical optimum.
