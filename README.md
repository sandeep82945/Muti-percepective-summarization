# Muti-percepective-summarization

### Abstract
> Scientific article summarization poses a challenge because large annotated datasets are unavailable. However, manuscript summaries written by expert peer reviewers, often at the start of their review comments, are not utilized. These multiple summaries of a single manuscript demonstrate to the editor how each reviewer has interpreted the manuscript and reveal the significant differences in perspective among the reviewers. The objective of multiperspective scientific summarization is to create a more comprehensive summary by incorporating multiple related but distinct perspectives of the reviewers rather than being influenced by a single summary. We propose a method to produce abstractive summaries of scientific documents by using those summaries as our gold summaries. This method includes performing extractive summarization to identify the essential parts of the paper by extracting contributing sentences. In the subsequent step, we utilize the extracted pertinent information to condition a transformer-based language model comprising of a single encoder followed by multiple decoders that share weights. The goal is to train the decoder to not only learn from a single reference summary but also to take into account multiple perspectives when generating the summary during the inference stage. Experimental results show that our method achieves the best average ROUGE F1 Score, ROUGE-2 F1 Score, and ROUGE-L F1 Score from the comparing systems.


### Architecture

<img src="https://user-images.githubusercontent.com/43180442/219563160-eed11995-3a4d-49e1-97ca-30b40182647d.png"  width="600" height="300">
