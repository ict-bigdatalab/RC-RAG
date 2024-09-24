# RC-RAG
Implementation code for the paper "Controlling Risk of Retrieval-augmented Generation: A Counterfactual Prompting Framework", accepted by EMNLP Findings 2024.

## Abstract

Retrieval-augmented generation (RAG) has emerged as a popular solution to mitigate the hallucination issues of large language models. However, existing studies on RAG seldom address the issue of predictive uncertainty, i.e., how likely it is that a RAG model's prediction is incorrect, resulting in uncontrollable risks in real-world applications. In this work, we emphasize the importance of risk control, ensuring that RAG models proactively refuse to answer questions with low confidence. Our research identifies two critical latent factors affecting RAG's confidence in its predictions: the quality of the retrieved results and the manner in which these results are utilized. To guide RAG models in assessing their own confidence based on these two latent factors, we develop a counterfactual prompting framework that induces the models to alter these factors and analyzes the effect on their answers. We also introduce a benchmarking procedure to collect answers with the option to abstain, facilitating a series of experiments. For evaluation, we introduce several risk-related metrics and the experimental results demonstrate the effectiveness of our approach. 

<p align="center">
  <picture>
    <img src="./fig/method.jpeg" width = "700">
  </picture>
</p>

