# Sunday - Expo

## The challenges and latest advances in the field of causal AI

The focus of this talk is on "causal AI". They published a paper where some of the conclusions are:

- Causal method do not overfit.
- Do not scale well.
- Very sensitive to initial assumptions.

## How we leverage machine learning and AI to develop life-changing medicines - a case study with COVID-19.

Benevolent AI developed a platform to innovate drug discovery in various areas.

- Knowledge foundations.
- Target identification.
- Precision medicine.
- Molecular design.

These 4 areas are covered in the talks.

### Knowledge Foundations

Rosalind: graph based model + tensor factorization.
Prediction lf clinical trial failure (3 out of 4 cases).
Check the paper (find the link).

### Target Identification

Sia: knowledge graph. Text derived data, ingested data, patient-level data (100s bn of public and commercial data points). The platform is agnostic to the type of disease and can be deployed.

Information extraction: NLP based. NER on diseases, genes, gene family, chemicals, functions. Entity linking: there are many ways a certain entity can be written in text. "Simple hierarchical multi-task neural end-to-end entity linking for biomedical text" in EMNLP 2020.

Aggregating evidence from disparate sources. Joint relationship extraction and inference. This is prediction of novel links in the graph. Representation learning for entities, entity tuples, relationships. "Learning informative representations of biomedical relations with latent variable models" in EMNLP 2020.

Example: someone comes asking for genes whose mutations causes a certain disease, and there is very little or no data. "Constructing large scale biomedical knowledge bases from scratch with rapid annotation of interpretable patterns" in ACL 2019, BioNLP.

### Precision Medicine (Amish)

Precision medicine as an "endotype" discovery problem. They find candidate endotypes and, to find targets, they apply network diffusion methods to the PPI network. They use EHRs to capture (among others) multivariate time series etc. They use ICD and other resources and apply sequential LVMs (Latent Variable Models). If data is available, they associate this to genetic data.

### Work on COVID-19 (Ollie)

Involved in the construction of the knowledge graph. Problem: the data discussed so far come from experiments and trial. For COVID-19, data was not available. This is why it is important to have a domain expert/scientist next to you. They thought of a different way of thinking about the problem. They have a knowledge graph not based on the novel coronavirus, but rather on the mechanisms underlying it, for which data are available. Endocytosis and the sever imflammatory created by the body as a reaction to this infection process were known. The question was "is there a drug working on these two processes simultaneously"?. They used network analysis to investigate this problem (pretty much like Neo4J). Baricitinib inhibits AAK1 and GAK which are involved in the endocytosis process (and, I suppose, also the inflammatory process). The paper was published in February, when the pandemic was raging in Italy.

They have presentations at ML for Molecules and, another one I got it right at ML4Hat ML4H.

## Making boats fly by scaling Reinforcement Learning with Software 2.0 (Quantum Black)

America's Cup: the New Zealand's boat uses special foils that makes it "fly" rather than float. The foils are ~2 meters long, and the boat can travel at 60 mph (or kph?). They have a sailing simulator where each design can be tested. They wanted to automate the evaluation process. Building auto-pilots for boats is difficult (14 inputs for a boat vs only 4 for a F1 car). Sails change shape with the wind ("compliant") which makes the evaluation harder. RL for this task is hard, as the final goal is not clearly defined. Optimization goals are loosely defined. In sailing, if you want to go from A to B, the straight line is almost certainly not the optimal choice, which depends on the wind (which changes with time).

At the beginning the agent was very basic and its performance was poor. They initially used curriculum learning (right?). At the end the agent was outperforming the sailors 66% of the time.

Hydra to handle configurations. Gym to create the environment. rlib, ray, tune libraries. They used spot instances to keep costs reasonable on AWS, which required code to handle interruptions.