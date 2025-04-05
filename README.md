# human-action-sequence

Machine learning methodology study and related code to find meaningful actions from human action sequence data.
Our first research data is OECD PIAAC Log Data.

# Human Action Sequence Analysis using NLP and Neural Networks

This repository contains the code and data preprocessing pipelines used in the study:

> **“Discovering Action Insights from Large-Scale Assessment Log Data Using Machine Learning”**  
> Under Submission

We propose a novel framework that combines **Word2Vec**, **Doc2Vec**, and **neural networks** to extract meaningful human actions from large-scale problem-solving logs, using data from the OECD’s **PIAAC** digital problem-solving assessment.


---

## 📊 Dataset

- Dataset: [OECD PIAAC Log Files](https://www.oecd.org/en/data/datasets/piaac-1st-cycle-database.html)
- Problem Sets Used:
  - `Party Invitation`: 1,388 participants, 35 actions
  - `Club Membership`: 1,345 participants, 25 actions
- We focused on complete and valid interaction sequences, stratified by performance score groups (e.g., 0 vs. 3)

> **Note**: Due to licensing, raw log data must be downloaded manually from the OECD website. This repository does not include the full PIAAC dataset.

---

## 🧠 Methodology Summary

1. **Feature Extraction**
   - Word2Vec (skip-gram) to embed individual actions
   - Doc2Vec (PV-DBOW) to embed full action sequences

2. **Feature Verification**
   - Clustering using Doc2Vec vectors
   - Silhouette score for cluster validation
   - Neural Network classification (F1 score up to 0.962)

3. **Case Comparisons**
   - Full sequences vs. meaningful actions only vs. actions excluding meaningful ones

---

## License & Contact

Please contact hyyang@kisti.re.kr for license and usage information."


## Citation

Yun, M., Jeon, M., & Yang, H. (2025). Discovering Action Insights from Large-Scale Assessment Log Data Using Machine Learning. *Under review at Scientific Reports*.




