# Lightweight Discursive Stance Analyzer

A lightweight, open-source NLP pipeline that segments argumentative text into sentence-level
propositions and classifies each proposition as **supporting**, **opposing**, or **neutral**
toward a user-defined premise.

The project explores whether zero-shot Natural Language Inference can support qualitative
decision-making by mapping the arguments contained in long-form text.

## Research question

Can a zero-shot NLI model identify the stance of individual statements toward a policy claim
without task-specific training?

## How it works

The pipeline follows four stages:

1. **Text cleaning** : removes URLs, excessive whitespace, and non-informative symbols.
2. **Discourse segmentation** : uses spaCy to split text into sentence-level propositions.
3. **Zero-shot stance classification** : compares each proposition with a target premise using
   `facebook/bart-large-mnli`.
4. **Evaluation** : measures accuracy, precision, recall, F1-score, and confusion matrices.

The final prompt uses the following hypothesis structure:

```text
This statement {candidate label} the following claim: [target premise].
```

Candidate labels are mapped as follows:

| Model label | Project label |
|---|---|
| supports the premise | `support` |
| opposes the premise | `oppose` |
| is neutral toward the premise | `neutral` |

## Tech stack

- Python
- Google Colab
- spaCy (`en_core_web_sm`)
- Hugging Face Transformers
- `facebook/bart-large-mnli`
- pandas
- scikit-learn
- matplotlib and seaborn
- Jupyter Notebook

## Repository structure

```text
discursive-stance-analyzer/
├── notebooks/
│   └── stance_analyzer.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Run in Google Colab

1. Open the notebook:

   [`notebooks/stance_analyzer.ipynb`](notebooks/stance_analyzer.ipynb)

2. Click **Open in Colab** from the notebook page, or open Colab and upload the notebook.
3. Run the cells in order.
4. The first model-loading cell downloads `facebook/bart-large-mnli`.
5. The notebook can run on CPU, although GPU accelerates classification.

## Datasets

The project uses two evaluation settings:

### 1. Manually constructed UBI evaluation set

A balanced set of 30 English statements related to the premise:

```text
Universal basic income is beneficial
```

The set contains 10 support, 10 oppose, and 10 neutral statements.

### 2. External climate-change benchmark

The notebook also evaluates the pipeline on the `stance_climate` subset of the
[TweetEval benchmark](https://huggingface.co/datasets/cardiffnlp/tweet_eval).

The original labels are mapped as follows:

| TweetEval label | Project label |
|---|---|
| `favor` | `support` |
| `against` | `oppose` |
| `none` | `neutral` |

## Results

### UBI evaluation set

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Support | 0.67 | 1.00 | 0.80 |
| Oppose | 0.83 | 1.00 | 0.91 |
| Neutral | 1.00 | 0.30 | 0.46 |

**Accuracy:** 0.767

The model correctly identified all support and oppose statements. Its main weakness was
neutral detection: descriptive or contextual statements were often interpreted as support.

### Climate-change tweet benchmark

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Support | 0.48 | 0.55 | 0.51 |
| Oppose | 0.43 | 0.91 | 0.59 |
| Neutral | 0.80 | 0.20 | 0.32 |

**Accuracy:** 0.490

Performance decreased substantially on real tweets. This reflects domain shift, informal
language, sarcasm, indirect argumentation, and the difficulty of detecting neutral stance
in short social-media posts.

## Prompt sensitivity

The project tested several zero-shot prompt formulations.

| Version | Approach | Accuracy on UBI set |
|---|---|---:|
| Version 1 | Short labels: `support`, `oppose`, `neutral` | 0.667 |
| Version 2 | Explicit relationship between statement and claim | **0.800** |
| Version 3 | Longer explicit neutral definition | 0.667 |
| Version 4 | More distinct concise definitions | 0.267 |

Version 2 was selected as the best-performing configuration. The comparison shows that
zero-shot classification is sensitive to prompt and label wording, even when the underlying
model remains unchanged.

## Key findings

- Zero-shot NLI can identify clear supporting and opposing arguments without task-specific training.
- Neutral stance is the most difficult class because descriptive statements can be mistaken for weak support or opposition.
- Prompt wording materially affects classification behavior.
- Performance does not automatically transfer from clean argumentative text to noisy social-media text.
- A lightweight zero-shot system is useful for exploratory decision support, but it is not yet reliable enough for high-stakes automated analysis.

## Limitations

- The UBI evaluation set is small and manually constructed.
- The climate evaluation uses a small sample of tweets and is affected by class imbalance after cleaning.
- Sentence-level segmentation may split complex arguments into incomplete propositions.
- Zero-shot confidence scores are not calibrated probabilities.
- The model was not fine-tuned on stance-detection data.
- Sarcasm, irony, and implicit stance remain difficult for NLI-based zero-shot classification.

## Future work

- Expand the evaluation set with independently labeled examples.
- Evaluate on the full TweetEval climate test split.
- Test additional NLI models and prompt templates.
- Add confidence-thresholding and abstention for uncertain predictions.
- Fine-tune a smaller transformer model on labeled stance data.
- Develop the notebook functions into a FastAPI service.

## References

- Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2020).  
  *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.*  
  Findings of the Association for Computational Linguistics: EMNLP 2020.  
  https://aclanthology.org/2020.findings-emnlp.148/

- Hugging Face. *facebook/bart-large-mnli.*  
  https://huggingface.co/facebook/bart-large-mnli

- spaCy. *English pipeline: en_core_web_sm.*  
  https://spacy.io/models/en
