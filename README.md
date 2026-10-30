# Mushroom Check

A website where you describe a gilled mushroom and a machine learning model estimates whether it is poisonous. When the mushroom doesn't resemble anything in its training data, it answers "Can't tell" rather than guessing.

**Live site:** https://dat158oblig2.onrender.com/

**Authors:** Oddne Bjørnør Horg

> **Never eat a wild mushroom based on this project.** It is a course project trained on 23 North American species.

## Key result

Rows in this dataset are variations of just 23 species, so a random train/test split puts nearly identical rows on both sides. That is why models reach about 100% here, and why that number says little about a mushroom the model hasn't seen.

The file has no species column, but linking rows that differ in one feature recovers exactly 23 groups, each with a single label. These match the 23 species in the dataset documentation. The project therefore evaluates on whole held-out species.

| Model | Random split accuracy | Unseen species accuracy | Unseen poisonous caught | Unseen log loss |
|---|---:|---:|---:|---:|
| Majority baseline | 64.9% | 34.4% | 0.0% | 9.06 |
| Naive Bayes | 95.7% | 49.5% | 16.4% | 3.68 |
| Logistic regression | 100% | 39.2% | 23.3% | 2.56 |
| Decision tree | 100% | 77.7% | 81.0% | 3.07 |
| **Random forest (chosen)** | 100% | 76.5% | 64.9% | **0.62** |
| Gradient boosting | 100% | 68.0% | 59.6% | 1.13 |

Figures are from leave-one-species-out validation on the 18 development species. The random forest was chosen for its log loss, because the decision rule below relies on its probabilities.

**A simpler rule beat every model above on unseen species.** A one-feature rule learned from the training species alone (`scripts/analysis.py`) always picks odor and reaches 98.0% accuracy, so the complex models are memorising species rather than generalising. On its own it calls 5.6% of unseen poisonous mushrooms edible, so it can't be trusted for an edible answer. The final system therefore uses it only to raise warnings, as described below. It was added after the first final test, so that test is no longer independent. The report covers this in more detail.

## How a verdict is made

Calling a poisonous mushroom edible is far worse than the opposite mistake, so the model can't answer "edible" on its own. All four of these must hold:

- The learned odor rule raises no warning. An odor linked to poison, or one the rule has never seen, is a warning. "Not sure" isn't.
- The model's poison risk is at most 2%.
- The input differs from some training mushroom in at most 1 of the features given.
- At least 10 features are filled in.

Otherwise the answer is "Likely poisonous" if the odor rule warns or the risk is 50% or more, and "Can't tell" if not. A safe odor never leads to an edible answer by itself. These limits were tuned on the development species only: the rule that gave the most edible answers on known species, while keeping the false-edible rate at or below 1% on unseen species at each level of hidden input.

To handle partly filled forms, training adds copies of the data with random features hidden.

Results for the full system with each of the 23 species held out in turn (up to 150 rows per species):

| Share of features left blank | Poisonous called edible | Poisonous flagged | Can't tell |
|---:|---:|---:|---:|
| 0% | 0.0% | 87.6% | 49.9% |
| 25% | 0.3% | 73.8% | 56.7% |
| 50% | 0.3% | 55.7% | 61.8% |

The odor rule took the 5 locked test species from 43.6% to 100% of poisonous mushrooms flagged, and from 63.1% to 18.8% "Can't tell", with no fully filled poisonous row called edible. The random forest alone reaches only 53.0% accuracy on those 5 species, so the safety comes from the decision rules. Unseen edible species never get an edible answer, as intended, and 25.6% of them wrongly get "Likely poisonous". On known species (held-out rows, all features given), 99.6% of edible mushrooms get an edible answer and all poisonous ones are flagged.

**Limitations:** The odor rule was added after the 5 locked species had been used, so none of the 23 species remains an independent test and these figures are post-hoc estimates. With so few species, 0.0% doesn't mean the true rate is zero. The data contains hypothetical samples from a field guide rather than field measurements, and it covers only Agaricus and Lepiota species. The documentation also merges "unknown edibility" into poisonous. Users may also misjudge features such as odor.

## Project layout

```text
data/raw/          Original dataset and documentation
mushroom/          Feature schema, model code and prediction logic
scripts/explore.py Data checks behind the design decisions
scripts/train.py   Model selection, evaluation and final fit
scripts/analysis.py Feature importance, error analysis, baselines and latency
report/            Project report, plus the script that builds it
app/               FastAPI backend and static website
tests/             Automated tests
artifacts/         metrics.json, plus model.joblib once trained
```

## Run it locally

Requires Python 3.12.

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux, then:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

```bash
python -m scripts.train
```

```bash
pytest
```

```bash
uvicorn app.main:app --port 7860
```

Open <http://localhost:7860>. Training takes a few minutes and uses fixed seeds, so it reproduces the figures above. It writes `artifacts/model.joblib` (not committed) and `artifacts/metrics.json`.

## Deploy

The `Dockerfile` trains the model during the build, so any host that builds Docker images can run the site. The live site runs on Render:

1. Create an account at render.com and connect it to GitHub.
2. Choose **New > Web Service** and select this repository.
3. Set the language to **Docker** and the instance type to **Free**, then deploy. Render sets `PORT` itself.

Pushes to `main` redeploy the site. Free services sleep after 15 minutes without traffic, so the first visit after that can take up to a minute.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/predict` | Verdict for `{"features": {"odor": "n", ...}}`, using the codes from `/api/schema` |
| `GET` | `/api/schema` | Features and allowed values |
| `GET` | `/api/example/{edible,poisonous,unfamiliar}` | Sample input |
| `GET` | `/api/metrics` | Evaluation results |

## Sources

- Dataset: Schlimmer, J. (1987). *Mushroom* [Dataset]. UCI Machine Learning Repository. <https://archive.ics.uci.edu/dataset/73/mushroom>
- Records drawn from: Lincoff, G. H. (Pres.) (1981). *The Audubon Society Field Guide to North American Mushrooms*. Alfred A. Knopf.
- Libraries: scikit-learn, pandas, NumPy, SciPy, FastAPI, Uvicorn.
- AI tools: this project was built with Claude Code (Anthropic), which was used for code, model evaluation design and documentation.
