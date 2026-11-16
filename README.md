# Cough spectrograms

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23529532.svg)](https://doi.org/10.5281/zenodo.23529532)

**Two stages of acoustic cough analysis, measured on open data with the dataset's own
folds: is this a cough, and what kind of cough is it.**

Cough audio has a decade of published results behind it, most of them reported on a
single split of a small dataset. This repository does both tasks on open data, with the
dataset's own folds, three seeds, and every number given as mean ± standard deviation.
The numbers are therefore smaller than the literature's -- and you can check them.

![Log-mel spectrograms of four confusable sounds](figures/01_spectrograms.png)

---

> ### Source code is not public
>
> The repository holding the code is private. **The source is available for
> technical review under NDA** -- contact me through the links at the end of
> this page. This page documents the method and the measured results.

---

## The road here

Cough acoustics is one of my longest-running lines of work. During the pandemic my team built a system that told COVID-19 from other respiratory conditions by the sound of a cough, published in *IEEE Open Journal of Engineering in Medicine and Biology* (2021, *Acoustery System for Differential Diagnosing of Coronavirus COVID-19 Disease*) and covered by the press in Asia. The same pipeline went on to clinical and commercial partners. This repository is its open core, retrained on public cough datasets so anyone can reproduce every number.

## Where does this come from?

Two peer-reviewed studies I co-authored on cough acoustics, both on **proprietary
clinical recordings** collected through the Acoustery system. Their figures are quoted
as reported by those papers. They are **not** this repository's results, and the two
sets of numbers are deliberately kept apart.

**Mitrofanova A.Y., Mikhaylov D., Shaznaev I., Chumanskaia V., Saveliev V.**
*Acoustery System for Differential Diagnosing of Coronavirus COVID-19 Disease.*
IEEE Open Journal of Engineering in Medicine and Biology **2**, 299–303 (2021).
[doi:10.1109/OJEMB.2021.3127078](https://doi.org/10.1109/OJEMB.2021.3127078)
About 3,000 cough recordings from hospitals in Russia, Belarus and Kazakhstan, April
to October 2020. CNN encoder with a recurrent network and an attention mechanism.
**Accuracy 85%, precision 78.5%, recall 73%.**

**Dvoryankin S.V., Pavalkis D., Ovsyannikov D.Y., Kulzhanova Sh.A., Tuleshova G.T.,
Mikhaylov D.M. et al.** *Can We Recognize a COVID-19 Cough?*
ISSN 1341-2051, **25**(12), 3967–3973 (2020), with Astana Medical University.
70 participants, 196 cough episodes, acoustic criteria reviewed by a pulmonologist.
**Separation of 56 confirmed against 130 non-acute records at accuracy 1.0.**

Both rest on data that cannot be released. This repository is the open counterpart:
the same family of problems on public datasets, reproducible from a clean clone.

---

## What are the two stages?

| | Question | Data | Result |
|---|---|---|---|
| **Stage 1** | Is this sound a cough? | ESC-50, 400 clips, cough vs 9 confusable human sounds | AUC **0.986 ± 0.011**, 10-way accuracy **0.838 ± 0.041** |
| **Stage 2** | Dry cough or wet cough? | COUGHVID, 2,063 physician-labelled recordings | AUC **0.705 ± 0.027**, balanced accuracy **0.655 ± 0.023** |

The gap between the two stages is the point of the repository. Detecting that a cough
happened is close to solved on clean audio -- as a ranking; what it costs at a
threshold is counted in the next section. Saying what *kind* of cough it was is a
different problem, and the measured number for it is much lower. It is lower than most
published cough-typing results, and lower than the physicians agree with each other.

---

## Is this a cough?

Ten human and respiratory classes from [ESC-50](https://github.com/karolpiczak/ESC-50),
40 clips each, five seconds, 44.1 kHz:

`coughing` · `sneezing` · `breathing` · `laughing` · `snoring` · `crying_baby` ·
`clapping` · `footsteps` · `brushing_teeth` · `drinking_sipping`

These are the right distractors. A cough detector that only has to beat traffic noise
is not being tested. Sneezing shares the explosive onset, breathing shares the airway,
laughing shares the repeated glottal bursts.

Five-fold cross-validation on ESC-50's **official folds**, three seeds -- fifteen runs.

| | 10-way accuracy | cough vs rest, AUC |
|---|---|---|
| **mel-CNN** | **0.838 ± 0.041** | **0.986 ± 0.011** |
| Linear baseline (logistic regression on the time-averaged log-mel vector) | 0.428 ± 0.037 | -- |
| Chance | 0.100 | 0.500 |

![Per-fold accuracy and cough-vs-rest ROC](figures/03_results.png)

The linear baseline matters. It is 64 numbers per clip -- the mel spectrum averaged
over time, every temporal structure destroyed -- and it already reaches 0.428.
Whatever the CNN is worth, it is worth the difference between those two rows, not the
difference from chance.

![Confusion matrix](figures/02_confusion.png)

**Cough is one of the three hardest classes in the set.** It is recalled 75% of the
time, and the errors are not spread evenly. 15% of coughs are called **laughing**, twice
as many as are called sneezing (7%), and 7% of laughs come back as coughs. That is not
the error you would predict: sneezing looks like the obvious confounder. Laughing is a
train of short glottal bursts at roughly cough spacing. Once the mel filterbank has
dropped the fine harmonic detail, the two are close neighbours. How many clips that
reading rests on is counted below.

`drinking_sipping` does worse (63%, lost to footsteps more often than to anything else)
and so does `breathing` (73%). At the other end `snoring` (97%) and `crying_baby` (95%)
are nearly free. They are long, periodic and low-frequency, with nothing else in the set
resembling them.

### Is that 120 predictions or 40 clips?

Every percentage above is a share of 120 predictions: the 40 clips of a class, scored
by three seeds. The seeds are three trainings of one network on the same clips, so they
are not three samples of coughs. `results/stage1_clip_checks.json` reads the same
fifteen runs again with the clip as the unit, and with ESC-50's own metadata for the
recording each clip was cut from (script in the private repository,
`src/stage1_clip_checks.py`; it needs neither audio nor training). Added on
5 October 2026.

**The cough errors are clips, not noise.** Each seed gets exactly 30 of the 40 cough
clips right (95 % interval for 30 of 40: 60–86 %). The 30 wrong predictions fall on 15
clips, and **7 clips are missed by all three seeds** -- 21 of the 30 errors. 25 clips
are right every time. A different seed does not move those seven. Over all ten classes
the three seeds give the same answer on 344 of the 400 clips, and 43 clips are never
classified correctly.

**"Second-hardest" was wrong.** The first version of this page said that only
`drinking_sipping` does worse than cough. The pooled confusion matrix above already
said otherwise:

| Class | Right, of 120 predictions | Right by majority of seeds, of 40 clips | Never right |
|---|---|---|---|
| drinking_sipping | 76 (63 %) | 26 | 9 |
| breathing | 88 (73 %) | 29 | 8 |
| **coughing** | **90 (75 %)** | **32** | **7** |
| laughing | 93 (78 %) | 31 | 5 |
| footsteps | 102 (85 %) | 34 | 5 |

Cough is third from the bottom by predictions and fourth by clips. Breathing, coughing
and laughing sit within three clips of each other (the interval on 32 of 40 is
65–90 %), and forty clips cannot rank them.

**Laughing rather than sneezing: six clips against two.** "15% against 7%" is 18
predictions against 8. By clip, of the seven coughs no seed recognises, five are called
laughing every time and two sneezing every time; by majority of seeds it is six against
two. If the two confusions were equally likely, a gap that large would turn up 45 %
(five against two) or 29 % (six against two) of the time. Three of the five
always-laughing clips also carry near-consecutive Freesound numbers
(87794, 87795, 87799). The dataset lists them as three recordings, but numbers that
close usually mean one upload session, and if so the count is three recordings against
two. The other direction is level: the laughs called cough are 8 predictions on
**3 clips**, and sneezes called cough are 5 predictions on 3 clips. So "laughing, not
sneezing, is the confounder" is what these forty clips suggest. They do not establish
it. The file names of all 15 clips are in the results file for anyone who wants to
listen.

**At a threshold, "close to solved" has a price.** Each seed's held-out cough
probabilities, pooled over the 400 clips, give AUC 0.983, 0.979, 0.978 -- a little
under the 0.986 mean of the per-fold values, because five separately trained models
now share one threshold.

| Operating point | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Coughs caught (of 40) before the first false alarm | 12 | 17 | 5 |
| Coughs caught with 3 false alarms in the 360 other clips (0.8 %) | 29 | 28 | 30 |
| Coughs caught with 18 false alarms (5 %) | 36 | 33 | 35 |
| False alarms needed to catch 36 of 40 | 16 | 26 | 20 |
| False alarms needed to catch 38 of 40 | 41 | 31 | 48 |
| False alarms needed to catch all 40 | 41 | 57 | 86 |

Held under one per cent false alarms, the detector finds 28 to 30 coughs of 40. To find
38 it fires on 31 to 48 of the other 360 clips (9–13 %), so even with one cough for
every nine other sounds only **44–55 % of the alarms are coughs**. At that setting the
false alarms come from sneezing first: 35 of the 120 over the three seeds, laughing 31,
drinking 22. Sneezing is the lesser confusion when the network has to name the class,
and the leading one when it only has to say "cough" generously. One clip is 2.5 points
of recall here, so the table is given as counts.

**Seeds move the number less than folds, as in stage 2.** Scored over all 400 clips the
three seeds get 343, 331 and 331 right: 0.838 ± 0.017 across seeds, against ± 0.041
across the fifteen runs. Averaging the three seeds' probabilities gets 345 -- two clips
more than the best single seed.

**What this does not show.** The 400 clips come from 326 source recordings, and
unevenly: the 40 coughs from 39, the 40 sneezes from 40, but the 40 `crying_baby` clips
from 18, one of which supplies seven. ESC-50's folds keep a source on one side of the
split, so nothing leaks, but "95% on `crying_baby`" is a statement about 18 recordings.
The Freesound-number argument is an inference from the metadata, not a check of who
uploaded what. And this is still one architecture on curated clips: the seven coughs it
never recognises are worth hearing before anything is concluded about coughs in
general.

---

## What kind of cough is it?

[COUGHVID](https://zenodo.org/records/7024894) (EPFL, CC BY 4.0) is about 34,000
crowdsourced recordings. Four physicians labelled a subset for **cough type: dry or
wet**, along with severity, wheezing, stridor and dyspnea. Dry versus wet (productive)
is the first split any clinician makes by ear, and it is the one open label of its
kind.

**2,063 recordings** survive the filter -- a majority expert label, ties dropped, and
the dataset's own cough detector above 0.8. The split is **1,587 dry / 476 wet**.

![Dry and wet coughs](figures/04_dry_wet.png)

![Cough type results](figures/05_coughtype_results.png)

| | ROC-AUC | balanced accuracy |
|---|---|---|
| **mel-CNN** | **0.705 ± 0.027** | **0.655 ± 0.023** |
| Linear baseline (time-averaged log-mel) | 0.559 ± 0.025 | 0.540 ± 0.023 |
| Always answer "dry" | 0.500 | 0.500 |
| *Physician vs physician, where they overlapped (n = 565) — raw agreement, not balanced accuracy; see "What the 77 % allows"* | *--* | *0.770 (κ 0.35)* |

**The ceiling is the labels, not the model.** The four physicians overlapped on only 565
judgements. Where they overlapped they agreed **77.0%** of the time, ranging from 67%
to 93% depending on the pair. A model cannot cleanly exceed the consistency of the
thing it is fitted to. That number belongs on the chart, and it is why the result here
is reported as ROC-AUC and balanced accuracy rather than accuracy. How high the
ceiling is, in numbers, is worked out below in "What the 77 % allows": a model that
reproduced the truth exactly would score AUC 0.71 to 0.85 against these labels, and
the CNN's 0.701 is 57 to 97 % of the way there (added 16 November 2026).

**Accuracy would be a lie here.** Always answering "dry" scores 0.769 without listening
to anything, which is roughly the physicians' own agreement rate. Any paper reporting
bare accuracy on an imbalanced cough-type set is reporting the class prior.

### What happens when the model has to make a call?

An AUC of 0.705 says nothing about what happens at a threshold.
`results/stage2_operating_points.json` reads the same fifteen runs three more ways
(script in the private repository, `src/operating_point.py`).

**The CNN is ahead of the linear baseline on every one of the 15 fold × seed pairs.**
Paired difference in ROC-AUC on the same fold and seed: **+0.146 ± 0.036** (mean ± sd,
95 % CI ±0.018), worst run +0.077, best +0.217. Per fold the CNN sits between 0.683
(fold 2) and 0.734 (fold 4), the baseline between 0.550 and 0.575. So the gap is
not a lucky split. It is small, but it is everywhere.

**Most of the run-to-run spread is the fold, not the seed.** Pooling each seed's
out-of-fold probabilities over all 2,063 recordings gives AUC 0.706, 0.700, 0.698.
That is **0.701 ± 0.004** across seeds, against ± 0.027 across the fifteen runs.
Retraining changes the number less than which 413 recordings you happen to score.
(And that fold spread is what sampling 413 recordings predicts, 0.027 against 0.027
— no fold is harder than another; see "What the 77 % allows", added 16 November 2026.)

**At the class-weighted argmax threshold the model over-calls "wet".** Pooled over
the 15 runs it flags **50 %** of recordings as wet when 23 % are: wet recall 0.74,
dry recall 0.57, precision on "wet" 0.34. Two of every three "wet" calls are wrong.

| Dry coughs kept correct (specificity) | Wet coughs caught (sensitivity), mean ± sd over 3 seeds |
|---|---|
| 90 % | **0.26** ± 0.03 |
| 80 % | **0.46** ± 0.00 |
| 70 % | **0.59** ± 0.00 |

Look at the table before imagining a product. If the false-alarm rate on dry coughs
has to stay at 10 %, the model finds one wet cough in four. It only becomes useful as
a *screen* -- high recall, low precision -- which is the opposite of what a
cough-typing claim usually implies.

### What the 77 % allows

Three sentences above carry no number: "the ceiling is the labels", the 0.770 in
the results table, and the ± on the stage-2 figures. All three were read on
16 November 2026 from the stored runs
(`results/label_ceiling_check.json`; script `src/label_ceiling_check.py` in the
private repository). The script first reproduces the pooled AUCs, the operating
points and the pooled counts above to every digit.

**How high the ceiling is.** Almost every label is one physician's call, and two
physicians agree 77.0 % of the time. Write that as two raters who mislabel a dry
cough as wet with probability a and a wet one as dry with probability b, over a
true wet prevalence w: the agreement and the labelled-wet share (476 of 2 063)
are two equations in three unknowns, so they pin a one-parameter family, from
"physicians never call a dry cough wet" (a = 0, b = 0.50, w = 0.46) to
"physicians never call a wet cough dry" (a = 0.15, b = 0, w = 0.10). Along it, a
model that reproduced the truth *exactly* — the best any model can do — is
scored against one-rater labels and gets **AUC 0.71 to 0.85** (0.739 under
symmetric errors, a = b = 0.13, where the true prevalence is 13 %). The CNN's
0.701 is **57 % to 97 % of the way from chance to the ceiling** (84 % under
symmetric errors). The perfect model's sensitivity against the labels is the
striking column: under symmetric errors it is 0.50 — half of the recordings
labelled wet are, on this model of the labels, dry coughs a physician called
wet, so a perfect model "misses" a wet cough half the time. Its balanced
accuracy ceiling is 0.71 to 0.85 against the CNN's 0.655.

| dry called wet, a | wet called dry, b | true wet share, w | perfect model: AUC | sensitivity | specificity | balanced accuracy | kappa | CNN's share of the ceiling |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.50 | 0.46 | 0.851 | 1.000 | 0.702 | 0.851 | 0.52 | 57 % |
| 0.02 | 0.47 | 0.42 | 0.847 | 0.949 | 0.744 | 0.847 | 0.54 | 58 % |
| 0.05 | 0.42 | 0.34 | 0.834 | 0.858 | 0.811 | 0.834 | 0.57 | 60 % |
| 0.08 | 0.35 | 0.27 | 0.811 | 0.746 | 0.877 | 0.811 | 0.59 | 65 % |
| 0.10 | 0.29 | 0.21 | 0.789 | 0.660 | 0.919 | 0.789 | 0.59 | 70 % |
| 0.13 (symmetric) | 0.13 | 0.13 | 0.739 | 0.502 | 0.977 | 0.739 | 0.56 | 84 % |
| 0.14 | 0.08 | 0.12 | 0.726 | 0.464 | 0.988 | 0.726 | 0.55 | 89 % |
| 0.15 | 0.00 | 0.10 | 0.707 | 0.414 | 1.000 | 0.707 | 0.52 | 97 % |

The CNN's share is (0.701 − 0.5) / (ceiling − 0.5). At the right-hand end of the
family the ceiling is 0.707 and the model is at it.

**The 0.770 is raw agreement, not balanced accuracy.** It sits in the balanced-
accuracy column of the results table above because that is the only column it
could go in, but two raters who both call 77 % of coughs dry agree 65 % of the
time by prevalence alone. Cohen's kappa takes that out. At the labelled subset's
marginals (23 % wet for both raters — the overlap's own marginals are not in the
stored file) the physicians sit at **0.35 on the kappa scale**, and the four do
not agree alike: the three pairs that involve physician 3 are at **0.07 to 0.11**,
a whisker above chance, the other three at **0.50 to 0.80**. The CNN against the
label scores **0.22 at the argmax threshold and 0.25 at its best threshold**; a
model reproducing the truth exactly would score **0.52 to 0.59** against
one-rater labels. So on one scale: physicians with each other 0.35 (0.07–0.80 by
pair), the CNN with the labels 0.25 at best, a perfect model with the labels
0.52–0.59.

| who against whom | judgements | raw agreement | kappa |
|---|---|---|---|
| physician 1 vs physician 2 | 83 | 0.855 | 0.59 |
| physician 1 vs physician 3 | 79 | 0.671 | 0.07 |
| physician 1 vs physician 4 | 85 | 0.929 | 0.80 |
| physician 2 vs physician 3 | 101 | 0.673 | 0.08 |
| physician 2 vs physician 4 | 112 | 0.821 | 0.50 |
| physician 3 vs physician 4 | 105 | 0.686 | 0.11 |
| all four, pooled | 565 | 0.770 | 0.35 |
| CNN against the label, class-weighted argmax (flags 50 %) | 2 063 per seed | 0.609 | 0.22 |
| CNN against the label, best threshold (flags 30 %) | 2 063 per seed | -- | 0.25 |
| a model reproducing the truth exactly, against one-rater labels | -- | 0.771 to 0.867 | 0.52 to 0.59 |

**The ± on this page are retraining spreads, not uncertainties.** 0.701 ± 0.004
is three seeds; the sampling sd over the 2 063 recordings is 0.013, three times
that. Resampling recordings (2 000 bootstrap draws; seed 0 shown, seeds 1 and 2
are in the JSON):

| quantity | seed 0 | 95 % interval over recordings | spread over 3 seeds (sd) |
|---|---|---|---|
| pooled out-of-fold ROC-AUC | 0.706 | [0.681, 0.730] | 0.004 |
| wet coughs caught at 90 % specificity | 0.26 | [0.22, 0.32] | 0.03 |
| wet coughs caught at 80 % specificity | 0.46 | [0.41, 0.50] | 0.00 |
| wet coughs caught at 70 % specificity | 0.60 | [0.53, 0.64] | 0.00 |
| balanced accuracy at the argmax threshold | 0.657 | [0.634, 0.679] | 0.002 |
| kappa at the argmax threshold | 0.22 | [0.19, 0.26] | 0.00 |

The "± 0.00" at 80 % and 70 % specificity in the table above is therefore not
precision: those sensitivities sit in [0.41, 0.50] and [0.53, 0.64].

**And the fold spread is sampling.** "Most of the run-to-run spread is the fold,
not the seed" is true and says less than it sounds: the sd of the fifteen run
AUCs is 0.027, and drawing 413 recordings at random from one seed's pooled
scores gives an AUC sd of **0.027 against 0.027** — the same number. Every fold
mean (0.683 to 0.734) lies inside the random-subsample interval [0.652, 0.757].
No fold is harder than another; a 413-recording test set is that noisy.

**What this does not show.** The label model assumes two physicians err
independently with the same error rates, that the 565 overlapping judgements
are representative of the 2 063 labels, and that "the truth" is a binary dry/wet
a physician could in principle get right; the kappa for the physicians uses the
labelled subset's marginals because the overlap's own are not stored, and 83 to
112 judgements per pair make each pair's figure rough. The bootstrap resamples
recordings, not speakers (the dataset cannot group them). Nothing here changes a
model or a number already on the page; it says what the numbers can and cannot
mean.

### How are the numbers checked?

The run-level results are in `results/`: `cv_runs.json` for stage 1 with per-clip
logits, `coughtype_runs.json` for stage 2 with per-recording probabilities.
`tests/test_readme_numbers.py` recomputes every figure quoted on this page from those
files -- the ± values, the 75 % cough recall, the 15 % laughing confusion, the
2,063 / 1,587 / 476 split, the operating points above. `tests/test_stage1_clips.py`
does the same for the clip-level reading of stage 1, from the stored logits; only the
source-recording counts are taken from `results/stage1_clip_checks.json`, because
ESC-50's metadata is not redistributed here. A regenerated results file that no longer
matches the text fails the test. `python -m pytest tests/`.

---

## Limitations

Read these before quoting any number above.

- **Stage 1 is 400 clips from 326 recordings.** Forty per class, 320 in training per fold. Small enough that
  the model memorises the training set in a few epochs without augmentation, and small
  enough that a 3-point difference between methods is noise.
- **ESC-50 is not clinical data.** Curated Freesound audio: one cough per clip, clean
  recording, no comorbidity, no microphone variety, no clinician's label. Nothing in
  stage 1 says anything about detecting *disease*.
- **Digital silence is a cue in stage 1.** Several clips contain long stretches of true
  silence around the event, which becomes a flat block in the standardised log-mel
  image. That block is informative and the network is free to use it. A property of the
  dataset, not of coughs.
- **COUGHVID labels are single-rater for most clips.** Only 107 recordings were judged
  by two or more physicians. Almost every training label is one doctor's opinion, and
  the measured 77% pairwise agreement says how much that opinion moves between doctors
  — on the kappa scale 0.35, and the three pairs involving one of the four physicians
  sit at 0.07 to 0.11, barely above chance (see "What the 77 % allows").
- **COUGHVID is crowdsourced audio.** Phone microphones, uncontrolled rooms, no
  diagnosis confirmed by test for most uploads, and no way to group recordings by
  speaker. If one person uploaded several clips they can land on both sides of a
  split. That would inflate stage 2, not deflate it.
- **Neither stage diagnoses anything.** Dry versus wet is a description of a sound. It
  is not a disease label, and this repository makes no clinical claim.
- **One architecture, no sweep.** The same small CNN runs both stages. A tuned model
  would score better and its reported number would be less trustworthy for exactly the
  reason this README keeps repeating.

---

## What happened along the way?

Stage 2 was not the first plan. An earlier idea was to follow cough detection with
asthma, but ESC-50 has no asthma and ICBHI 2017 has one asthma patient out of 126.
Dry versus wet on COUGHVID was what remained.

torchaudio would not load on the machine I built this on (an ABI conflict with torch
2.9.1 under Python 3.14). So the mel filterbank is twenty lines of numpy, audio is
read with soundfile, and COUGHVID's webm files go through ffmpeg first. Because
people do not cough in the first second of a phone recording, the loader
takes the five-second window with the highest smoothed energy.

On a laptop GPU stage 1 takes about 41 s per fold, ten minutes in all; stage 2 about
200 s per fold, around 45 minutes. Both training
scripts write one JSON per (seed, fold) and skip finished pairs on restart.
The operating-point reading and the test file that pins every number were added on
17 September 2026, after the first version of this page had gone up without them.
The clip-by-clip reading of stage 1 followed on 5 October 2026 and withdrew one claim:
cough is not the second-hardest class.

---

The frame rate, the band count and the frequency ceiling decide what survives into the
image, and those decisions should be visible, not buried in a default argument. That
is the other reason the mel filterbank is written out by hand.

## What else is related?

- [**making-pinns-work**](https://github.com/drdmitrymikhaylov/making-pinns-work) -- the
  same way of reporting applied to physics-informed neural networks: why they fail to
  converge, measured over seeds rather than asserted.
- [**navierpinn-mineral-ore-body-reconstruction**](https://github.com/drdmitrymikhaylov/navierpinn-mineral-ore-body-reconstruction) -- a PINN solver inside
  a 3D ore-body modelling application.

## Which data and licences apply?

Code **MIT**. Neither dataset is redistributed here; both are downloaded at setup and
keep their own terms.

- **ESC-50** -- CC BY-NC 3.0. K. J. Piczak, *ESC: Dataset for Environmental Sound
  Classification*, ACM Multimedia 2015. The non-commercial term travels with it.
- **COUGHVID** -- CC BY 4.0. L. Orlandic, T. Teijeiro, D. Atienza, *The COUGHVID
  crowdsourcing dataset, a corpus for the study of large-scale cough analysis
  algorithms*, Scientific Data 8, 156 (2021).
  [doi:10.1038/s41597-021-00937-4](https://doi.org/10.1038/s41597-021-00937-4)

---

**Prof. Dr. Dmitry Mikhaylov** · Abu Dhabi, UAE
[LinkedIn](https://www.linkedin.com/in/dmitry-mikhaylov) ·
[ORCID](https://orcid.org/0009-0009-2108-6820) ·
[Substack](https://dmitrymikhaylov.substack.com)
