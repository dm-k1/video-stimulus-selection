# video-stimulus-selection

Selection of a video compilation of about one minute from 46 clips of news footage about migration, for use as stimulus material in an experiment. The aim is a compilation that shows every kind of content in the set at least once, chosen by a documented and repeatable procedure.

## Contents

| Path | What it is |
|---|---|
| `stimulus_selection.ipynb` | The full pipeline, with an explanation above every step |
| `robustness_checks.ipynb` | Checks whether the chosen compilations depend on the conventions of the ranking and the stability threshold (changes no files) |
| `src/` | The functions the notebook uses, one file per part of the pipeline |
| `videos/` | The 46 clips, named `<source>_<nn>.mp4`, all 1920×1080, 25 fps, H.264, AAC stereo, −23 LUFS (not in the public repository; see *The footage*) |
| `videos.csv` | One row per clip: source, original file name, duration, loudness, number of shots |
| `results/` | Every table the notebook generates (see below) |
| `compilations/` | The rendered compilations (not in the public repository) and `compilations.csv`, the order and timing of their clips (step 8) |
| `one_off/` | Code that was run once and is kept for the record (renaming the original files) |
| `docs/methods_notes.md` | Draft of the methods section, with the figures from the saved run |
| `environment.yml` | The conda environment |

## The procedure

1. **Normalize** the clips to one picture format and one loudness.
2. **Detect the shots** in every clip.
3. **Describe** every shot with Gemini, from the picture only.
4. **Content space**: embed the descriptions and keep the dimensions that are stronger than noise.
5. **Dimensions**: rotate them, keep the ones that are stable under resampling, and check them against the frames with an image model.
6. **Content types**: a table of what occurs in each clip, yes or no, in which every tag has survived resampling.
7. **Candidates**: evaluate every possible combination of clips, rank those that show (nearly) all content types, and check the best ones: is their content mix representative, does the ranking survive a different weighting of shots, and what does a shorter compilation cost.
8. **Render** the chosen compilations: join their clips into one video each.

## Generated files (`results/`)

| File | Made in step | Content |
|---|---|---|
| `video_descriptions.csv` | 3 | Gemini's description of every clip and shot |
| `shot_dimensions.csv` | 5 | Every shot with its scores on the stable dimensions |
| `dimensions.csv` | 5 | The stable dimensions: stability, visibility in the pictures, typical words |
| `clip_profiles.csv` | 6 | Every clip's position on every stable dimension |
| `clip_types.csv` | 6 | Clips by content types, 1 or 0 |
| `candidate_compilations.csv` | 7 | The highest-ranked compilations that differ enough from each other |
| `robust_compilations.csv` | 7 | The compilations that rank high under every shot weighting |
| `length_limits.csv` | 7 | The best compilation up to each length limit |

Step 8 writes the videos to `compilations/`, with `compilations/compilations.csv` listing the clips of each video in playing order, with their start and end times.

`video_descriptions.csv` cannot be regenerated identically, because the model does not return the same wording twice. It is the fixed input for steps 4 to 8.

## The result

Two compilations were rendered, both the best-ranked compilation under every shot weighting within their length: `option_A_grenzen02.mp4` (58.0 s) and `option_B_opvang05.mp4` (57.0 s), both **without sound**, because the content was selected on the picture alone. They share nine clips and differ in the clip that shows people walking with luggage. Each has 10 clips from 8 sources and shows 13 of the 14 content types. Showing all 14 types takes at least 61 to 62 s, and a compilation under 60 s was wanted; the type left out, City street, is the one least visible in the pictures. Both compilations stay first in 12 of 13 variations of the conventions in the ranking and the stability threshold (`robustness_checks.ipynb`). The order of the clips is not decided yet.

## Running it

```bash
conda env create -f environment.yml
conda activate migration-stimuli
jupyter lab stimulus_selection.ipynb
```

`ffmpeg` is installed with the environment. Then run all cells (Kernel → Restart Kernel and Run All Cells). Normalizing the clips (step 1), describing them with Gemini (step 3) and rendering the compilations (step 8) are switched off, because their results are final; switching step 3 back on needs a Gemini API key. `robustness_checks.ipynb` runs the main notebook itself first and then the checks; it changes no files.

## The footage

The clips are short excerpts from broadcasts by BBC News, Democracy Now!, FRANCE 24, Nieuwsuur, NOS, NOS op 3, Sky News and the IND. The rights to the footage remain with the broadcasters, so the clips and the rendered compilations are not included in the public repository. `videos.csv` lists for every clip the outlet, the title and URL of the broadcast, the original file and the trimmed segment (`trimmed_start`, `trimmed_end`), so the set can be reconstructed. Running the notebook needs the clips in `videos/`: the checks on the files (steps 0 to 2), the picture check (5d), the shot lengths (7b) and rendering (step 8) read them. Without the clips, every result can still be read from the saved notebook outputs and the tables in `results/`.

## Author

D.M. Kloppenburg, University of Amsterdam.

To cite this repository, use the "Cite this repository" button on GitHub or the details in `CITATION.cff`.

## License

The code is released under the MIT License (see `LICENSE`). The footage is not covered by it.
