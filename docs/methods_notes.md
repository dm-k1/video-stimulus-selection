# Methods: selection of the stimulus compilation

Draft text for the methods section. All figures come from the outputs saved in `stimulus_selection.ipynb`; the code is in `src/`. The points at the end are still open and are not yet covered by this text.

## Material

The material consisted of 46 clips (2.5 to 15.3 s; 395 s in total) taken from 11 broadcasts on migration by BBC News, Democracy Now!, FRANCE 24, Nieuwsuur, NOS, NOS op 3, Sky News and the Dutch Immigration and Naturalisation Service (IND). All clips were converted to 1920×1080, 25 fps, H.264 video and AAC stereo audio. Audio was levelled to −23 LUFS (EBU R128) with a tolerance of ±0.5.

## Shots and descriptions

Cuts were detected automatically from frame-to-frame change (ffmpeg scene score above 0.3); detections less than 0.5 s apart or within 0.5 s of the start or end of a clip were discarded. This gave 108 shots, the shortest lasting 1.38 s.

Every clip was described by a vision-language model (Gemini 3.5 Flash-Lite) from the picture only, without sound. Clips were sampled at 3 frames per second, so every shot was sampled at least four times. The model was instructed to describe only what was visible, not to interpret, not to use the words migrant, refugee or asylum seeker, and not to state or guess ethnicity, nationality or religion. It returned one description per shot, 112 in total. The descriptions come from a single run of the model; because the model does not return the same wording twice, they are one sample of its output, not a fixed property of the footage. The agreement between two embedding models (below) checks only the embedding step, not the descriptions; the comparison with the pictures (under Dimensions) gives a partial check that does not rely on them. The descriptions serve as a measurement of the visible content, and the later steps test that measurement: dimensions that do not return under resampling are dropped, and the remaining dimensions are compared with an image model that never sees the descriptions. The model is proprietary and may change or be withdrawn, so this step cannot be repeated with the same model later; the stored descriptions make every later step exactly reproducible.

## Content space

Camera vocabulary (for example "wide shot", "close-up") was removed from the shot descriptions, which were then embedded with a sentence-embedding model (all-mpnet-base-v2). Distances between shots correlated at *r* = .88 with those from a second model (all-MiniLM-L6-v2).

The embeddings were submitted to a principal component analysis. Parallel analysis (1,000 column-wise permutations, 95th percentile) retained 19 components, which held 64% of the variance; distances in these 19 components correlated at *r* = .92 with distances in the full space.

## Dimensions

The 19 components were rotated with varimax. With an oblimin rotation, the largest correlation between the rotated dimensions was .23 (1 of 171 pairs above .20; median .04), below the threshold of .32, and the stable dimensions matched their oblimin counterparts at .88 to .99, so the orthogonal solution was retained.

Stability was assessed by repeating the analysis on 1,000 random subsamples of 80% of the shots and matching the resulting dimensions one-to-one to the originals. Sixteen dimensions matched at .75 or higher on average and were retained.

As a check that did not rely on the descriptions, frames were sampled at 3 per second and embedded with an image model (CLIP ViT-B/16), and averaged per shot. Shots could be matched to frames for 101 of the 112 described shots; four clips were left out because the cut detection and the description disagreed on the number of shots. Text distances and picture distances correlated at *r* = .54, and at *r* = .48 for pairs of shots from different broadcasts. Each stable dimension was predicted from the picture embeddings with ridge regression, leaving out one broadcast at a time; the correlations between predicted and observed scores ranged from .23 to .85.

## Content types

For every clip and both poles of every stable dimension, the standardised score of the clip's furthest shot towards that pole was taken. A clip was tagged with a pole if this score lay at least 2 *SD* out. A tag was retained only if the clip received it in more than half of the 1,000 subsample analyses, and a pole counted as a content type if at least two clips carried a retained tag.

Of 63 initial tags, 17 were dropped as unstable and one was added. One stable pole could not be interpreted and was excluded. One content type that no single pole captured (walking with luggage) was added manually with four tags, and 3 of 598 cells of the automatic table were corrected by hand. These manual tags were proposed from the descriptions and then checked against the footage. The final table held 14 content types: highway and trucks, inflatable boats, ships, police and checkpoints, close-up of a face, men sitting on grass, lying down and resting, walking with luggage, aid being handed out, tent camp, halls with cots, city street, classroom, and lines of men standing.

## Selection

All 2^46 combinations of clips were evaluated exactly, by counting combinations per set of content types and total length. The shortest combinations showing all 14 content types lasted between 61 and 62 s. The candidate pool consisted of all combinations of 50 to 65 s that showed at least 13 of the 14 content types (*n* = 2,654, of which 35 showed all 14).

Candidates were ranked on the sum of three measures, each standardised within the pool: (a) the clarity with which each content type was shown, taken from the position of the clearest clip on the corresponding dimension; (b) coverage of both poles of the 16 stable dimensions; and (c) the mean distance in the embedding space from the footage of the whole set to the nearest selected shot, reversed. In (c) every shot was weighted by its duration, because clips are always shown whole; shot durations came from the cut detection for 42 clips and were divided evenly over the described shots for the four clips where the number of detected and described shots differed.

Three checks followed. First, the content mix of the 50 best compilations was compared with that of 5,000 compilations of the same length drawn without regard to content, using the duration-weighted energy distance between a compilation's shots and all shots. All 50 were somewhat broader than the whole set, giving rare content more room than its share (median: more extreme than 75% of the random compilations), but mostly within the range of the random compilations. Second, the ranking was repeated with shots weighted by the square root of their duration and with every shot counted once. The rankings of the pool agreed closely (Spearman ρ = .96 and .92), but the top shifted: 32 and 22 of the 50 best stayed among the 50 best. A compilation's robust rank was therefore defined as its worst rank over the three weightings. Third, the pool was cut at lengths from 65 down to 55 s and ranked again within each cut.

Up to 62 s, all 14 content types could be shown, but not by a compilation that ranked high under all weightings. Up to 62, 61, 59 and 58 s, the best compilation consisted of the same nine clips plus one of three clips showing people walking with luggage, which set the length, and it ranked first under all three weightings; up to 60 s, a variant that showed city street instead of lines of men standing ranked first under two weightings and second under the third. Below 58 s few compilations remained (33 up to 57 s, 2 up to 55 s). A compilation of under 60 s was wanted for the experiment. This was a practical constraint, not a result of the analysis, and since showing all 14 content types took at least 61 s, one type had to be given up. Two compilations were rendered: the best up to 59 s and the best up to 58 s (58.8 and 57.8 s by the clip lengths; 58.0 and 57.0 s of picture, because the sound of each clip runs a few frames longer than its picture). Both consist of 10 clips from 8 sources and show 13 of the 14 content types. City street, the type they leave out, fell out because of the length limit, not because it was judged less important; that it was also the type least visible in the pictures (*r* = .23), with only two retained tags, one of them set by hand, means that little is lost.

The compilations are shown without sound. The content was selected on the picture alone, so the voice-overs, languages and music of the broadcasts were not part of the selection and are left out of the stimulus.

Further robustness checks (`robustness_checks.ipynb`) rank the pool again with each ranking measure left out or counted double, with other ceilings for strength (3 and 5 *SD*) and coverage (1.5 and 2.5 *SD*), and with a stability threshold of .70 or .80 for keeping a dimension, and report where the two chosen compilations then rank among the compilations of their length (134 compilations up to 59 s, 73 up to 58 s). Both ranked first in 12 of these 13 variants. The 58.0 s compilation dropped to second place only with a threshold of .70, which added one dimension that gave no new content type; the compilation then ranked first differed in two clips, showing city street instead of lines of men standing. The 57.0 s compilation dropped to sixth place only when strength was left out of the ranking altogether. Removing two dimensions with a threshold of .80 affected no content type and left both compilations first.

## Nature of the procedure

The procedure is a form of maximum variation sampling: the compilation is chosen to span the range of content in the set, so that a rare content type is represented as much as a common one. It was developed on the same 46 clips to which it was applied, and thresholds were set after inspecting results, so it should be read as a documented selection procedure and not as a test of how news footage on migration is structured. Hierarchical clustering and non-negative matrix factorisation were tried first and did not give stable categories.

## Reproducibility

Every random step used a fixed seed (NumPy generator, seed 0): the 1,000 permutations of the parallel analysis, the 1,000 subsamples used for the stability of the dimensions and, as the same 1,000 subsamples, for the retention of tags, and the 5,000 random compilations of the mix check. All other steps are deterministic (exact principal component analysis, rotation, leave-one-broadcast-out prediction, exact enumeration of all combinations, and the ranking), so a rerun of the analysis gives the same numbers. The shot descriptions are the exception: the language model does not return the same wording twice, so they were generated once and stored, and every later step starts from that stored file.

## Still open

- The choice between the two rendered compilations, which differ only in the clip showing walking with luggage.
- The order of the clips: the rendered versions play them in the order of `videos.csv`, which puts clips from the same broadcast back to back.
