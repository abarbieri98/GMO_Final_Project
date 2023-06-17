# Evolving ambience with Differential Evolution

## Goal of the project
This repo contains the final project for the Global and Multi-Objective Optimization course @ DSSC.
The main goal is to use Differential Evolution and variations on the theme to optimize some audio plugins in order to match a target guitar sound. The source code implements multiple DE algorithms with a Lamarckian take based on the papers cited in `presentation.pdf`

## Folder content

The contents of the repo are the following:
- `project_source.py`: source code implementing the DE/rand/1 and JADE algorithm, together with a Lamarckian optimiziation step based on the Hookes-Jeeves optimization algorithm.
- `example.ipynb`: notebook containing an example of use of the algorithms
- `songs`: brief audio clips, divided into dry and wet, to test the algorithms
- `presentation.pdf`: overall peresentation of the project with focus on the algorithms used and the results obtained.

## Requirements

To properly work, the source code needs Supermassive, Freq Echo and Space Modulator VSTs, distributed by [Valhalla DSP](https://valhalladsp.com). As for libraries, the needed ones are the following:
- numpy
- [pedalboard](https://github.com/spotify/pedalboard)
- scipy
- librosa






