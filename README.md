# author-name-disambiguation-using-mcmc
<p align="justify">This repository contains the code for my master's thesis entitled "Author Name Disambiguation using Markov Chain Monte Carlo (MCMC)", submitted to <a href="https://west.uni-koblenz.de/">WeST</a>, University of Koblenz-Landau. The thesis was supervised by 
<a href="https://scholar.google.com/citations?user=PlKALskAAAAJ&hl=en&oi=ao">Dr. Zeyd Boukhers</a> and <a href="https://scholar.google.com/citations?user=aRiTyU0AAAAJ&hl=en&oi=ao">Dr. Claudia Schon</a>.</p>

## Abstract
<p align="justify">The ambiguity of author names in digital libraries leads to incorrect document retrieval and ultimately to incorrect attribution to authors. Name disambiguation is still a hot research topic due to the challenges it presents. To address this problem, this thesis introduces AND-MCGC - <b>A</b>uthor <b>N</b>ame <b>D</b>isambiguation using <b>M</b>arkov <b>C</b>hain-based <b>G</b>raph <b>C</b>lustering, a method based on Markov Chain Monte Carlo sampling to generate disjoint clusters of papers such that each cluster contains papers belonging to a single real-world author. The method constructs a network of papers and repeatedly modifies the topology of this network to generate sub-graphs with homogeneous papers. In modifying the topology of the network, several discriminative features are used, such as the authors' research area, the pattern of co-authorship, the topical publication patterns over the years, and affiliations. The proposed approach achieves an F1 score of 50.29%, outperforming one of the baselines used for comparison. Extensive experiments were conducted to identify the features that contribute most to name disambiguation. The best results were obtained when all features were combined.</p>

## Installation
### Environment setup

To run the project you need `python=3.8`, so it is recommended to setup a virtual environment.
```
conda create -n mcmc_venv python=3.8 anaconda
conda activate mcmc_venv
```
Clone the repository and install the required packages.
```
cd $HOME
git clone https://github.com/nagaraj-bahubali/author-name-disambiguation-using-mcmc.git
cd author-name-disambiguation-using-mcmc
pip install -r requirements.txt
```

### Data
Download the [dataset][1], unzip it and place it in the [data/input][2] folder. The data folder should look like below.

```
data
    input
        demo
        Aminer-534K
    output
        demo_output
```

## Reproduction
If you want to get quick results, just run the project on [demo][3] dataset. To reproduce the entire results change the [configurations][4] by updating `path_to_dataset = './data/input/Aminer-534K/'`.

run using virtual environment
```
python3 main.py
```
or run directly using docker
```
docker build -t and-mcgc:latest .
docker run and-mcgc:latest
```

Once the code is finished running it generates output similar to the files available in [output/demo_output][5].</br>
`disambiguated_files`: contains the disjoint clusters of paper IDs for each atomic name.</br>
`clustering_results.pickle`: contains the pairwise validation results of the generated clusters for each atomic name.</br>
`summary.log`: contains the generated log along with the overall validation results.

[1]: https://zenodo.org/record/7268458#.Y2jiruzMK3J
[2]: https://github.com/nagaraj-bahubali/author-name-disambiguation-using-mcmc/tree/main/data/input
[3]: https://github.com/nagaraj-bahubali/author-name-disambiguation-using-mcmc/tree/main/data/input/demo
[4]: https://github.com/nagaraj-bahubali/author-name-disambiguation-using-mcmc/blob/main/src/config.py
[5]: https://github.com/nagaraj-bahubali/author-name-disambiguation-using-mcmc/tree/main/data/output/demo_output
