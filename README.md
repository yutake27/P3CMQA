# P3CMQA <!-- omit in toc --> 
P3CMQA is a protein model quality assessment tool using 3DCNN.

A paper about this method is in preparation.

The web server version is available at this [page](http://www.cb.cs.titech.ac.jp/p3cmqa).

If you want to use a local version, please download this repository.

## Table of Contents <!-- omit in toc --> 
- [Requirements](#requirements)
  - [Software and databases that must be installed](#software-and-databases-that-must-be-installed)
  - [Requirements when not using Docker images](#requirements-when-not-using-docker-images)
- [Sample data directory structure](#sample-data-directory-structure)
- [Usage without Docker](#usage-without-docker)
  - [1. Preprocess](#1-preprocess)
  - [2. Side Chain optimization using `Scwrl4`   (**optional**)](#2-side-chain-optimization-using-scwrl4---optional)
  - [3. Prediction](#3-prediction)
- [Usage with docker](#usage-with-docker)
  - [1. Pull docker image](#1-pull-docker-image)
  - [2. Preprocess](#2-preprocess)
  - [3. Side Chain optimization using `Scwrl4`   (**optional**)](#3-side-chain-optimization-using-scwrl4---optional)
  - [4. Prediction with docker](#4-prediction-with-docker)
- [Output format](#output-format)
- [Training and test data](#training-and-test-data)
- [Reference](#reference)

## Requirements
In this method, there are two main processes: preprocessing and prediction.

In the preprocessing part, homology search and local structure prediction are performed, and in the prediction part, inferences are performed using the files obtained in the preprocessing.

For the prediction part, we have released a Dockerfile and the Docker images to simplify the construction of the environment.

The preprocessing part uses other large software and databases, and it was difficult to include it in the Docker image, so this part is not included in the Docker image.

Therefore, the preprocessing part should be installed regardless of whether you use the Docker image or not.

### Software and databases that must be installed
1. Python
    
    version 3.7 or later

2. [PSI-BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download) (to generate PSSM)

    If you do not have installed PSI-BLAST, please install it. You can download blast+ package from [here](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/)．

    After the download is complete, make sure to export the PATH so that the `psiblast` command can be used.

    ```sh
    # example
    export PATH=$PATH:/path/to/blast/ncbi-blast-2.11.0+/bin
    ```


3. uniref90 database (to generate PSSM using `psiblast`)

    If you do not have downloaded uniref90 database, please download `uniref90.fasta.gz` from [here](https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref90/).
    Please note that it takes quite a long time to download.

    After downloading, please unzip `uniref90.fasta.gz`.

    Then, create a database with the command `makeblastdb` included in the blast package as follows.

    ```sh
    makeblastdb -in uniref90.fasta -dbtype prot -out uniref90
    ```

    When the database creation is complete, please name the directory where the database is saved.

    ```sh
    export uniref90=/path/to/uniref90_directory
    ```

4. [SCRATCH-1D (SSpro, ACCpro20)](http://download.igb.uci.edu/) (to predict SS and RSA)

    Please download and install SCRATCH-1D using the below commands.

    ```sh
    wget http://download.igb.uci.edu/SCRATCH-1D_1.2.tar.gz
    tar -xvzf SCRATCH-1D_1.2.tar.gz
    cd SCRATCH-1D_1.2
    perl install.pl
    ```

    After installation, please export the PATH to use `run_SCRATCH-1D_predictors.sh`.

    ```sh
    export PATH=$PATH:path/to/SCRATCH-1D_1.2/bin
    ```

    The `blast-2.2.26` included in `SCRATCH-1D/pkg` is a 32-bit Linux version and may cause errors.

    `SCRATCH-1D_1.2/README.txt` contains instructions on how to replace it with a 64-bit version, if necessary, replace it.

    The 64-bit version is available at https://ftp.ncbi.nlm.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26.

5. Clone this repository and download the pre-trained model

    The trained model is not included in the GitHub repository, so you have to download it and put it directly under the repository.

    We provide the pre-training model at [here](http://www.cb.cs.titech.ac.jp/~takei/P3CMQA/trained_model.npz).

    ```sh
    $ git clone https://github.com/yutake27/P3CMQA.git
    $ cd P3CMQA
    $ wget http://www.cb.cs.titech.ac.jp/~takei/P3CMQA/trained_model.npz
    $ ls
    LICENSE  README.md  data  result  src  trained_model.npz
    ```

6. [Scwrl4](http://dunbrack.fccc.edu/SCWRL3.php/#installation) (to optimize sidechain) **Not essential, but optional**

    Please export PATH to SCWRL4 to use `Scwrl4`

### Requirements when not using Docker images


1. [chainer](https://chainer.org/)  version 7.7.0 (Deep learning tool)

    ```bash
    $ pip install chainer
    ```

    If you have a GPU, you can make predictions faster. Please Install [cupy](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy) based on your CUDA version. 

    For example, If the CUDA version is 10.2, run the below command to install cupy.

    ```bash
    $ pip install cupy-cuda102
    ```
   
2. [Biopython](https://biopython.org/) version 1.7.8

    ```bash
    $ pip install biopython
    ```

3. [Prody](http://prody.csb.pitt.edu) version 2.0

    ```bash
    $ pip install prody
    ```


## Sample data directory structure

```txt
data
├── pdb
│   └── sample
│        ├── sample_1.pdb
│        └── sample_2.pdb
├── fasta
│   └── sample.fasta
└── profile
   └── sample
         ├── sample.pssm
         ├── sample.ss
         └── sample.acc20
```



## Usage without Docker


### 1. Preprocess
  ```bash
  $ python preprocess.py -f ../data/fasta/sample.fasta -d $uniref90/uniref90 -n num_thread
  ```

Use `-f` to specify the fasta file path, `-d` to specify the uniref90 database path, and `-n` to specify the number of threads to use (default=1).

Then You can get `sample.pssm`,` sample.ss` and `sample.acc20` under `data/profile/sample`.

### 2. Side Chain optimization using `Scwrl4`   (**optional**)

   ```bash
   $ Scwrl4 -i sample_1.pdb -o sample_1.pdb
   ```

### 3. Prediction

* ### **If you want to predict multiple model structure for one target**

    ```bash
    $ python predict.py -d ../data/pdb/sample -f ../data/fasta/sample.fasta
    ```

    Use `-d` to specify the pdb directory path and `-f` to specify the fasta file path.

    The results are written under `data/score/sample`.
    You will get a file with the global score of all model structures and files with the scores of each residue for each model structure.

    In this example, you will get `data/score/sample/sample.csv`.


    If you have a **GPU**,

    ```bash
    $ python predict.py -d ../data/pdb/sample -f ../data/fasta/sample.fasta -g 0
    ```

    Use `-g` to specify the GPU ID (negative value indicates CPU).

    If you want to **specify the output directory**,

    ```bash
    $ python predict.py -d ../data/pdb/sample -f ../data/fasta/sample.fasta -o path/to/dir
    ```

    Use `-o` to specify the directory path where you want to output the results.

    The results are written in   `path/to/dir/sample.csv`.

    If you want to **sepecify the profile directory**,

    ```bash
    $ python predict.py -d ../data/pdb/sample -f ../data/fasta/sample.fasta -p path/to/profile/dir
    ```

    Directory `path/to/profile/dir` should have `sample.pssm`, `sample.ss` and `sample.acc20`.


* ### **If you want to predict single model structure**

    ```bash
    $ python predict.py -i ../data/pdb/sample/samle_1.pdb -f ../data/fasta/sample.fasta
    ```

    Use ```-i``` to specify the pdb file path and `-f` to specify the fasta file path.
    The results are written in ```data/score/sample/sample_1.txt```.

    

    If you have a **GPU**,
    
    ```bash
    $ python predict.py -i ../data/pdb/sample/sample_1.pdb -f ../data/fasta/sample.fasta -g 0
    ```
    
    Use `-g` to specify the GPU ID (negative value indicates CPU).
    
    
    
    If you want to **specify the output directory**, 
    
    ```bash
    $ python predict.py -i ../data/pdb/sample/samle_1.pdb -f ../data/fasta/sample.fasta -o path/to/dir
    ```
    
    Use `-o` to specify the directory path where you want to output the results.
    
    The results are written in ```path/to/dir/sample_1.txt```


## Usage with docker
### 1. Pull docker image
We have released two versions of the image on Dockerhub, a CPU version, and a GPU version. The Dockerhub repository is [here](https://hub.docker.com/repository/docker/yutake27/p3cmqa).

If you do not have a GPU environment, please pull `yutake27/p3cmqa:cpu`.

```sh
docker pull yutake27/p3cmqa:cpu
```

For GPU versions, we have released two types of images for CUDA and cuDNN versions.

The first image is `p3cmqa:cuda11.0-cudnn8`, which supports CUDA 11.0 and cuDNN 8.0, and the other is `p3cmqa:cuda10.2-cudnn7`, which supports CUDA 10.2 and cuDNN 7.0.

Please pull the appropriate image for your version of CUDA and cuDNN.

```
docker pull yutake27/p3cmqa:cuda11.0-cudnn8
```

If you want to use another version of CUDA or cuDNN, please modify the Dockerfile included in the repository and build it. Note that it is possible to perform a prediction without cuDNN.

### 2. Preprocess
```sh
$ python preprocess.py -f ../data/fasta/sample.fasta -d $uniref90/uniref90 -n num_thread
```

Use `-f` to specify the fasta file path, `-d` to specify the uniref90 database path, and `-n` to specify the number of threads to use (default=1).

Then You can get ```sample.pssm```,``` sample.ss``` and ```sample.acc20``` under ```data/profile/sample```.

### 3. Side Chain optimization using `Scwrl4`   (**optional**)

   ```bash
   $ Scwrl4 -i sample_1.pdb -o sample_1.pdb
   ```
### 4. Prediction with docker
There is a python script named `docker_predict.py` to simplify prediction with Docker.

This script starts the container, executes the prediction, and terminates the container, so there is no need to enter docker commands by yourself.

To execute the prediction, please give the name of the Docker image (`Repository:tag`) and optional arguments as in the following command.

```sh
python docker_predict.py yutake27/p3cmqa:cuda11.0-cudnn8 -g 0 -d ../data/pdb/sample -f ../data/fasta/sample.fasta
```

The arguments other than the name of the docker image are the same as without docker, so please check [here](#3.-Prediction).

If you do not use the above script, please run the container and execute the prediction as follows.
```sh
# docker run
docker run -dit --rm -u "$(id -u $USER):$(id -g $USER)" --gpus=0 \
-v /absolute/path/to/P3CMQA:/home \
-v /absolute/path/to/P3CMQA/data/pdb/sample:/home/data/pdb/sample \
-v /absolute/path/to/P3CMQA/data/fasta:/home/data/fasta/ \
-v /absolute/path/to/P3CMQA/data/profile/sample:/home/data/profile/sample \
-v /absolute/path/to/P3CMQA/data/score/sample:/home/data/score/sample \
 --name p3cmqa yutake27/p3cmqa:cuda11.0-cudnn8

# docker exec
docker exec -it -u "$(id -u $USER):$(id -g $USER)" p3cmqa /bin/bash \
-c "cd home/src && python predict.py -d ../data/pdb/sample -f ../data/fasta/sample.fasta -g 0"
```

## Output format

* ### For one model structure

   ```txt
   # Model name : ../data/pdb/sample/sample_1.pdb
   # Model Quality Score : 0.04516
   Resid	Resname	Score
   1	MET	0.13082
   2	ALA	0.17883
   3	ALA	0.28698
   		:
   ```

    For one model structure, the above text file is generated.

    The first line shows the name of the model structure and the second line shows the global score (score for the whole model structure). The global score ranges from 0 to 1, the closer the score is to 1, the more similar it is to the natural structure.

    The third and subsequent lines indicate the residue number, residue name, and predicted local score (score for each residue). The Local score ranges from 0 to 1, as well as the global score.

    You can load this file as a csv file using python in the following way.

   ```python
   import pandas as pd
   df = pd.read_csv('sample_1.txt', sep='\t', header=2)
   >	    Resid Resname Score
   0  1  SER   0.04631
   1  2  ASN   0.06995
   2  3  ALA   0.07282
   ..  ...    ...  ...
   ```

* ### For all model structures

    ```txt
    Model_name, Score
    sample_1, 0.045157402753829956
    sample_2, 0.3176875412464142
    ```

   For all model structures, the above csv file is generated.

   The first column shows the name of the model structure and the second column shows the global score.



## Training and test data

You can download the training data from here [training_CASP7-10.tar.gz](http://www.cb.cs.titech.ac.jp/p3cmqa/training_casp7-10.tar.gz).

You can also download the test data from here [test_CASP11-13.tar.gz](http://www.cb.cs.titech.ac.jp/p3cmqa/test_casp11-13.tar.gz).

Please note that the download size is large.



## Reference
1. Y. Takei and T.Ishida, in preparation.
2. R. Sato and T. Ishida, “Protein model accuracy estimation based on local structure quality assessment using 3D convolutional neural network,” *PLoS One*, vol. 14, no. 9, p. e0221347, 2019. Available from:https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0221347
3. G. G. Krivov, M. V. Shapovalov, and R. L. Dunbrack, “Improved prediction of protein side-chain conformations with SCWRL4,” *Proteins Struct. Funct. Bioinforma.*, vol. 77, no. 4, pp. 778–795, 2009. Available from: https://pubmed.ncbi.nlm.nih.gov/19603484/
4. D. J. Lipman *et al.*, “Gapped BLAST and PSI-BLAST: a new generation of protein database search programs,” *Nucleic Acids Res.*, vol. 25, no. 17, pp. 3389–3402, 1997. Available from: https://pubmed.ncbi.nlm.nih.gov/9254694/
5. C. N. Magnan and P. Baldi, “SSpro/ACCpro 5: Almost perfect prediction of protein secondary structure and relative solvent accessibility using profiles, machine learning and structural similarity,” *Bioinformatics*, vol. 30, no. 18, pp. 2592–2597, 2014. Available from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4215083/
