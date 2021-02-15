# P3CMQA

P3CMQA is a protein model quality assessment tool using 3DCNN.

A paper about this method is in preparation.

The web server version is available at this [page](http://www.cb.cs.titech.ac.jp/p3cmqa).

If you want to use a local version, please download this repository.



## Requirement

1. [chainer](https://chainer.org/)  version 7.7.0 (Deep learning tool)

    ```bash
    $ pip install chainer
    ```

    If you have a GPU, you can make predictions faster. Please Install [cupy](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy) based on your CUDA version. 

    For example, If the CUDA version is 10.2.

    ```bash
    $ pip install cupy-cuda102
    ```
   
2. [Biopython](https://biopython.org/)

    ```bash
    $ pip install biopython
    ```

3. [Prody](http://prody.csb.pitt.edu)

    ```bash
    $ pip install prody
    ```

4. [PSI-BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download) (to generate PSSM)

    Export PATH to BLAST to use ```psiblast```

5. [SCRATCH-1D (SSpro, ACCpro20)](http://download.igb.uci.edu/#sspro) (to predict SS and RSA)

    ```bash
    $ wget http://download.igb.uci.edu/SCRATCH-1D_1.2.tar.gz
    $ tar -xvzf SCRATCH-1D_1.2.tar.gz
    $ cd SCRATCH-1D_1.2
    $ perl install.pl
    ```

    Export PATH to SCRATCH-1D to use ```run_SCRATCH-1D_predictors.sh```

6. [EMBOSS](http://emboss.sourceforge.net/download/#Stable) (to align pdb and fasta)

    ```bash
    $ wget ftp://emboss.open-bio.org/pub/EMBOSS/EMBOSS-6.6.0.tar.gz
    $ tar -xvzf EMBOSS-6.60.tar.gz
    $ cd EMBOSS-6.60
    $ ./configure --prefix=hoge
    $ make
    $ make install
    ```

    Export PATH to EMBOSS package to use ```needle```

7. [Scwrl4](http://dunbrack.fccc.edu/SCWRL3.php/#installation) (to optimize sidechain) **Not essential**

    Export PATH to SCWRL4 to use ```Scwrl4```

   

## Preparation

1. Clone this repository

    ```bash
    $ git clone this_repository
    ```

2. Download pre-training model from [here](http://www.cb.cs.titech.ac.jp/~takei/P3CMQA/trained_model.npz) and put it under P3CMQA directory.

    ```bash
    $ cd P3CMQA
    $ wget http://www.cb.cs.titech.ac.jp/~takei/P3CMQA/trained_model.npz
    $ ls
    src data train trained_model.npz
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



## Usage

* ## Preprocess

   ```bash
   $ python preprocess.py -f ../data/fasta/sample.fasta -d path/to/uniref90/uniref90 -n num_thread
   ```

   Use `-f` to specify the fasta file path, `-d` to specify the uniref90 database path and `-n` to specify the number of thread to use (default=1).

   Then You can get ```sample.pssm```,``` sample.ss``` and ```sample.acc20``` under ```data/profile/sample```.

* ## Side Chain optimization using `Scwrl4`   (**optional**)

   ```bash
   $ Scwrl4 -i sample_1.pdb -o sample_1.pdb
   ```

* ## Prediction

   * ### **If you want to predict multiple model structure for one target**

      ```bash
      $ python predict.py -d ../data/pdb/sample -f ../data/fasta/sample/fasta
      ```

      Use `-d` to specify the pdb directory path and `-f` to specify the fasta file path.

      The results are written under ```data/score/sample``` .
      You will get a file with the global score of all model structures and files with the scores of each residue for each model structure.

      In this example, you will get `data/score/sample/sample.csv`, `data/score/sample/sample_1.txt` and `data/score/sample/sample_2.txt`.


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

      The results are written in   `path/to/dir/sample.csv`,  `path/to/dir/sample_1.txt` and `path/to/dir/sample_2.txt`.

      

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



## Output format

* ### For each model structure

   ```txt
   # Model name : sample_1.pdb
   # Model Quality Score : 0.4236
   Resid	Resname	Score
   1	SER	0.04631
   2	ASN	0.06995
   3	ALA	0.07282
   		:
   ```

    For each model structure, the above text file is generated.

    The first line shows the name of the model structure and the second line shows the global score (score for the whole model structure). Global score ranges from 0 to 1, with the closer the score is to 1, the more similar it is to the natural structure.

    The third and subsequent lines indicate the residue number, residue name, and predicted local score (score for each residue). Local score ranges from 0 to 1, as well as the global score.

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
    sample_1, 0.4236
    sample_2, 0.3452
    ```

   For all model structures, the above csv file is generated.

   The first column shows the name of the model structure and the second column shows the global score.



## Training and test data

You can download the training data from here [training_CASP7-10.tar.gz](http://www.cb.cs.titech.ac.jp/p3cmqa/training_casp7-10.tar.gz) .

You can also download the test data from here [test_CASP11-13.tar.gz](http://www.cb.cs.titech.ac.jp/p3cmqa/test_casp11-13.tar.gz) .

Please note that the download size is large.



## Reference

1. R. Sato and T. Ishida, “Protein model accuracy estimation based on local structure quality assessment using 3D convolutional neural network,” *PLoS One*, vol. 14, no. 9, p. e0221347, 2019. Available from:https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0221347
2. P. Rice, I. Longden, and A. Bleasby, “EMBOSS: The European Molecular Biology Open Software Suite,” *Trends Genet.*, vol. 16, no. 6, pp. 276–277, Jun. 2000. Available from: http://europepmc.org/article/MED/10827456
3. G. G. Krivov, M. V. Shapovalov, and R. L. Dunbrack, “Improved prediction of protein side-chain conformations with SCWRL4,” *Proteins Struct. Funct. Bioinforma.*, vol. 77, no. 4, pp. 778–795, 2009. Available from: https://pubmed.ncbi.nlm.nih.gov/19603484/
4. D. J. Lipman *et al.*, “Gapped BLAST and PSI-BLAST: a new generation of protein database search programs,” *Nucleic Acids Res.*, vol. 25, no. 17, pp. 3389–3402, 1997. Available from: https://pubmed.ncbi.nlm.nih.gov/9254694/ 
5. C. N. Magnan and P. Baldi, “SSpro/ACCpro 5: Almost perfect prediction of protein secondary structure and relative solvent accessibility using profiles, machine learning and structural similarity,” *Bioinformatics*, vol. 30, no. 18, pp. 2592–2597, 2014. Available from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4215083/
