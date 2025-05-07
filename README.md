## **MolGraph-xLSTM**  
MolGraph-xLSTM is a graph-based dual-level xLSTM framework designed for molecular property prediction. This model improves molecular feature extraction and enhances prediction accuracy for both classification and regression tasks.

![MolGraph-xLSTM Architecture](mol-xlstm.png)

## **Requirements**  
To ensure reproducibility, all dependencies are listed in `requirements.txt`. Below are the tested installation steps for setting up the environment on **Linux** using **Conda and Python 3.10.0**.

## **Installation**  
Clone the repository and set up the Conda environment:  

git clone https://github.com/JasperDurinck/MolGraph-xLSTM.git  
cd MolGraph-xLSTM  

conda create -n molgraph-xlstm python=3.10.0 -y  
conda activate molgraph-xlstm 

pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.0+cu121 torchvision torchaudio  
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html  
pip install -r requirements.txt  

Next install the 2.0.2 version of xLSTM  https://github.com/NX-AI/xlstm/tree/main
(also provided in this repo with a setup file)

## **Conda level install of cuda, slstm compile issue work around (optional setup method for handling specific conda user level environment compile issue)**

bash xLSTM_setup/setup_xLSTM.sh 

cd xLSTM_setup

pip install .

cd ../

(see 'Issue_fix_user_level_conda' dir for setting up path to conda env in cuda_init.py replacement file)

## **Data**
The following datasets are used in our experiments:

| **Dataset**  | **Samples** | **Task Type** |
|-------------|------------|--------------|
| **BACE**    | 1.5k       | Binary Classification |
| **BBBP**    | 2.0k       | Binary Classification |
| **HIV**     | 41.1k      | Binary Classification |
| **ClinTox** | 1.5k       | Binary Classification |
| **Sider**   | 1.4k       | Binary Classification |
| **Tox21**   | 7.8k       | Binary Classification |
| **ESOL**    | 1.1k       | Regression |
| **Freesolv**| 0.6k       | Regression |
| **Lipo**    | 4.2k       | Regression |
| **Caco2**   | 0.9k       | Regression |

Each dataset was split into **training (80%), validation (10%), and test (10%)** subsets.  
All partitioned datasets are located in the `datasets` folder.

##

## **Running Code**
1. Classification  
   python main.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.0001 --batch_size 128 --epochs 100 --trial 44 --dataset bace --num_tasks 1 --classification --num_blocks 2
   --slstm 0 --data_dir "datasets" --num_gc_layers 4 --power 4 --num_dim 128 --num_experts 4 --num_heads 4
2. Regression  
   python main.py --lr_decay_epochs 800 --lr_decay_rate 0.5 --learning_rate 0.0002 --batch_size 128 --epochs 200 --trial 41 --dataset freesolv --num_tasks 1 --num_blocks 2
   --slstm 0 --data_dir "datasets" --num_gc_layers 4  --power 4 --num_dim 128 --dropout 0.5 --mlp_layer 1 --num_experts 8 --num_heads 8

## **License**
This project is licensed under the MIT License.

## 🛠️ Fork Changes


1. **Added missing dependencies and updated versions**  
   Several dependencies were missing from `requirements.txt`, and some specified versions were outdated  
   or unavailable via `pip`. These have been added or updated accordingly.

2. **Remove non-used dependencies** 
   
    There are package import (such as 'unimol_tools' which was used for 'self.geom3d = UniMolRepr(data_type='molecule')', however self.geom3d is never used in the code since it was probably tested as feature and commented out in dataloader.py '#data.geom3d_feature = torch.tensor(unimol_feature[smiles])#torch.tensor(self.geom3d.get_repr(smiles)).squeeze()[0]'). We removed non-used dependencies that are being imported in the original code (and are also not listed as requirements). 

   The original repository included code that used SmilesTokenizer from DeepChem (from deepchem.feat.smiles_tokenizer import SmilesTokenizer), specifically in the load_dataset() method of the PygOurDataset class, which also referenced a missing vocab.txt file (self.tokenizer_simple = SmilesTokenizer('utils/vocab.txt')). However, DeepChem was not listed as a dependency, and the tokenizer functionality was not actually used elsewhere in the code. To prevent errors related to the missing vocab.txt file or the absence of DeepChem, we removed the unused DeepChem-related code entirely from the repository (deepchem was also not specified as a requirement and the vocab was not provided nor mentioned where to obtain it form).

3. **Updated code for compatibility with newer dependency versions**  

4. **Flexible SMILES column and max_len specification**  
   The original code assumes the dataset includes a `'smiles'` column, but some datasets (e.g., BRACE)  
   use different column names like `'mol'` (cause of error in original code). This fork allows the user to specify the correct column name and also adjust max_len. #TODO improve further with respect to args  

5. **Refactored code, created a more modular installable package**  
   For simpler integration into custom pipelines.
   