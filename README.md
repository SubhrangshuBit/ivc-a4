# Assignment 4 - Semantic Segmentation
#### Team: 
  * Nazia Tasnim
  * Subhrangshu Bit

##### Preliminary Instructions
1. Download the dataset, place it under this directory and unzip (Done)

2. Install required packages for running our code (using conda (miniconda on SCC) is recommended) (Done)
conda create --name myenv python=3.8 -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install pandas
pip install numpy
pip install pillow

3. Implement required modules and debug on cpu mode
python train.py

4. Once you're done with step. 3, submit the batch job to run on GPU. (more details introduced in 'scc_quick_setup_guide.txt')
qsub job_script
