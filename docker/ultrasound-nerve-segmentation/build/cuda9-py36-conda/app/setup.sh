
#create env
/opt/conda/bin/conda create -q -v -y --name py36-tf18-gpu

# 重新进入虚拟环境
source activate
# 退出虚拟环境
conda deactivate

#activate
source activate py36-tf18-gpu

#install requirements.txt
/opt/conda/bin/conda install --yes --file ./requirements.txt

#cleanup
#/opt/conda/bin/conda clean -p
