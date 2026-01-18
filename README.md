# RPlanRAG
[ICASSP 2026] The implementation code for our paper: Reasoner-Assisted Planning: Enhance the Ability of Graph-RAG to Handle Complex Questions

## 1 Start

### 1 Clone our project

```bash
git clone https://github.com/icvplayer/RPlanRAG.git
```

### 2 Download the evaluation dataset

Our evaluation dataset utilizes Longbench[1] and LVEval[2], and the pertinent data can be accessed on the HuggingFace repository ([icvplayer/RPlanRAG](https://huggingface.co/datasets/icvplayer/RPlanRAG)).

### 3 Train the planning model

We use LlamaFactory([hiyouga/LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ LLMs & VLMs (ACL 2024)](https://github.com/hiyouga/LLaMA-Factory)) to train the Planning model. The specific parameter information is saved in the "train_Plan_model.sh" file.

Here, the training set we are using is stored in the file "Plan_training_data_all.json" ([icvplayer/RPlanRAG](https://huggingface.co/datasets/icvplayer/RPlanRAG)).

### 4 Train the Reasoner

Use LlamaFactory (https://github.com/hiyouga/LlamaFactory) to train the Reasoner. The specific parameter information is saved in the "train_Reasoner.sh" file of the current repository. The training set we use is stored in the "Summary_training_data_all.json" file ([icvplayer/RPlanRAG](https://huggingface.co/datasets/icvplayer/RPlanRAG)).

Based on the previous step, train the inference engine. The relevant code is located in "freestyleRAG/My_train_RL.py". You need to randomly extract 5000 or more instances from "Summary_training_data_all.json" for RL training.

Before conducting RL training, the following environment configurations need to be completed:

```bach
# Create a Python environment
conda create -n grpo python=3.10
# Activate the environment
conda activate grpo

cd freestyleRAG

pip install -r requirements_RL.txt

python My_train_RL.py
```

### 5 Eval

In order to run the following code, you need to install "ollama"[Ollama](https://ollama.com/).

#### 5.1 Eval LongBench

The evaluation dataset used in this part is located at: "longbench.json"

After obtaining the Planning model and Reasoner, we conducted the evaluation. To evaluate hotpotqa, 2wiki and musique in LongBench, we first created the environment and then ran the relevant code:


```bash
# Create a Python environment
conda create -n rplanrag python=3.10
# Activate the environment
conda activate rplanrag

pip install -r requiremens.txt

cd freestyleRAG

export api_key=your_api_key

# for hotpotqa:
export data_name=hotpotqa

python eval/eval.py

# for 2wikimqa:
export data_name=2wikimqa

python eval/eval.py

# for musique:
export data_name=musique

python eval/eval.py
```

#### 5.2 Eval LVEval

The evaluation dataset used in this part is located at: "hotpotwikiqa_mixup_16k.jsonl"

To evaluate LVEval, based on the aforementioned environment, execute the following code:


```bash
# for HW-Mix:
export data_name=hotpotwikiqa_mixup_16k

python eval/eval_lveval.py
```


[1] Bai, Y.; Lv, X.; Zhang, J.; Lyu, H.; Tang, J.; Huang, Z.; Du, Z.; Liu, X.; Zeng, A.; Hou, L.; et al. 2023. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508.

[2] Yuan, T.; Ning, X.; Zhou, D.; Yang, Z.; Li, S.; Zhuang, M.; Tan, Z.; Yao, Z.; Lin, D.; Li, B.; et al. 2024. Lv-eval: A balanced long-context benchmark with 5 length levels up to 256k. arXiv preprint arXiv:2402.05136.
