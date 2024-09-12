# HuatuoGPT-Vision, Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale


<div align="center">
<h5>
  📃 <a href="https://arxiv.org/abs/2406.19280" target="_blank">Paper</a>  • 🖥️ <a href="https://vision.huatuogpt.cn/#/" target="_blank">Demo</a>
</h5>
</div>
 
<div align="center">
<h4>
  📚 <a href="https://huggingface.co/datasets/FreedomIntelligence/PubMedVision" target="_blank">PubMedVision</a> 
</h4>
</div>

<div align="center">
<h4>
  🤗 <a href="https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-34B" target="_blank">HuatuoGPT-Vision-34B</a>  • 🤗 <a href="https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B">HuatuoGPT-Vision-7B</a> 
</h4>
</div>

## ✨ Updates
- [06/28/2024]: We released our medical MLLMs, including [HuatuoGPT-Vision-34B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-34B) and [HuatuoGPT-Vision-7B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B).
- [06/26/2024]: We released [PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision), a **1.3M** high-quality medical VQA dataset for injecting medical visual knowledge.

## 🩻 PubMedVision
- **PubMedVision** is a large-scale, high-quality medical VQA dataset, constructed from the image-text pairs from PubMed and reformatted using GPT-4V.

|                             | # Data     | Download |
| ------------------ | ---------- | ------------- | 
| **PubMedVision Dataset**   | **1,294,062** | [HF Link](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision) |

- **PubMedVision** could significantly improve the medical multimodal capabilities of MLLMs such as LLaVA-v1.5. 

|                                         | **VQA-RAD** | **SLAKE** | **PathVQA** | **PMC-VQA** |
| --------------------------------------- | ----------- | --------- | ----------- | ----------- |
| LLaVA-v1.6-34B                          | 58.6        | 67.3      | 59.1        | 44.4        |
| LLaVA-v1.5-LLaMA3-8B                    | 54.2        | 59.4      | 54.1        | 36.4        |
| LLaVA-v1.5-LLaMA3-8B + **PubMedVision** | **63.8**    | **74.5**  | **59.9**    | **52.7**    |

|                           | **OmniMedVQA**  | **MMMU Health & Medicine (Test Set)** |
| ----------------------------------- | ------------ | -------------------------- |
| LLaVA-v1.6-34B                      | 61.4         | 48.8                       |
| LLaVA-v1.5-LLaMA3-8B                | 48.8         | 38.2                       |
| LLaVA-v1.5-LLaMA3-8B + **PubMedVision** | **75.1**    | **49.1**                   |

## 👨‍⚕️ HuatuoGPT-Vision
HuatuoGPT-Vision is our medical multimodal LLMs, built on **PubMedVision**.

### Model Access
Our model is available on Huggingface in two versions:
|                 | Backbone           | Checkpoint                                                                            |
|----------------------|--------------------|---------------------------------------------------------------------------------------|
| **HuatuoGPT-Vision-7B**  | Qwen2-7B           | [HF Link](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B)             |
| **HuatuoGPT-Vision-34B** | Yi-1.5-34B         | [HF Link](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-34B)            |

### Model Usage

- **Command Line Interface**

Chat via the command line:
```bash
python cli.py --model_dir path-to-huatuogpt-vision-model
```

- **Model Inference**

Inference using our ChatBot:
```python
query = 'What does the picture show?'
image_paths = ['image_path1']

from cli import HuatuoChatbot
bot = HuatuoChatbot(path-to-huatuogpt-vision-model)
output = bot.inference(query, image_paths)
print(output) # Prints the output of the model
```

### Performance of Medical Multimodal
|                             | **VQA-RAD** | **SLAKE** | **PathVQA** | **PMC-VQA** |
|-----------------------------|-------------|-----------|-------------|-------------|
| LLaVA-Med-7B                   | 51.4        | 48.6      | 56.8        | 24.7        |
| LLaVA-v1.6-34B              | 58.6        | 67.3      | 59.1        | 44.4        |
| **HuatuoGPT-Vision-7B**         | 63.7        | 76.2     | 57.9        | 54.3        |
| **HuatuoGPT-Vision-34B**        | **68.1**    | **76.9**  | **63.5**    | **58.2**    |

|                           | **OmniMedVQA**  | **MMMU Health & Medicine (Test Set)** |
|---------------------------|-----------------|---------------------------------------|
| LLaVA-Med-7B                 | 44.5            | 36.9                                  |
| LLaVA-v1.6-34B            | 61.4            | 48.8                                  |
| **HuatuoGPT-Vision-7B**        | 74.0            | 50.6                                  |
| **HuatuoGPT-Vision-34B**      | **76.9**        | **54.4**                               |

## 🩺 HuatuoGPT Series 

Explore our HuatuoGPT series:
- [**HuatuoGPT**](https://github.com/FreedomIntelligence/HuatuoGPT): Taming Language Models to Be a Doctor
- [**HuatuoGPT-II**](https://github.com/FreedomIntelligence/HuatuoGPT-II): One-stage Training for Medical Adaptation of LLMs
- [**HuatuoGPT-Vision**](https://github.com/FreedomIntelligence/HuatuoGPT-Vision): Injecting Medical Visual Knowledge into Multimodal LLMs at Scale


## Citation
```
@misc{chen2024huatuogptvisioninjectingmedicalvisual,
      title={HuatuoGPT-Vision, Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale}, 
      author={Junying Chen and Ruyi Ouyang and Anningzhe Gao and Shunian Chen and Guiming Hardy Chen and Xidong Wang and Ruifei Zhang and Zhenyang Cai and Ke Ji and Guangjun Yu and Xiang Wan and Benyou Wang},
      year={2024},
      eprint={2406.19280},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.19280}, 
}
```
