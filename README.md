# HuatuoGPT-Vision, Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale


<div align="center">
<h5>
  📃 <a href="https://arxiv.org/abs/2406.19280" target="_blank">Paper</a>  • 🖥️ <a href="#" target="_blank">Demo (coming)</a>
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

## 📚 PubMedVision
We leveraged GPT-4V to reformat the image-text pairs from PubMed, creating a large-scale, high-quality medical VQA dataset, PubMedVision. PubMedVision can be found [here](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision).

PubMedVision could significantly improve the medical multimodal capabilities of MLLMs such as LLaVA-v1.5. Experimental results:

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
