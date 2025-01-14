# ROSITA

Official code implementation for the paper **"Efficient Open-set Single image Test Time Adaptation of Vision Language Models"**

![image](docs/ROSITA.png)

> <p align="justify"> <b> <span style="color: green;">ROSITA framework</span></b>:
>     Adapting models to dynamic, real-world environments characterized by shifting data distributions and unseen test scenarios is a critical challenge in deep learning. In this paper, we consider a realistic and challenging Test Time Adaptation setting, where a model must continuously adapt to test samples that arrive sequentially, one at a time, while distinguishing between known and unknown classes. Existing Test Time Adaptation methods fail to handle this setting due to their reliance on closed-set assumptions or batch processing, making them unsuitable for real-world open-set scenarios. We address this limitation by establishing a comprehensive benchmark for {\em Open-set Single image Test Time Adaptation using Vision-Language Models}. Furthermore, we propose ROSITA, a novel framework that leverages dynamically updated feature banks to identify reliable test samples and employs a contrastive learning objective to improve the separation between known and unknown classes. Our approach effectively adapts models to domain shifts for known classes while rejecting unfamiliar samples. Extensive experiments across diverse real-world benchmarks demonstrate that ROSITA sets a new state-of-the-art in open-set TTA, achieving both strong performance and computational efficiency for real-time deployment.

---

### Installation

Please follow the instructions at [INSTALL.md](docs/INSTALL.md) to setup the environment.

### Dataset preparation

Please follow the instructions at [DATASETS.md](docs/DATASETS.md) to prepare the datasets.

### Experiments

Please follow the instructions at [RUN.md](docs/RUN.md) to run the experiments.

---

### Acknowledgements

The baselines have been established with the help these repositories:

1. [TPT](https://github.com/azshue/TPT)
2. [PromptAlign](https://github.com/jameelhassan/PromptAlign)
3. [OWTTT](https://github.com/Yushu-Li/OWTTT)
