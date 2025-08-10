# Composite Blood Cell Dataset & Segmentation Pipeline

## 📌 Overview
This repository contains the code, data, and methodology for our **Synthetic Composite Blood Cell Generation and Segmentation** project.  
The pipeline enables:
- **Class-specific segmentation** of white blood cell (WBC) types from raw microscopic images.
- **Synthetic composite image generation** in both *grid* and *background-based* styles.
- **Annotation handling** for detection/segmentation tasks.
- **Dataset augmentation** to improve robustness in downstream models.

We provide:
- All Python scripts (`.py`) for preprocessing, segmentation, and composite generation.
- Processed datasets (generated composites and masks).
- Instructions to reproduce experiments.

> ⚠ **Note**: The raw PBC and ALL-IDB datasets do not belong to us. Please download them from their official sources and cite them in any work.
>1. @inproceedings{acevedo2018pbc,
  title={A dataset of microscopic peripheral blood cell images for development of automatic recognition systems},
  author={Acevedo, A. and Merino, A. and Alférez, J. and Molina, A. and Puigdomènech, A.},
  booktitle={Proc. Int. Conf. Pattern Recognit. (ICPR)},
  pages={3697--3702},
  year={2018}
}

> @inproceedings{labati2011allidb,
  title={ALL-IDB: The acute lymphoblastic leukemia image database for image processing},
  author={Labati, R. R. and Piuri, V. and Scotti, F.},
  booktitle={Proc. IEEE Int. Conf. Image Process. (ICIP)},
  pages={2045--2048},
  year={2011}
}


---

