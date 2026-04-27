# Anonymous FE-Compatible Neural Inference Repository

This repository contains the anonymized research code for reproducing the experiments reported in our paper on "Enabling Functional Encryption-Based Inference on Residual Architectures".

The main goal of this repository is to support reproducibility of the paper's experimental results, including:

- replacing the privacy-degenerate identity branch in residual networks with FE-admissible linear transformations;
- applying dimensionality constraints to reduce structural reconstruction risk;
- executing the proposed architecture under an FE-compatible inference pipeline;
- evaluating the accuracy, runtime, and cryptographic execution characteristics on public benchmark datasets.

This repository is provided for **academic research and reproducibility purposes only**.


## 1. Installation

### Environment

1) Python 환경을 구축하고 필요한 라이브러리를 설치합니다.
```bash
# conda create -n fe_inference python=[x.xx]
# conda activate fe_inference
# [pytorch 등 필요한 라이브러리 설치 방법]

2) FE 관련 C파일을 컴파일해야 합니다.
```bash
makefile에 본인의 GPU에 맞는 sm_xx를 입력합니다.
# cd fe/
# make


## 2. Data Preparation

This repository does **not** redistribute raw datasets.

The experiments use publicly available benchmark datasets:

- CIFAR-10
- Tiny-ImageNet

Users should download the datasets from their official sources and comply with the original dataset licenses, citation requirements, and terms of use.

다운로드한 데이터세트의 경로가 py 파일 내 경로와 동일한지 확인하세요


## 3. Reproducing the Main Experiments

This section provides the commands for reproducing the main experimental results reported in the paper.

### 3.1 Train Baseline ResNet-50

These models are trained in plaintext and do not perform functional-encryption execution during training. 
For comparison with the FE-compatible pipeline, we train/evaluate variants corresponding to split points 1--4 and the dimensionality-constrained architecture used by the proposed method.
FE-Compatible ResNet-50은 평문으로 학습된 정보를 활용합니다.

[Split, 평문 모델의 학습 명령어]
# python ~~~.py ~~~~~~
Here, x specifies the split position, where x ∈ {1,2,3,4}.

### 3.2 Evaluation

[Split, 평문 모델의 추론 명령어]
# python ~~~.py ~~~~~~
Here, x specifies the split position, where x ∈ {1,2,3,4}.


### 3.3 FE-Compatible Inference

The FE-compatible inference pipeline includes integer-domain encoding, cryptographic encryption/decryption operations, and stage-wise execution of the proposed architecture.

[Split 함수암호 모델의 추론 명령어]
# python ~~~.py ~~~~~~
Here, x specifies the split position, where x ∈ {1,2,3,4}.


## 4. External Assets, Licenses, and Terms

This repository uses or references the following external assets. Raw datasets are not redistributed in this repository. Users are responsible for downloading datasets from their official sources and complying with the original licenses and terms of use.

| Asset | Use in this work | Source / Citation | License / Terms |
|---|---|---|---|
| CIFAR-10 | Image classification experiments | Official CIFAR-10 dataset; Krizhevsky, "Learning Multiple Layers of Features from Tiny Images" | This repository does not redistribute the raw CIFAR-10 dataset. Users should download it from the official source and comply with the applicable terms. |
| Tiny-ImageNet | Image classification experiments | Tiny-ImageNet / ImageNet-derived dataset | This repository does not redistribute raw Tiny-ImageNet images. Users should obtain the dataset from the appropriate source and comply with the applicable ImageNet/Tiny-ImageNet terms of use. |
| ResNet-50 architecture | Backbone architecture for CIFAR-10 and Tiny-ImageNet experiments | He et al., "Deep Residual Learning for Image Recognition" | Original architecture cited in the paper. |
| ResNet-50 implementation reference | Implementation reference for the ResNet-50 backbone | `bubbliiiing/faster-rcnn-pytorch` | MIT License. Our implementation was developed with reference to this MIT-licensed implementation and substantially modified for FE-compatible inference, including identity-branch replacement, dimensionality constraints, and integer-domain execution. |
| RLWE-IPFE | Functional encryption implementation/component used in the FE-compatible inference pipeline | `fentec-project/IPFE-RLWE` | MIT License. The original copyright and license notice are preserved where applicable. |


## 5. Responsible Release and Intended Use

This repository is released for academic research and reproducibility purposes.

The code is intended to support the experimental results reported in the paper on FE-compatible neural inference and architecture--cryptography co-design.

The implementation is **not** intended for:

- production deployment;
- security-critical systems;
- processing sensitive personal data without independent review;
- making unverified claims of cryptographic security.

The cryptographic routines and Python/C integration are provided to reproduce the experimental pipeline. They should not be treated as an audited production-grade cryptographic library.

The paper identifies a privacy-degenerate identity branch in FE-based inference and evaluates architecture-level alternatives that are more compatible with functional encryption. The proposed method should be interpreted within the scope and limitations stated in the paper.

During anonymous review, issues or concerns can be reported through the conference review process or the anonymized repository issue tracker.


## 6. Limitations

This repository supports the experiments reported in the paper, but the following limitations should be noted:

- The implementation is research code and has not undergone production-level security auditing.
- The security analysis focuses on structural reconstruction resistance under dimensionality constraints.
- Formal guarantees against adaptive or learning-based attacks are outside the current scope.
- Current FE schemes impose computational and numerical constraints, including integer-domain encoding and cryptographic overhead.
- The reported accuracy results are intended to characterize the utility impact of FE-compatible architectural changes, not to claim statistically significant accuracy superiority.
- The reported runtime results characterize the cost of the FE-compatible pipeline, which is largely determined by repeated cryptographic operations.


## 7. License

The code in this repository is released under the license provided in the `LICENSE` file.

If this repository includes or adapts MIT-licensed components, the original copyright and license notices are preserved where applicable.

## 9. Contact

During anonymous review, please use the anonymized repository issue tracker or the official conference review process.

