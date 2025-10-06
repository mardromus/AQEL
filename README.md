# 🧠 Adaptive Quantum Ensemble Learning (AQEL) – Enhanced CMO-QF Implementation

This repository presents an **enhanced Adaptive Quantum Ensemble Learning (AQEL)** model based on **CMO-QF (Cyclic Multi-Optimizer Quantum Framework)**.  
It integrates **feature encoding, adaptive variational circuits, and hybrid classical-quantum optimization** using [PennyLane](https://pennylane.ai/).

---

## 🚀 Overview

The project demonstrates how quantum machine learning (QML) models can adapt their architecture dynamically during training — increasing circuit depth, changing optimizers, and improving expressivity based on real-time loss behavior.

It combines:
- 🧩 **Adaptive Quantum Feature Mapping (AQFM)**
- 🔄 **Neural Adaptive Variational Quantum Circuit (NAVQC)**
- ⚙️ **Cyclic Multi-Optimizer Quantum Framework (CMO-QF)**
- 🧠 **Custom Data Generation & Preprocessing Pipeline**

---

## 🧬 Features

✅ Custom 2D data generation (make_moons variant)  
✅ Adaptive scaling and train-test split (no sklearn dependency)  
✅ Dynamic optimizer cycling: `BFGS`, `L-BFGS-B`, `Powell`  
✅ Early stopping and learning rate decay  
✅ Architecture expansion (adds quantum layers when loss stagnates)  
✅ Built-in visualization of training progress and decision boundaries  
✅ Supports 2D datasets (extendable to higher dimensions)  

---

## 🧪 Quantum Components

### 1. **AQFM (Adaptive Quantum Feature Map)**
Encodes classical data into quantum states using hybrid rotation layers (`RY`, `RZ`) with cyclic entanglement.  
Handles feature–qubit mismatch via modulo mapping.

### 2. **NAVQC (Neural Adaptive Variational Quantum Circuit)**
A multi-layer variational circuit leveraging `qml.StronglyEntanglingLayers` for expressive quantum learning.  
The number of layers can increase dynamically during training.

### 3. **CMO-QF (Cyclic Multi-Optimizer Quantum Framework)**
Cycles through optimizers (BFGS, L-BFGS-B, Powell) in short blocks for robust convergence.  
Performs adaptive reinitialization and regularization to prevent overfitting.

---

## 🧰 Dependencies

Make sure the following Python packages are installed:

```bash
pip install numpy matplotlib pennylane scipy scikit-learn
