# Physics-Enhanced Graph Neural Networks For Soft Sensing in Industrial Internet of Things

This paper has been officially accepted for publication in the **IEEE Internet of Things Journal**.
https://ieeexplore.ieee.org/document/10638707
___

**Abstract:** 

The Industrial Internet of Things (IIoT) is reshaping manufacturing, industrial processes, and infrastructure management. By fostering new levels of automation, efficiency, and predictive maintenance, IIoT is transforming traditional industries into intelligent, seamlessly interconnected ecosystems. However, achieving highly reliable IIoT can be hindered by factors such as the cost of installing large numbers of sensors, limitations in retrofitting existing systems with sensors, or harsh environmental conditions that may make sensor installation impractical. Soft (virtual) sensing leverages mathematical models to estimate variables from physical sensor data, offering a solution to these challenges. Data-driven and physics-based modeling are the two main methodologies widely used for soft sensing. The choice between these strategies depends on the complexity of the underlying system, with the data-driven approach often being preferred when the physics-based inference models are intricate and present challenges for state estimation. However, conventional deep learning models are typically hindered by their inability to explicitly represent the complex interactions among various sensors. To address  this limitation, we adopt  Graph Neural Networks (GNNs), renowned for their ability to effectively  capture the complex  relationships between sensor measurements. In this research, we propose physics-enhanced GNNs, which integrate principles of physics into graph-based methodologies. This is achieved by augmenting additional nodes in the input graph derived from the underlying characteristics of the physical processes. Our evaluation of the proposed methodology on the case study of district heating networks reveals significant improvements over purely data-driven GNNs, even in the presence of noise and parameter inaccuracies.
___ 


## Data
1. The synthetic dataset is included in the `Data` folder.
2. `Data` Folder includes raw TESPY simulator + extracted CSV file for both raw sensor values and characteristics of physical process.
3. You can customize and change network topology by `DHN_Topology_TESPY.py`

___


## Getting Started:
You just need to run `main.ipynb` to reproduce the result.


## Citing our paper

If this work was useful to you, please cite our paper:

```BibTeX
@ARTICLE{niresi_pegnn,
  author={Niresi, Keivan Faghih and Bissig, Hugo and Baumann, Henri and Fink, Olga},
  journal={IEEE Internet of Things Journal}, 
  title={Physics-Enhanced Graph Neural Networks For Soft Sensing in Industrial Internet of Things}, 
  year={2024},
  volume={},
  number={},
  doi={10.1109/JIOT.2024.3434732}}
```
