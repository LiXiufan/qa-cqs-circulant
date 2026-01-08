# qa-cqs-circulant
This repository is for numerical experiments of the algorithm solving banded circulant linear systems. The algorithm is developed based on a concept of [classical combination of quantum states (CQS)](https://iopscience.iop.org/article/10.1088/1367-2630/ac325f) and with the technique of [quantum Fourier transformation (QFT)](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview). We use the one-dimensional heat transfer problem as an example of our algorithm. The advantage of the algorithm is that, given a K-banded circulant matrix and the condition number k_c, it has provable performance guarantees for the truncated error with a threshold polynomial to O(K k_c log(k_c)).

For feedback, please contact Xiufan at: e1117166@u.nus.edu.

## New Release! Our paper has been accepted by _New Journal of Physics_, 2025, Dec. 30. Great thanks to @georgepwhuang for huge contributions!

For BibTeX citations, please copy the following notes:

@article{10.1088/1367-2630/ae3205,
	author={Huang, Po-Wei and Li, Xiufan and Koor, Kelvin and Rebentrost, Frank Patrick},
	title={Classical combinations of quantum states for solving banded circulant linear systems},
	journal={New Journal of Physics},
	url={http://iopscience.iop.org/article/10.1088/1367-2630/ae3205},
	year={2025}
}
 
