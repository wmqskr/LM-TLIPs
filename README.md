LM-TLIPs
===
LMTLIPs uses the most advanced large model technology ESM-2 to extract S/ sequence and Y sequence information. On the basis of fine-tuning the large model, transfer learning technology is introduced to solve the problem that Y site is difficult to predict accurately due to insufficient data. Based on experimental validation independently tested on the S/T dataset as well as the Y-site dataset, LM-TLIPs outperforms existing optimal prediction tools, demonstrating its superior ability to identify phosphorylation sites.

Requirement
===
*Python 3.9
*Pytorch 2.1.0+cu121
*Transformers 4.42.3
*Numpy 1.26.4

How to run
===
*ST_code_ESM: fine tune ESM-2
*ST_code_ml:train machine learning
*ST_intergration: intergrate multiple predict probability
*Y_code_ESM: fine tune ESM-2
*Y_code_ml:train machine learning
*Y_intergration: intergrate multiple predict probability

