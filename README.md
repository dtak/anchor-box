## Anchor-box

This repo contains the code for our paper, Guarantee Regions for Local Explanations.

### Getting started

The demo reproduces the results in Table 3. To run this script:

Step 1: Have python 3.8.5. Install the packages in `requirements.txt`.

```pip install -r requirements.txt```

Step 2: 

```python oracle.py```

### Notes

Note 1: Feel free to define a custom `b` function in `oracle.py` such as the ones we discuss in the paper.

Note 2: This code benefits a lot form GPU acceleration. The first execution is somewhat slower because tensorflow takes time to trace the computation, but subsequent runs are much faster.
