# SR-INR

## Code for Successive Refinement for Implicit Neural Representation (SR-INR)

To train a model from scratch (using Kodak image 8 for example), run:

```bash
python train.py --image_id 8 --num_iters 50000 
```

To quantize a trained model, run
```bash
python compress.py --image_id 8 --compression_type Quantization 
``` 
Note that only half-precision model is supported right now. Other quantization methods TBD. \\

To apply magnitude pruning to a model, run
```bash
python compress.py --image_id 8 --compression_type Mag_Pruning --prune_ratio 0.5 --refine_iteration 1000
```
prune_ratio of 0.5 is 50% pruning. \\

To apply SuRP to a model, run
```bash
python compress.py --image_id 8 --compression_type SuRP --surp_iter 50000 --image_iter 1000
```
image_iter is the number of iterations per each image synthesis.
