# Successive Refinement for Implicit Neural Representation (SR-INR)
![Demo GIF](results/image_22/SuRP/result_animation_looped.gif)

To train a model from scratch (using Kodak image 21 for example), run:

```bash
python train.py --num_iters 100000 --layer_size 150 --num_layers 2 --image_id 21
```

To quantize a trained model, run
```bash
python compress.py --image_id 21 --compression_type Quantization --layer_size 150 --num_layers 2 --quant_level half
``` 
Note that only half-precision model is supported right now. Other quantization methods TBD.

To apply magnitude pruning to a model, run
```bash
python compress.py --image_id 21 --compression_type Mag_Pruning --layer_size 150 --num_layers 2  --prune_ratio 0.5 --refine_iter 5000
```
prune_ratio of 0.5 is 50% pruning.

To apply SuRP to a model, run
```bash
python compress.py --image_id 21 --compression_type SuRP --layer_size 150 --num_layers 2 --surp_iter 100000 --image_iter 1000
```
image_iter is the number of iterations per each image synthesis.

For the existing models:
* Image 1 is trained with layer_size of 50 and num_layers of 15 (a deeper net with less weights per layer).
* Image 21 is trained with layer_size of 150 and num_layers of 2 (a shallower net with more weights per layer).
* Image 22 is trained with layer_size of 300 and num_layers of 2 (a shallower net with even more weights per layer).

