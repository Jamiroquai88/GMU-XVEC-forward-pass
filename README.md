# X-Vector-forward-pass-OpenCL
Forward pass of X-Vector based architecture on GPU using OpenCL API

## Compile
```bash
make
```

## Example run using pretrained net in Kaldi nnet3 format
```bash
./main -i fea/example.txt \
  -n $nn_path \
  -o output.txt
