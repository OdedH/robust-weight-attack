# Versatile-Weight-Attack
The implementatin of *Robust Backdoors via Limited Bit Flipping*, including ours *Real-world Triggered Samples Attack* with comparison to the vanilla *Triggered Samples Attack* from *Versatile Weight Attack
via Flipping Limited Bits*.

## Install 
1. In our experiments we used python 3.10.12.
2. We also provide dependencies that you can use by running the following command:
```shell
pip install -r requirements.txt:
```

## Run Our Code

First set your relevant root of the dataset "config.py" weather its "cifar_root" or "gtsrb_root".
Secondly after you have trained your model, set the path to the model in "config.py" weather its "cifar_model_path" or "gtsrb_model_path".


### Real-World Triggered Samples Attack (RW-TSA)

Running the below command will attack all samples with a trigger (CIFAR-10 validation set) into class 0,
with the parameters chosen in Table 1 of our work. This will run both TSA and RW-TSA and output comparison results into 'results.xlsx'.

```shell
python triggered_samples_attack/RW_TSA.py --target 0
```
You can set "target" to perform TSA with different target class.

For attacking GRSRB with the parameters chosen in Table 1 of our work, run the following command:
```shell
python triggered_samples_attack/RW_TSA.py --target 0 --dataset_type gtsrb --lams1 35
```

For conducting hyper-parameter search, for example the one we did on CIFAR-10, run the following command:
```shell
python triggered_samples_attack/RW_TSA.py --target 0 --ext-max-iters 2000 --k-bits 10 20 30 --lams1 20 35 50 65 80 95 --dataset_type gtsrb --n-aux 256
```

We used the pretrained 8-bit quantized ResNet on CIFAR-10. -> "models/cifar_resnet_quan_8/model.th". We also provide the same model pretrained on GTSRB dataset . -> "models/gtsrb_resnet_quan_8/model.ckpt".
In order to train the model yourself, run the following command:
```shell
python custom_nets/run_basic_baseline.py
```
And then use the checkpoint path you get in "config.py".