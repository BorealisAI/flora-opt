# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


## Gradient Accumulation (T5-3B)

### Baseline (tuned learning rate)

python examples/flax/encoder_decoder_seq2seq.py optimizer=adafactor optimizer.learning_rate=4e-4 model.pretrained=true model.model_name_or_path=t5-3b grad_acc.steps=16 training.per_device_train_batch_size=1 training.eval_steps=1250

### LoRA (tuned learning rate)
python examples/flax/encoder_decoder_seq2seq.py optimizer=adafactor optimizer.learning_rate=1e-3 model.pretrained=true model.model_name_or_path=t5-3b grad_acc.steps=16 training.per_device_train_batch_size=1 training.eval_steps=1250 lora.disabled=false lora.tune_others=true lora.rank=256

### Flora (reusing the learning rate from the baseline)
python examples/flax/encoder_decoder_seq2seq.py optimizer=adafactor optimizer.learning_rate=4e-4 model.pretrained=true model.model_name_or_path=t5-3b grad_acc.steps=16 training.per_device_train_batch_size=1 training.eval_steps=1250 grad_acc.impl=compressed grad_acc.tau=256


## Momentum (T5-small)
python examples/flax/encoder_decoder_seq2seq.py optimizer=adafactor model.pretrained=false model.model_name_or_path=t5-small optimizer.learning_rate=1e-3  training.per_device_train_batch_size=4 training.eval_steps=200000 optimizer.momentum=0.9  training.num_train_epochs=10

### LoRA (tuned learning rate)
python examples/flax/encoder_decoder_seq2seq.py optimizer=adafactor model.pretrained=false model.model_name_or_path=t5-small optimizer.learning_rate=3e-3  training.per_device_train_batch_size=4 training.eval_steps=200000 training.num_train_epochs=10 optimizer.momentum=0.9 lora.disabled=false lora.tune_others=true lora.rank=256

### Flora (reusing the learning rate from the baseline)
python examples/flax/encoder_decoder_seq2seq.py optimizer=flora model.pretrained=false model.model_name_or_path=t5-small optimizer.learning_rate=1e-3  training.per_device_train_batch_size=4 training.eval_steps=200000 optimizer.b1=0.9  training.num_train_epochs=10 optimizer.tau=256 optimizer.kappa=1000 
