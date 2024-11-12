# !bin/bash/
# python train_classifier_skin.py --model resnet
# python train_classifier_skin.py --model vgg --use_synthetic True &&
# python train_classifier_skin.py --model resnet --use_synthetic True
python train_classifier_skin.py --model vgg --use_synthetic True --use_proto True &&
python train_classifier_skin.py --model resnet --use_synthetic True --use_proto True

# FOR EVALUATION
python eval_classifier.py --model resnet
python eval_classifier.py --model resnet --use_synthetic True
python eval_classifier.py --model resnet --use_synthetic True --use_proto True
python eval_classifier.py --model vgg
python eval_classifier.py --model vgg --use_synthetic True
python eval_classifier.py --model vgg --use_synthetic True --use_proto True