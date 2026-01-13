```
# cd fedRBE
cd /home/yuliya-cosybio/repos/cosybio/fedRBE

featurecloud app build ./evaluation_clusterization_after_correction/federated_kmeans_upd fc_kmeans_upd
```

Rotation BE:
```
# balanced
cd /home/yuliya-cosybio/repos/cosybio/fedRBE && \
featurecloud controller stop && \
cd ./evaluation_data/simulated_rotation/balanced/before && \
featurecloud controller start --data-dir="$PWD" && \
featurecloud test start --app-image=fc_kmeans_upd --client-dirs=lab1,lab2,lab3

# wait 2 minutes until the end of the workflow
cp -a ./tests/results_test_1* ../after/fc_kmeans_res/
...
sudo rm -rf ./tests ./logs ./workflows
sudo mkdir -p ./tests ./logs ./workflows

# mild_imbalanced
cd /home/yuliya-cosybio/repos/cosybio/fedRBE && \
featurecloud controller stop && \
cd ./evaluation_data/simulated_rotation/mild_imbalanced/before && \
featurecloud controller start --data-dir="$PWD" && \
featurecloud test start --app-image=fc_kmeans_upd --client-dirs=lab1,lab2,lab3

# wait 2 minutes until the end of the workflow
cp -a ./tests/results_test_1* ../after/fc_kmeans_res/
...
sudo rm -rf ./tests ./logs ./workflows
sudo mkdir -p ./tests ./logs ./workflows

# strong_imbalanced
cd /home/yuliya-cosybio/repos/cosybio/fedRBE && \
featurecloud controller stop && \
cd ./evaluation_data/simulated_rotation/strong_imbalanced/before && \
featurecloud controller start --data-dir="$PWD" && \
featurecloud test start --app-image=fc_kmeans_upd --client-dirs=lab1,lab2,lab3

# wait 2 minutes until the end of the workflow
cp -a ./tests/results_test_1* ../after/fc_kmeans_res/
...
sudo rm -rf ./tests ./logs ./workflows
sudo mkdir -p ./tests ./logs ./workflows


```

# After correction evaluation


```

# balanced
cd /home/yuliya-cosybio/repos/cosybio/fedRBE && \
featurecloud controller stop && \
cd ./evaluation_data/simulated_rotation/balanced/before_corrected/ && \
featurecloud controller start --data-dir="$PWD" && \
featurecloud test start --app-image=fc_kmeans_upd --client-dirs=lab1,lab2,lab3

# wait 2 minutes until the end of the workflow
cp -a ./tests/results_test_1* ../before_corrected/fc_kmeans_res/
...
sudo rm -rf ./tests ./logs ./workflows
sudo mkdir -p ./tests ./logs ./workflows

# mild_imbalanced
cd /home/yuliya-cosybio/repos/cosybio/fedRBE && \
featurecloud controller stop && \
cd ./evaluation_data/simulated_rotation/mild_imbalanced/before_corrected/ && \
featurecloud controller start --data-dir="$PWD" && \
featurecloud test start --app-image=fc_kmeans_upd --client-dirs=lab1,lab2,lab3

# wait 2 minutes until the end of the workflow
cp -a ./tests/results_test_1* ../before_corrected/fc_kmeans_res/
...
sudo rm -rf ./tests ./logs ./workflows
sudo mkdir -p ./tests ./logs ./workflows

# strong_imbalanced
cd /home/yuliya-cosybio/repos/cosybio/fedRBE && \
featurecloud controller stop && \
cd ./evaluation_data/simulated_rotation/strong_imbalanced/before_corrected/ && \
featurecloud controller start --data-dir="$PWD" && \
featurecloud test start --app-image=fc_kmeans_upd --client-dirs=lab1,lab2,lab3

# wait 2 minutes until the end of the workflow
cp -a ./tests/results_test_1* ../before_corrected/fc_kmeans_res/
...
sudo rm -rf ./tests ./logs ./workflows
sudo mkdir -p ./tests ./logs ./workflows





```
# run evaluation_clusterization_after_correction/01_clustering.ipynb
```