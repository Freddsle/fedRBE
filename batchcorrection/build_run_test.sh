docker build . --tag=batchcorrection:latest --no-cache

featurecloud test start --client-dirs ./client1,./client2,./client3,./client4 --generic-dir ./generic --app-image batchcorrection --channel "local" --query-interval 3.0 --print-logs
