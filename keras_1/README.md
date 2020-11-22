#Simple MLFlow and Keras

This is just an example to run a simple Fully connected NN on the MNIST database
and log the metrics into MLFlow


The ```MLFLOW_TRACKING_URI``` environmental variable should be set if a remote MLFlow tracking instance is used

```export MLFLOW_TRACKING_URI=http://<host>:<port>```


In my case I have an MLFlow server running on my local machine, with a postgresql db as a backend store and a
Azure Blob Storage location as the artifact store  

The script will create a plot with the training loss and accuracy on a png file. This plot (or artifact) together
with the model are logged to an Azure Blog Storage.   

See [this](https://mlflow.org/docs/latest/tracking.html#azure-blob-storage) for details about configuring an Azure Blob
Storage as an MLFLow artifact root


