There should be three files in the data directory to run the scripts in the repo. The two `.dat` files contain the real and approximate consumer models for a set of 100 consumers, with data over a 1 year period. 

1. `consumer_models/real_consumer_models_100.dat` : The real consumer model, with the 4R2C thermal parameters, load data, AC operation model and user behavior model. 
2. `consumer_models/approximate_cons_models_100_3.dat` : The approximate consumer models. This is a list of 3 model objects per consumer, with estimated thermal model (1R1C) and other parameters as described in the paper.
3. `SGP_Singapore.486980_IWEC.csv` : Weather data used for the run.

The data required to run the experiments is available at the following link: [Link to data (DR-NTU repository)](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/XQRYSK)