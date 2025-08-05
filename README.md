# Imperial25.Clustering
TBD

### Left to do

1. Run the pipeline with 3-5 datasets, and see what the Lambda and Bandwidths on DIBmix return with. Then set a value accordingly. (This will have to be done by Efthymios).
2. Run the investigation with all preferred 12 real life datasets.
3. Create visualisations for the results and the poster and handle results.
4. Write up report, design poster. 
    On the report, datasets section nearly done.
    Must also add, restraints and problems that we adapted around (categorical, DIB bug, auto categorical detector?)
    Then results and conclusion, this goes hand-in-hand with 3.
    Oh, and the bibliography.


- I have the datasets, just need to do metadata, and remove automatic type checking.
- I've done the metadata, now its just time to tune the scripts so that I can send it off to Efthymios, I have to double check that the MICE missingness fix (I.e. after imputation, using mode/mean) is a valid method. It might ruin RMSE.