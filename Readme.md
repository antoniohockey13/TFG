## TFG - Estudio de algoritmos de reconstrucción de vértices usando información del detector "MIPs Timing Detector" del Gran Colisionador de Hadrones


### Part 1-Simulated Data Simulados Artificialy
In this part data has been simulated with the script Utilities_Functions/GenerarConjuntoVerticesyTrazas.py

To run this part use the Aleatorio_\*.py scripts depending on what algorithm gonna be used (\* is the name of the algorithm).

The script Aleatorio_AjustarAlgoritmos.py is used to plot the evolution of the results depending on the initial parameters used for each algorithm.

### Part 2-CMS Simulated Data
In this part it has been used simulated data within the officil CMS requirement. To acces this data go into the folder Data. This data is read with the script Utilities_Functions/read_data. 

To run this part use the CMS_\*.py scripts depending on what algorithm gonna be used (\* is the name of the algorithm).

The script Cluster_CMS_data.py read only one data and compute the results from all the algorithm.

### Evaluate Results
To evluate the results the script Utilities_Functions/evaluar.py is used. The clustertovertex function is the one which assign each cluster to each simulated vertex. 
