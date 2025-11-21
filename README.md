# Diagnosing Non-Intermittent Anomalies in Reinforcement Learning Policy Executions (Short Paper)

This project provides a code used in the experiments of the paper below.

## Requirements
1. Python 3.8.7

## Set Up
1. Clone the repository
```
git clone https://github.com/avi-natan/rldx-sifu.git
```
2. Install the packages specified in the requirements.txt file
```
pip install -r /path/to/requirements.txt
```
3. (OPTIONAL) The project is old, as such some inconsistencies with libraries might be expected.\
If step 2 fails, use the provided google drive link to download a zipped version of a virtual environment:\
https://drive.google.com/drive/folders/1Q0mh6lc0lIF6OWzwOV5ciBuxfl0FiqY3?usp=sharing \
In such a case, directory paths might need to be updated in the following file: \
.idea/misc.xml
4. Unzip the zip file named 'experiment_directories.zip' into the root folder of the project

## Usage
Run main with the script parameter: FILENAME.json \
where FILENAME is the name of an experiment file located in the folder 'experimental inputs' \
Example:
```
e2000_Acrobot-1.json
```

## Citation

If you use this, please cite:

Bibtex:
```
@inproceedings{natan2024diagnosing,
  title={Diagnosing Non-Intermittent Anomalies in Reinforcement Learning Policy Executions (Short Paper)},
  author={Natan, Avraham and Stern, Roni and Kalech, Meir},
  booktitle={35th International Conference on Principles of Diagnosis and Resilient Systems (DX 2024)},
  pages={23--1},
  year={2024},
  organization={Schloss Dagstuhl--Leibniz-Zentrum f{\"u}r Informatik}
}
```

Harvard:
```
Natan, A., Stern, R. and Kalech, M., 2024. Diagnosing Non-Intermittent Anomalies in Reinforcement Learning Policy Executions (Short Paper). In 35th International Conference on Principles of Diagnosis and Resilient Systems (DX 2024) (pp. 23-1). Schloss Dagstuhl–Leibniz-Zentrum für Informatik.
```

## Authors

Avraham Natan, Roni Stern, Meir Kalech

## License

This project is licensed under the CC BY-NC 4.0 License. See the [LICENSE](./LICENSE) file for details.