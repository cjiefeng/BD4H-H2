# BD4H-H2

## Getting started
Create virtual environment:
``` 
$ python3 -m venv venv
```

Activate virtual environment:
``` 
$ source venv/bin/activate
```

Install all dependencies:
``` 
$ pip install -r requirements.txt
```

## Run pipelines
Run with mimic-iii dataset
```
$ cd src && PYTHONPATH=../:. python3 mimic_iii/run_mortality.py
```

Run with mimic-iivdataset (mortality)
```
$ cd src && PYTHONPATH=../:. python3 mimic_iv/run_mortality.py
```

Run with mimic-iv dataset (icu admission)
```
$ cd src && PYTHONPATH=../:. python3 mimic_iv/run_admission.py
```

## Directory structure
```
.
├── data/                           # all datasets
├── notebook/                       # experimental and exploration jupyter notebooks
├── README.md
├── requirements.txt
├── src
│   ├── medical_explainer.py        # uniacs explainer
│   ├── mimic_iv                    # pipelines specific to mimic-iv dataset
│   │   ├── original_models.py      # models without uniacs
│   │   ├── run_admission.py        # run models with icu_admission as label
│   │   ├── run_mortality.py        # run models with mortality as label
│   │   └── uniacs_models.py        # models with uniacs
│   ├── preprocessing.py            # preprocessing & train test split pipeline 
│   └── utils
│       └── reports.py              # report model metrics
└── submissions/                    # all BD4H report submissions
```
