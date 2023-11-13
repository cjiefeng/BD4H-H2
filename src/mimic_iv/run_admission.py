import preprocessing
from mimic_iv import original_models, uniacs_models

SEED = 99


def main():
    X_train, X_test, y_train, y_test = preprocessing.run(
        dir="../data/mimiciv_sepsis.csv",
        extra_cols=["testtrain", "hadm_id"],
        ohe_cols=[
            "gender",
            "marital_status",
            "ethnicity",
            "insurance",
            "ed_medgp_antibiotic_hrgp",
        ],
        drop_cols=["icu_adm_flag", "hosdeath_flag"],
        label_col="icu_adm_flag",
        seed=SEED,
        test_size=0.7,
    )

    print("original models")
    original_models.run(X_train, X_test, y_train, y_test, SEED)
    print("")
    print("------------------------------------------------------------------")
    print("uniacs models")
    uniacs_models.run(X_train, X_test, y_train, y_test, SEED)
    print("")


if __name__ == "__main__":
    main()
