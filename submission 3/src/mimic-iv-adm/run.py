import preprocessing


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
    )


if __name__ == "__main__":
    main()
