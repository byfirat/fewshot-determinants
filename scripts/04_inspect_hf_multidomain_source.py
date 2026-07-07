from fewshot_determinants.paths import CONFIGS_DIR
from fewshot_determinants.utils.io import load_yaml
from fewshot_determinants.data.hf_source import available_domains, hf_to_dataframe, load_hf_split


def main():
    cfg = load_yaml(CONFIGS_DIR / "data.yaml")
    source_cfg = cfg["source"]
    split = source_cfg.get("inspect_split", "train")
    inspect_n = int(source_cfg.get("inspect_sample_size", 5000))

    hf_ds = load_hf_split(
        dataset_name=source_cfg["dataset_name"],
        dataset_config=source_cfg["dataset_config"],
        split=f"{split}[:{inspect_n}]",
    )
    df = hf_to_dataframe(hf_ds)

    print("Columns:")
    print(list(df.columns))
    print("\nTop domain counts in inspection sample:")
    print(available_domains(df).head(20).to_string())


if __name__ == "__main__":
    main()
