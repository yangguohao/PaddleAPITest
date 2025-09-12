import typer
import csv
from pathlib import Path

app = typer.Typer()

def _get_cases(api_name: str, original_csv: str):
    with (
        open(original_csv, "r") as infile,
        open(f"filtered_result_{api_name}.csv", "w", newline="") as outfile,
    ):
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 写入头部行
        header = next(reader)
        writer.writerow(header)

        # 处理数据行
        for row in reader:
            first_col = row[0]

            if api_name not in first_col:
                continue

            last_col = float(row[-1]) if row[-1].strip() else 0
            second_last_col = float(row[-2]) if row[-2].strip() else 0

            if last_col < 1e-16 and second_last_col < 1e-16:
                continue

            writer.writerow(row)

    with (
        open(f"filtered_result_{api_name}.csv", "r") as infile,
        open(f"error_config_{api_name}.txt", "w") as outfile,
    ):
        reader = csv.reader(infile)

        header = next(reader)

        outs = []
        for row in reader:
            first_col = row[0]
            last_col = float(row[-1]) if row[-1].strip() else 0
            second_last_col = float(row[-2]) if row[-2].strip() else 0

            if last_col < 1e-16 and second_last_col < 1e-16:
                continue
            outs.append(row[2].replace('""', '"'))
        outfile.write("\n".join(outs))


@app.command()
def get_cases(
    api_names: list[str],
    config_path: Path = Path("TotalStableFull.csv"),
):
    for api_name in api_names:
        _get_cases(api_name, config_path.as_posix())



if __name__ == "__main__":
    app()