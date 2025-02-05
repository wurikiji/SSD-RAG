import fire
def main(
    log_path: str
):
  with open(log_path) as f:
    lines = f.readlines()
    elapsed_total = 0.0
    elapsed_cache = 0.0
    for line in lines:
      if "seconds" not in line:
        continue



if __name__ == "__main_"
  fire.Fire(main)