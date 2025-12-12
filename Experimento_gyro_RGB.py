import subprocess

EXPERIMENTS = [
    "./exps/finetune_gyro_rgb.json",
    "./exps/icarl_gyro_rgb.json",
    "./exps/wa_gyro_rgb.json",
    "./exps/der_gyro_rgb.json"
]

def run_experiment(config):
    print(f"\n=== Entrenando con {config} ===")
    cmd = ["python", "main.py", "--config", config]
    subprocess.run(cmd)

if __name__ == "__main__":
    for exp in EXPERIMENTS:
        run_experiment(exp)
