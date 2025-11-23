import subprocess
import os
import time

# Lista de experimentos faltantes
experiments = [
    # FastText faltantes
    ("fasttext", 100, "svm"),
    ("fasttext", 100, "rf"),
    ("fasttext", 200, "lr"),
    ("fasttext", 200, "svm"),
    ("fasttext", 200, "rf"),
    ("fasttext", 300, "lr"),
    ("fasttext", 300, "svm"),

    # BERT faltantes
    ("bert", 100, "lr"),
    ("bert", 100, "svm"),
    ("bert", 100, "rf"),
    ("bert", 200, "lr"),
    ("bert", 200, "svm"),
    ("bert", 200, "rf"),
    ("bert", 300, "lr"),
    ("bert", 300, "svm"),
    ("bert", 300, "rf"),
    ("bert", 768, "svm"),
    ("bert", 768, "rf"),
]

python_exe = r"iaa_venv\Scripts\python.exe"
total = len(experiments)

print(f"="*60)
print(f"EJECUTANDO {total} EXPERIMENTOS RESTANTES")
print(f"="*60)

for i, (emb, dim, clf) in enumerate(experiments, 1):
    print(f"\n[{i}/{total}] Ejecutando {emb}-{dim}-{clf}...")
    start_time = time.time()

    cmd = [
        python_exe,
        "src/experiment_runner.py",
        "--embedding", emb,
        "--dim", str(dim),
        "--classifier", clf
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"[OK] Completado en {elapsed:.1f}s")
            # Extraer F1 del output
            for line in result.stdout.split('\n'):
                if 'F1' in line and ':' in line:
                    print(f"  {line.strip()}")
                    break
        else:
            print(f"[ERROR] Codigo {result.returncode}")
            print(result.stderr[:200])
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Despues de 30 minutos")
    except Exception as e:
        print(f"[EXCEPCION] {e}")

print(f"\n{'='*60}")
print("TODOS LOS EXPERIMENTOS COMPLETADOS")
print(f"{'='*60}")

# Contar archivos de resultados
result_files = len([f for f in os.listdir('reports') if f.startswith('results_') and f.endswith('.csv')])
print(f"\nTotal de resultados: {result_files}/30")
