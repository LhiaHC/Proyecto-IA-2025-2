"""
Script para crear un dataset de ejemplo con datos sintéticos.
Útil para probar el pipeline sin el dataset real.

Uso:
    python create_sample_data.py
"""
import pandas as pd
import numpy as np
import os

# Configuración
np.random.seed(42)
N_SAMPLES = 1000
OUTPUT_PATH = "data/phishing_email.csv"

# Plantillas de texto para correos legítimos
legitimate_templates = [
    "Dear customer, thank you for your purchase. Your order #{} has been confirmed.",
    "Hi, this is a reminder about your appointment on {}. Please confirm your attendance.",
    "Your monthly statement is ready. Please review your account balance of ${}.",
    "Welcome to our newsletter! Here are this week's top stories and updates.",
    "Your package has been shipped. Tracking number: {}. Expected delivery in 3-5 days.",
    "Meeting scheduled for {} at {}. Please join us in conference room A.",
    "Your subscription has been renewed. Next billing date: {}.",
    "Thank you for contacting support. Ticket #{} has been created.",
    "Important: Your password will expire in {} days. Please update it soon.",
    "Congratulations! You have earned {} loyalty points this month.",
]

# Plantillas de texto para correos de phishing
phishing_templates = [
    "URGENT: Your account will be suspended! Click here immediately to verify: {}",
    "You have won ${}! Claim your prize now by providing your bank details.",
    "Security alert: Unusual activity detected. Reset password here: {}",
    "Your package is held at customs. Pay ${} immediately to release it: {}",
    "FINAL NOTICE: Your account is overdue. Pay ${} now or face legal action.",
    "Congratulations! You are the {}th visitor. Click to claim your iPhone now!",
    "Dear sir/madam, I am a prince from {}. I need your help to transfer ${}.",
    "Your PayPal account has been limited. Verify immediately: {}",
    "You have {} unread messages. Click here to view: {}",
    "TAX REFUND: You are owed ${}. Claim now by providing your SSN and bank info.",
    "Your credit card ending in {} has been charged ${}. If not you, click here: {}",
    "Microsoft Security: Your computer is infected! Call {} immediately.",
]

def generate_email(is_phishing: bool) -> str:
    """Genera un correo sintético."""
    if is_phishing:
        template = np.random.choice(phishing_templates)
        # Rellenar placeholders
        email = template.format(
            np.random.choice(["malicious-link.com", "phishing-site.net", "fake-bank.org"]),
            np.random.randint(100, 10000),
            np.random.randint(1, 1000000),
            np.random.choice(["Nigeria", "Kenya", "UAE"]),
            np.random.choice(["1-800-FAKE", "555-0000"])
        )
    else:
        template = np.random.choice(legitimate_templates)
        # Rellenar placeholders
        email = template.format(
            np.random.randint(1000, 9999),
            "2025-10-20",
            np.random.randint(100, 500),
            np.random.randint(10, 100)
        )

    # Añadir algo de variación
    if np.random.random() > 0.5:
        email = email + " Thank you for your attention."

    return email

# Generar dataset
print(f"Generando {N_SAMPLES} correos de ejemplo...")
data = []

for i in range(N_SAMPLES):
    is_phishing = i >= N_SAMPLES // 2  # 50% phishing, 50% legítimo
    email_text = generate_email(is_phishing)
    label = 1 if is_phishing else 0

    data.append({
        "text_combined": email_text,
        "label": label
    })

# Crear DataFrame y mezclar
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Guardar
os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Dataset creado exitosamente: {OUTPUT_PATH}")
print(f"Total de filas: {len(df)}")
print(f"Distribución de clases:")
print(df["label"].value_counts())
print(f"\nPrimeras 3 filas:")
print(df.head(3))
