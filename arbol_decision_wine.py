from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Cargar el dataset del vino
wine = load_wine()
X, y = wine.data, wine.target

# 2. Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# 3. Crear y entrenar el árbol con profundidad máxima de 2
tree_limited = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

tree_limited.fit(X_train, y_train)

# Evaluar el modelo limitado
y_pred_limited = tree_limited.predict(X_test)
acc_limited = accuracy_score(y_test, y_pred_limited)

print("=== Modelo con max_depth=2 ===")
print(f"Precisión en datos de prueba: {acc_limited * 100:.2f}%\n")

print("Reglas aprendidas:")
print(export_text(
    tree_limited,
    feature_names=wine.feature_names
))

# 4. Crear y entrenar el árbol sin limitar la profundidad
tree_full = DecisionTreeClassifier(
    max_depth=None,
    random_state=42
)

tree_full.fit(X_train, y_train)

# Evaluar el modelo completo
y_pred_full = tree_full.predict(X_test)
acc_full = accuracy_score(y_test, y_pred_full)

print("\n=== Modelo sin límite de profundidad (max_depth=None) ===")
print(f"Precisión en datos de prueba: {acc_full * 100:.2f}%\n")

print("Reglas aprendidas:")
print(export_text(
    tree_full,
    feature_names=wine.feature_names
))